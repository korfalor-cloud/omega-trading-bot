"""
CandleStick Transformer — Kaggle GPU Training Notebook
Copy-paste this into a Kaggle notebook and run all cells.

Setup:
  1. Go to kaggle.com → Create → New Notebook
  2. Click "Settings" (right panel) → Accelerator → GPU T4
  3. Paste this code into cells (split by # === CELL ===)
  4. Click "Run All"
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install requests torch numpy

# ============================================================
# CELL 2: All source code in one cell (for Kaggle notebook)
# ============================================================
import math
import json
import time
import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# ─── CONFIG ───

@dataclass
class TokenizerConfig:
    PRICE_BIN_COUNT: int = 1024
    PRICE_CHANGE_RANGE: float = 0.10
    VOLUME_BIN_COUNT: int = 256
    PATTERN_COUNT: int = 128
    PATTERN_OFFSET: int = 0
    TIMEFRAME_COUNT: int = 16
    TIMEFRAME_OFFSET: int = 128
    INDICATOR_BIN_COUNT: int = 256
    INDICATOR_OFFSET: int = 144
    PRICE_OFFSET: int = 400
    PAD_TOKEN: int = 0
    BOS_TOKEN: int = 1
    EOS_TOKEN: int = 2
    MASK_TOKEN: int = 3
    SEP_TOKEN: int = 4
    SPECIAL_COUNT: int = 16
    TOKENS_PER_CANDLE: int = 8

    @property
    def vocab_size(self):
        return (self.SPECIAL_COUNT + self.TIMEFRAME_COUNT + self.PATTERN_COUNT
                + self.INDICATOR_BIN_COUNT + self.PRICE_BIN_COUNT * 3 + self.VOLUME_BIN_COUNT)


@dataclass
class ModelConfig:
    VOCAB_SIZE: int = 3744
    D_MODEL: int = 512
    N_HEADS: int = 8
    N_LAYERS: int = 12
    D_FF: int = 2048
    DROPOUT: float = 0.1
    MAX_SEQ_LEN: int = 512
    LOCAL_WINDOW: int = 10
    MAX_RELATIVE_POSITION: int = 128
    SIGNAL_CLASSES: int = 3
    CONFIDENCE_DIM: int = 1


@dataclass
class TrainingConfig:
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.01
    BETA1: float = 0.9
    BETA2: float = 0.95
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    GRAD_CLIP: float = 1.0
    WARMUP_STEPS: int = 1000
    MIN_LR: float = 1e-5
    ALPHA_CANDLE: float = 1.0
    BETA_SIGNAL: float = 2.0
    GAMMA_CONFIDENCE: float = 0.5
    WINDOW_SIZE: int = 64
    STRIDE: int = 1
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.1
    CHECKPOINT_DIR: str = "checkpoints"
    SAVE_EVERY: int = 10
    LOG_EVERY: int = 100
    USE_FP16: bool = True


# ─── PATTERNS ───

class PatternID(IntEnum):
    NONE = 0
    DOJI = 16; LONG_LEGGED_DOJI = 17; DRAGONFLY_DOJI = 18; GRAVESTONE_DOJI = 19
    HAMMER = 20; INVERTED_HAMMER = 21; SHOOTING_STAR = 22; HANGING_MAN = 23
    MARUBOZU_BULL = 24; MARUBOZU_BEAR = 25; SPINNING_TOP = 26
    BELT_HOLD_BULL = 27; BELT_HOLD_BEAR = 28
    ENGULFING_BULL = 32; ENGULFING_BEAR = 33; HARAMI_BULL = 34; HARAMI_BEAR = 35
    PIERCING_LINE = 36; DARK_CLOUD = 37; TWEEZER_TOP = 38; TWEEZER_BOTTOM = 39
    KICKING_BULL = 40; KICKING_BEAR = 41
    MORNING_STAR = 48; EVENING_STAR = 49
    THREE_WHITE_SOLDIERS = 50; THREE_BLACK_CROWS = 51
    THREE_INSIDE_UP = 52; THREE_INSIDE_DOWN = 53
    ABANDONED_BABY_BULL = 56; ABANDONED_BABY_BEAR = 57
    COUNT = 64


def _body(c): return abs(c[3] - c[0])
def _range(c): return c[1] - c[2]
def _upper_shadow(c): return c[1] - max(c[0], c[3])
def _lower_shadow(c): return min(c[0], c[3]) - c[2]
def _is_bullish(c): return c[3] > c[0]
def _is_bearish(c): return c[3] < c[0]
def _body_midpoint(c): return (c[0] + c[3]) / 2.0

def detect_doji(c, threshold=0.05):
    r = _range(c)
    return _body(c) / r < threshold if r > 0 else True

def detect_hammer(c):
    b = _body(c)
    return b > 0 and _lower_shadow(c) >= 2.0 * b and _upper_shadow(c) < b * 0.5

def detect_marubozu_bull(c):
    r = _range(c)
    return _is_bullish(c) and r > 0 and _body(c) / r > 0.9

def detect_marubozu_bear(c):
    r = _range(c)
    return _is_bearish(c) and r > 0 and _body(c) / r > 0.9

def detect_engulfing_bull(candles):
    if len(candles) < 2: return False
    p, c = candles[-2], candles[-1]
    return _is_bearish(p) and _is_bullish(c) and c[0] <= p[3] and c[3] >= p[0]

def detect_engulfing_bear(candles):
    if len(candles) < 2: return False
    p, c = candles[-2], candles[-1]
    return _is_bullish(p) and _is_bearish(c) and c[0] >= p[3] and c[3] <= p[0]

def detect_morning_star(candles):
    if len(candles) < 3: return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    return (_is_bearish(c1) and _is_bullish(c3) and _body(c2) < _body(c1) * 0.3 and c3[3] > _body_midpoint(c1))

def detect_evening_star(candles):
    if len(candles) < 3: return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    return (_is_bullish(c1) and _is_bearish(c3) and _body(c2) < _body(c1) * 0.3 and c3[3] < _body_midpoint(c1))

PATTERN_REGISTRY = [
    (16, detect_doji, 1), (20, detect_hammer, 1),
    (24, detect_marubozu_bull, 1), (25, detect_marubozu_bear, 1),
    (32, detect_engulfing_bull, 2), (33, detect_engulfing_bear, 2),
    (48, detect_morning_star, 3), (49, detect_evening_star, 3),
]

def detect_patterns(candles):
    detected = []
    for pid, det, n in PATTERN_REGISTRY:
        if len(candles) >= n:
            try:
                w = candles[-n:]
                if det(w) if n > 1 else det(w[0]):
                    detected.append(pid)
            except: pass
    return detected

def detect_pattern_for_candle(candles):
    p = detect_patterns(candles)
    return max(p) if p else 0


# ─── TOKENIZER ───

class CandleStickTokenizer:
    def __init__(self, config=None):
        self.config = config or TokenizerConfig()
        self.timeframe_map = {
            "1m": 128, "5m": 130, "15m": 131, "1h": 133, "4h": 135,
            "1d": 139, "1w": 141, "1M": 142,
        }

    @property
    def vocab_size(self): return self.config.vocab_size
    @property
    def tokens_per_candle(self): return self.config.TOKENS_PER_CANDLE
    @property
    def pad_token(self): return self.config.PAD_TOKEN
    @property
    def bos_token(self): return self.config.BOS_TOKEN
    @property
    def eos_token(self): return self.config.EOS_TOKEN

    def _quantize_price(self, pct, n_bins):
        rng = self.config.PRICE_CHANGE_RANGE
        c = max(-rng, min(rng, pct))
        return int((c + rng) / (2 * rng) * (n_bins - 1))

    def _quantize_volume(self, vol, ref):
        if ref <= 0 or vol <= 0: return 0
        ratio = vol / ref
        log_r = math.log10(max(0.1, min(10.0, ratio)))
        return int((log_r + 1.0) / 2.0 * 255)

    def _compute_indicators(self, candles, idx):
        offset = self.config.INDICATOR_OFFSET
        lookback = min(14, idx + 1)
        closes = candles[max(0, idx-lookback+1):idx+1, 3]
        if len(closes) < 2:
            return offset + 128, offset + 128
        diffs = np.diff(closes)
        up = np.sum(diffs[diffs > 0])
        dn = -np.sum(diffs[diffs < 0])
        rsi = int((up / (up + dn)) * 255) if (up + dn) > 0 else 128
        bb_lookback = min(20, idx + 1)
        bb_closes = candles[max(0, idx-bb_lookback+1):idx+1, 3]
        mean, std = np.mean(bb_closes), np.std(bb_closes)
        z = (closes[-1] - mean) / (2 * std) if std > 0 else 0
        bb = int((max(-1, min(1, z)) + 1) / 2 * 255)
        return offset + max(0, min(255, rsi)), offset + max(0, min(255, bb))

    def tokenize_candle(self, candle, all_candles, idx, timeframe="1h", ref_vol=None):
        o, h, l, c, v = candle[:5]
        tf = self.timeframe_map.get(timeframe, 133)
        if o == 0: cc, hc, lc = 0, 0, 0
        else: cc, hc, lc = (c-o)/o, (h-o)/o, (l-o)/o
        cb = self.config.PRICE_OFFSET + self._quantize_price(cc, 1024)
        hb = self.config.PRICE_OFFSET + 1024 + self._quantize_price(hc, 1024)
        lb = self.config.PRICE_OFFSET + 2048 + self._quantize_price(lc, 1024)
        if ref_vol is None: ref_vol = v if v > 0 else 1.0
        vb = 16 + 128 + 128 + self._quantize_volume(v, ref_vol)
        hist = all_candles[:idx+1]
        pt = self.config.PATTERN_OFFSET + detect_pattern_for_candle(hist)
        i1, i2 = self._compute_indicators(all_candles, idx)
        return [tf, cb, hb, lb, vb, pt, i1, i2]

    def tokenize_sequence(self, candles, timeframe="1h", add_bos=True, add_eos=False):
        if len(candles) == 0: return [self.bos_token] if add_bos else []
        ref = float(np.mean(candles[:, 4])) if np.any(candles[:, 4] > 0) else 1.0
        tokens = []
        if add_bos: tokens.append(self.bos_token)
        for i in range(len(candles)):
            tokens.extend(self.tokenize_candle(candles[i], candles, i, timeframe, ref))
        if add_eos: tokens.append(self.eos_token)
        return tokens

    def pad_sequence(self, tokens, max_len):
        if len(tokens) >= max_len: return tokens[:max_len]
        return tokens + [self.pad_token] * (max_len - len(tokens))

    def decode_price_bins(self, cb, hb, lb, op):
        rng, n = self.config.PRICE_CHANGE_RANGE, 1024
        def bp(b, off): return (b - off) / (n-1) * 2 * rng - rng
        return (op*(1+bp(cb, self.config.PRICE_OFFSET)),
                op*(1+bp(hb, self.config.PRICE_OFFSET+1024)),
                op*(1+bp(lb, self.config.PRICE_OFFSET+2048)))


# ─── MODEL ───

class RelativePositionBias(nn.Module):
    def __init__(self, max_rel, n_heads):
        super().__init__()
        self.max_rel = max_rel
        self.bias_table = nn.Parameter(torch.zeros(2*max_rel+1, n_heads))
        nn.init.xavier_uniform_(self.bias_table)

    def forward(self, seq_len):
        pos = torch.arange(seq_len, device=self.bias_table.device)
        rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(-self.max_rel, self.max_rel)
        return self.bias_table[rel + self.max_rel].permute(2, 0, 1)


class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, n_heads, local_window, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_window = local_window
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_bias=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Global attention
        ag = (q @ k.transpose(-2, -1)) / self.scale
        if rel_bias is not None: ag = ag + rel_bias.unsqueeze(0)
        causal = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
        ag = ag.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        og = self.dropout(F.softmax(ag, dim=-1)) @ v

        # Local attention
        al = (q @ k.transpose(-2, -1)) / self.scale
        if rel_bias is not None: al = al + rel_bias.unsqueeze(0)
        local = torch.ones(L, L, device=x.device).bool()
        for i in range(L):
            local[i, max(0, i-self.local_window+1):i+1] = False
        al = al.masked_fill((causal | local).unsqueeze(0).unsqueeze(0), float("-inf"))
        ol = self.dropout(F.softmax(al, dim=-1)) @ v

        ogf = og.transpose(1, 2).contiguous().view(B, L, D)
        olf = ol.transpose(1, 2).contiguous().view(B, L, D)
        g = self.gate(torch.cat([ogf, olf], -1))
        return self.out_proj(g * ogf + (1 - g) * olf)


class CandleAwareLN(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.vol_scale = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
        self.vol_w = nn.Parameter(torch.zeros(1))

    def forward(self, x, vol=None):
        m, v = x.mean(-1, keepdim=True), x.var(-1, keepdim=True, unbiased=False)
        out = self.gamma * (x - m) / torch.sqrt(v + self.eps) + self.beta
        if vol is not None:
            out = out + self.vol_w * self.vol_scale(vol)
        return out


class CandleBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiScaleAttention(cfg.D_MODEL, cfg.N_HEADS, cfg.LOCAL_WINDOW, cfg.DROPOUT)
        self.n1 = CandleAwareLN(cfg.D_MODEL)
        self.n2 = CandleAwareLN(cfg.D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.D_MODEL, cfg.D_FF), nn.GELU(),
            nn.Dropout(cfg.DROPOUT), nn.Linear(cfg.D_FF, cfg.D_MODEL), nn.Dropout(cfg.DROPOUT))
        self.rel = RelativePositionBias(cfg.MAX_RELATIVE_POSITION, cfg.N_HEADS)

    def forward(self, x, vol=None):
        L = x.size(1)
        x = x + self.attn(self.n1(x, vol), self.rel(L))
        x = x + self.ffn(self.n2(x, vol))
        return x


class CandleTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.VOCAB_SIZE, cfg.D_MODEL)
        self.drop = nn.Dropout(cfg.DROPOUT)
        self.pos = nn.Parameter(torch.zeros(1, cfg.MAX_SEQ_LEN, cfg.D_MODEL))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([CandleBlock(cfg) for _ in range(cfg.N_LAYERS)])
        self.norm = nn.LayerNorm(cfg.D_MODEL)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); (m.bias is not None and nn.init.zeros_(m.bias))
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def _volatility(self, ids):
        B, L = ids.shape
        tpc = 8
        vol = torch.zeros(B, L, 1, device=ids.device)
        for ci in range(2, L, tpc):
            if ci + 1 < L:
                hi = ids[:, ci].float()
                lo = ids[:, ci+1].float()
                v = (hi - lo).abs()
                end = min(ci + tpc, L)
                vol[:, ci:end, 0] = v.unsqueeze(1)
        return vol

    def forward(self, ids, mask=None):
        B, L = ids.shape
        x = self.tok_emb(ids) + self.pos[:, :L]
        x = self.drop(x)
        vol = self._volatility(ids)
        if mask is not None: x = x * mask.unsqueeze(-1)
        for layer in self.layers: x = layer(x, vol)
        x = self.norm(x)
        if mask is not None: x = x * mask.unsqueeze(-1)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── HEADS ───

class NextCandleHead(nn.Module):
    def __init__(self, d_model, vocab_size, tpc=8):
        super().__init__()
        self.tpc = tpc
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, 4, d_model*2, 0.1, batch_first=True), 2)
        self.proj = nn.Linear(d_model, vocab_size)
        self.pos = nn.Parameter(torch.zeros(1, tpc, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, h):
        B = h.size(0)
        mem = h[:, -16:]
        q = self.pos.expand(B, -1, -1)
        return self.proj(self.decoder(q, mem))


class TradeSignalHead(nn.Module):
    def __init__(self, d_model, n_cls=3):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(1))
        self.clf = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(d_model//2, n_cls))

    def forward(self, h):
        x = h[:, -8:]
        w = self.attn(x)
        return self.clf((x * w).sum(1))


class ConfidenceHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(1))
        self.scorer = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(d_model//2, d_model//4), nn.GELU(),
                                    nn.Linear(d_model//4, 1), nn.Sigmoid())

    def forward(self, h):
        x = h[:, -8:]
        w = self.attn(x)
        return self.scorer((x * w).sum(1))


class TradingHeads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.candle = NextCandleHead(cfg.D_MODEL, cfg.VOCAB_SIZE)
        self.signal = TradeSignalHead(cfg.D_MODEL)
        self.conf = ConfidenceHead(cfg.D_MODEL)

    def forward(self, h):
        return {"next_candle": self.candle(h), "trade_signal": self.signal(h), "confidence": self.conf(h)}


class TradingLoss(nn.Module):
    def __init__(self, a=1.0, b=2.0, g=0.5):
        super().__init__()
        self.a, self.b, self.g = a, b, g
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.se = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, tgt):
        cl = self.ce(pred["next_candle"].reshape(-1, pred["next_candle"].size(-1)), tgt["next_candle"].reshape(-1))
        sl = self.se(pred["trade_signal"], tgt["trade_signal"])
        conf = self.mse(pred["confidence"], tgt["confidence"])
        return {"total_loss": self.a*cl + self.b*sl + self.g*conf, "candle_loss": cl, "signal_loss": sl, "confidence_loss": conf}


# ─── DATA ───

class BinanceFetcher:
    def __init__(self):
        self.base = "https://api.binance.com/api/v3"

    def fetch_klines(self, symbol="BTCUSDT", interval="1h", limit=1000, start_time=None, end_time=None):
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start_time: params["startTime"] = start_time
        if end_time: params["endTime"] = end_time
        r = requests.get(f"{self.base}/klines", params=params, timeout=30)
        r.raise_for_status()
        return np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[0])] for k in r.json()], dtype=np.float64)

    def fetch_all(self, symbol="BTCUSDT", interval="1h", days=365):
        from datetime import datetime, timedelta
        end = int(datetime.now().timestamp() * 1000)
        start = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        all_c, cur = [], start
        while cur < end:
            c = self.fetch_klines(symbol, interval, 1000, cur, end)
            if len(c) == 0: break
            all_c.append(c)
            cur = int(c[-1, 5]) + 1
            time.sleep(0.2)
        return np.vstack(all_c) if all_c else np.zeros((0, 6))


def create_split(candles, val=0.15, test=0.1):
    n = len(candles)
    ts = int(n * (1 - test))
    vs = int(n * (1 - test - val))
    return candles[:vs], candles[vs:ts], candles[ts:]


class CandleDataset(Dataset):
    def __init__(self, candles, tokenizer, cfg=None, timeframe="1h", augment=False):
        self.candles = candles
        self.tok = tokenizer
        self.cfg = cfg or TrainingConfig()
        self.tf = timeframe
        self.augment = augment
        self.tpc = tokenizer.tokens_per_candle
        self.ws = min(self.cfg.WINDOW_SIZE, (510) // self.tpc)
        self.max_len = self.tpc * self.ws + 2
        self.stride = self.cfg.STRIDE
        self.indices = list(range(0, max(0, len(candles) - self.ws - 1), self.stride))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.ws
        ic = self.candles[s:e].copy()
        if self.augment:
            for j in range(4):
                ic[:, j] += np.random.normal(0, 0.001, len(ic)) * np.abs(ic[:, j])
            ic[:, 1] = np.maximum(ic[:, 1], np.maximum(ic[:, 0], ic[:, 3]))
            ic[:, 2] = np.minimum(ic[:, 2], np.minimum(ic[:, 0], ic[:, 3]))
            ic[:, :5] = np.maximum(ic[:, :5], 0)

        tokens = self.tok.tokenize_sequence(ic, self.tf, True, False)
        ids = self.tok.pad_sequence(tokens, self.max_len)
        mask = [1]*min(len(tokens), self.max_len) + [0]*max(0, self.max_len-len(tokens))
        mask = mask[:self.max_len]

        ti = e
        if ti < len(self.candles):
            all_up = self.candles[:ti+1]
            ref = float(np.mean(all_up[:, 4])) if np.any(all_up[:, 4] > 0) else 1.0
            tt = self.tok.tokenize_candle(self.candles[ti], all_up, ti, self.tf, ref)
            o, c = self.candles[ti, 0], self.candles[ti, 3]
            pc = (c-o)/o if o != 0 else 0
            sig = 0 if pc > 0.005 else (1 if pc < -0.005 else 2)
            conf = min(1.0, abs(pc)/0.05)
        else:
            tt, sig, conf = [0]*8, 2, 0.0

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "target_candle": torch.tensor(tt, dtype=torch.long),
            "target_signal": torch.tensor(sig, dtype=torch.long),
            "target_confidence": torch.tensor([conf], dtype=torch.float32),
        }


# ─── TRAINER ───

class CosineWarmup:
    def __init__(self, opt, warmup, total, min_lr=1e-5):
        self.opt, self.warmup, self.total, self.min_lr = opt, warmup, total, min_lr
        self.base = [pg["lr"] for pg in opt.param_groups]
        self.step_n = 0

    def step(self):
        self.step_n += 1
        if self.step_n <= self.warmup:
            s = self.step_n / max(1, self.warmup)
        else:
            p = (self.step_n - self.warmup) / max(1, self.total - self.warmup)
            s = max(self.min_lr / self.base[0], 0.5 * (1 + math.cos(math.pi * p)))
        for pg, b in zip(self.opt.param_groups, self.base):
            pg["lr"] = b * s

    def get_lr(self): return self.opt.param_groups[0]["lr"]


class CandleTrainer:
    def __init__(self, model, heads, tokenizer, config=None, device="cuda"):
        self.model = model.to(device)
        self.heads = heads.to(device)
        self.tok = tokenizer
        self.cfg = config or TrainingConfig()
        self.device = torch.device(device)
        self.loss_fn = TradingLoss(self.cfg.ALPHA_CANDLE, self.cfg.BETA_SIGNAL, self.cfg.GAMMA_CONFIDENCE).to(self.device)
        self.use_fp16 = self.cfg.USE_FP16 and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.use_fp16)
        self.global_step = 0
        self.history = {"train": [], "val": []}

    def _setup_opt(self, total_steps):
        decay, no_decay = [], []
        for n, p in list(self.model.named_parameters()) + list(self.heads.named_parameters()):
            (no_decay if "bias" in n or "norm" in n else decay).append(p)
        self.opt = AdamW([{"params": decay, "weight_decay": self.cfg.WEIGHT_DECAY},
                          {"params": no_decay, "weight_decay": 0}], lr=self.cfg.LEARNING_RATE, betas=(self.cfg.BETA1, self.cfg.BETA2))
        self.sched = CosineWarmup(self.opt, self.cfg.WARMUP_STEPS, total_steps, self.cfg.MIN_LR)

    def train_epoch(self, dl):
        self.model.train(); self.heads.train()
        losses = {"total": 0, "candle": 0, "signal": 0, "confidence": 0}; n = 0
        for batch in dl:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            tc = batch["target_candle"].to(self.device)
            ts = batch["target_signal"].to(self.device)
            tconf = batch["target_confidence"].to(self.device)
            self.opt.zero_grad()
            with autocast(self.device.type, enabled=self.use_fp16):
                h = self.model(ids, mask)
                pred = self.heads(h)
                l = self.loss_fn(pred, {"next_candle": tc, "trade_signal": ts, "confidence": tconf})
            self.scaler.scale(l["total_loss"]).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.heads.parameters()), self.cfg.GRAD_CLIP)
            self.scaler.step(self.opt); self.scaler.update(); self.sched.step()
            for k in losses: losses[k] += l[f"{k}_loss" if k != "total" else "total_loss"].item()
            n += 1; self.global_step += 1
        return {k: v/max(1,n) for k,v in losses.items()}

    @torch.no_grad()
    def validate(self, dl):
        self.model.eval(); self.heads.eval()
        losses = {"total": 0, "candle": 0, "signal": 0, "confidence": 0}; n = 0; correct = 0; total = 0
        for batch in dl:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            tc = batch["target_candle"].to(self.device)
            ts = batch["target_signal"].to(self.device)
            tconf = batch["target_confidence"].to(self.device)
            with autocast(self.device.type, enabled=self.use_fp16):
                h = self.model(ids, mask)
                pred = self.heads(h)
                l = self.loss_fn(pred, {"next_candle": tc, "trade_signal": ts, "confidence": tconf})
            for k in losses: losses[k] += l[f"{k}_loss" if k != "total" else "total_loss"].item()
            correct += (pred["trade_signal"].argmax(1) == ts).sum().item()
            total += ts.size(0); n += 1
        d = {k: v/max(1,n) for k,v in losses.items()}
        d["signal_accuracy"] = correct / max(1, total)
        return d

    def save(self, path, epoch, val_loss):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "heads": self.heads.state_dict(),
                     "opt": self.opt.state_dict(), "val_loss": val_loss, "step": self.global_step,
                     "cfg": {"model": self.model.cfg.__dict__}}, path)

    def fit(self, train_candles, val_candles, timeframe="1h"):
        td = CandleDataset(train_candles, self.tok, self.cfg, timeframe, augment=True)
        vd = CandleDataset(val_candles, self.tok, self.cfg, timeframe, augment=False)
        tdl = DataLoader(td, self.cfg.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(self.device.type=="cuda"))
        vdl = DataLoader(vd, self.cfg.BATCH_SIZE, shuffle=False, num_workers=0)
        total = len(tdl) * self.cfg.EPOCHS
        self._setup_opt(total)

        print(f"Training on {self.device} | Params: {self.model.count_parameters():,} | Samples: {len(td)} | Steps: {total:,}")
        best = float("inf")
        ckpt_dir = Path(self.cfg.CHECKPOINT_DIR); ckpt_dir.mkdir(parents=True, exist_ok=True)

        for ep in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            tl = self.train_epoch(tdl)
            vl = self.validate(vdl)
            self.history["train"].append(tl); self.history["val"].append(vl)
            dt = time.time() - t0
            lr = self.sched.get_lr()
            print(f"Ep {ep:3d}/{self.cfg.EPOCHS} | Train: {tl['total']:.4f} | Val: {vl['total']:.4f} | Acc: {vl['signal_accuracy']:.3f} | LR: {lr:.2e} | {dt:.0f}s")

            if vl["total"] < best:
                best = vl["total"]
                self.save(str(ckpt_dir/"best_model.pt"), ep, best)
            if ep % self.cfg.SAVE_EVERY == 0:
                self.save(str(ckpt_dir/f"ckpt_{ep}.pt"), ep, vl["total"])

        self.save(str(ckpt_dir/"final_model.pt"), self.cfg.EPOCHS, vl["total"])
        with open(ckpt_dir/"history.json", "w") as f: json.dump(self.history, f, indent=2)
        print(f"Done. Best val loss: {best:.4f}")
        return self.history


# ============================================================
# CELL 3: Fetch data and train
# ============================================================

def main():
    print("=" * 60)
    print("  CandleStick Transformer — GPU Training on Kaggle")
    print("=" * 60)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU! Go to Settings → Accelerator → GPU T4")

    # Fetch BTC data
    print("\nFetching BTC/USDT data from Binance...")
    fetcher = BinanceFetcher()
    candles = fetcher.fetch_all(symbol="BTCUSDT", interval="1h", days=365)
    print(f"Got {len(candles)} candles")
    print(f"Price: ${candles[:,3].min():.0f} - ${candles[:,3].max():.0f}")

    # Split
    train, val, test = create_split(candles)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Build full model
    tok_cfg = TokenizerConfig()
    model_cfg = ModelConfig(
        VOCAB_SIZE=tok_cfg.vocab_size,
        D_MODEL=512,
        N_HEADS=8,
        N_LAYERS=12,
        D_FF=2048,
        DROPOUT=0.1,
        MAX_SEQ_LEN=512,
        LOCAL_WINDOW=10,
        MAX_RELATIVE_POSITION=128,
    )
    tokenizer = CandleStickTokenizer(tok_cfg)
    model = CandleTransformer(model_cfg)
    heads = TradingHeads(model_cfg)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Train
    cfg = TrainingConfig()
    cfg.EPOCHS = 100
    cfg.BATCH_SIZE = 32
    cfg.LEARNING_RATE = 3e-4
    cfg.WARMUP_STEPS = 500
    cfg.WINDOW_SIZE = 64
    cfg.CHECKPOINT_DIR = "checkpoints/candle_llm_gpu"
    cfg.USE_FP16 = (device == "cuda")
    cfg.SAVE_EVERY = 10

    trainer = CandleTrainer(model, heads, tokenizer, config=cfg, device=device)
    history = trainer.fit(train, val, timeframe="1h")

    # Final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    best_train = min(h["total"] for h in history["train"])
    best_val = min(h["total"] for h in history["val"])
    best_acc = max(h["signal_accuracy"] for h in history["val"])
    print(f"Best train loss: {best_train:.4f}")
    print(f"Best val loss:   {best_val:.4f}")
    print(f"Best val accuracy: {best_acc:.3f}")

    # Save to Kaggle output
    print("\nDownloading checkpoints...")
    import shutil
    shutil.make_archive("candle_llm_checkpoints", "zip", "checkpoints/candle_llm_gpu")
    print("Saved: candle_llm_checkpoints.zip")


if __name__ == "__main__":
    main()
