"""
CandleStick Transformer — The LLM Architecture
A custom transformer designed for candlestick sequence modeling.

Key features:
  - Multi-scale attention (local window + global)
  - Relative position encoding with temporal decay
  - Candle-aware layer normalization
  - Causal masking for autoregressive generation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, get_model_config


class RelativePositionBias(nn.Module):
    """
    Learnable relative position biases with temporal decay.
    Recent candles get stronger attention signals.
    """

    def __init__(self, max_relative_position: int, n_heads: int):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.n_heads = n_heads

        # Bias table: (2 * max_rel_pos + 1) x n_heads
        self.bias_table = nn.Parameter(
            torch.zeros(2 * max_relative_position + 1, n_heads)
        )
        nn.init.xavier_uniform_(self.bias_table)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns bias of shape (n_heads, seq_len, seq_len)"""
        positions = torch.arange(seq_len, device=self.bias_table.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)

        # Clamp to max relative position
        relative_positions = relative_positions.clamp(
            -self.max_relative_position, self.max_relative_position
        )
        # Shift to non-negative index
        indices = relative_positions + self.max_relative_position

        biases = self.bias_table[indices]  # (L, L, n_heads)
        return biases.permute(2, 0, 1)  # (n_heads, L, L)


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention combining:
      1. Global causal attention (full sequence)
      2. Local windowed attention (recent candles)
    Combined via learned gate.
    """

    def __init__(self, d_model: int, n_heads: int, local_window: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_window = local_window

        # Global attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Gate to blend local and global attention
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask. True = masked (blocked)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # (L, L)

    def _local_mask(self, seq_len: int, window: int, device: torch.device) -> torch.Tensor:
        """Create local window attention mask. True = masked."""
        mask = torch.ones(seq_len, seq_len, device=device).bool()
        for i in range(seq_len):
            start = max(0, i - window + 1)
            mask[i, start:i + 1] = False
        return mask

    def forward(
        self, x: torch.Tensor, rel_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            rel_bias: (n_heads, seq_len, seq_len) relative position biases

        Returns:
            (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, H, L, head_dim)

        # ── Global Attention ──
        attn_global = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        if rel_bias is not None:
            attn_global = attn_global + rel_bias.unsqueeze(0)

        causal_mask = self._causal_mask(L, x.device)
        attn_global = attn_global.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_global = F.softmax(attn_global, dim=-1)
        attn_global = self.dropout(attn_global)
        out_global = torch.matmul(attn_global, v)  # (B, H, L, head_dim)

        # ── Local Attention ──
        attn_local = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if rel_bias is not None:
            attn_local = attn_local + rel_bias.unsqueeze(0)

        local_mask = self._local_mask(L, self.local_window, x.device)
        combined_mask = causal_mask | local_mask
        attn_local = attn_local.masked_fill(combined_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_local = F.softmax(attn_local, dim=-1)
        attn_local = self.dropout(attn_local)
        out_local = torch.matmul(attn_local, v)

        # ── Gated Fusion ──
        # Reshape for gate input
        out_global_flat = out_global.transpose(1, 2).contiguous().view(B, L, D)
        out_local_flat = out_local.transpose(1, 2).contiguous().view(B, L, D)

        gate_input = torch.cat([out_global_flat, out_local_flat], dim=-1)  # (B, L, 2D)
        g = self.gate(gate_input)  # (B, L, D)

        out = g * out_global_flat + (1.0 - g) * out_local_flat
        return self.out_proj(out)


class CandleAwareLayerNorm(nn.Module):
    """
    Layer normalization that also considers candle volatility
    (high-low range) as a scaling factor.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

        # Learnable volatility scaling
        self.vol_scale = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
        )
        self.vol_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, volatility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            volatility: (batch, seq_len, 1) — normalized high-low range per candle
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta

        if volatility is not None:
            vol_emb = self.vol_scale(volatility)  # (B, L, D)
            out = out + self.vol_weight * vol_emb

        return out


class CandleTransformerBlock(nn.Module):
    """
    Single transformer block with:
      - Multi-scale attention
      - Gated feed-forward network
      - Candle-aware layer norm
      - Residual connections
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.attn = MultiScaleAttention(
            config.D_MODEL, config.N_HEADS, config.LOCAL_WINDOW, config.DROPOUT
        )
        self.norm1 = CandleAwareLayerNorm(config.D_MODEL)
        self.norm2 = CandleAwareLayerNorm(config.D_MODEL)

        self.ffn = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_FF),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_FF, config.D_MODEL),
            nn.Dropout(config.DROPOUT),
        )

        # Relative position bias for this layer
        self.rel_pos = RelativePositionBias(
            config.MAX_RELATIVE_POSITION, config.N_HEADS
        )

    def forward(
        self, x: torch.Tensor, volatility: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            volatility: (batch, seq_len, 1)
        """
        # Pre-norm attention with residual
        L = x.size(1)
        rel_bias = self.rel_pos(L)

        normed = self.norm1(x, volatility)
        attn_out = self.attn(normed, rel_bias)
        x = x + attn_out

        # Pre-norm FFN with residual
        normed = self.norm2(x, volatility)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class CandleTransformer(nn.Module):
    """
    The full CandleStick Transformer LLM.

    Takes tokenized candlestick sequences and produces contextualized
    hidden representations for downstream tasks (next candle prediction,
    trade signals, confidence scoring).
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = get_model_config()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.embedding_dropout = nn.Dropout(config.DROPOUT)

        # Learnable temporal decay for position encoding
        self.position_encoding = nn.Parameter(
            torch.zeros(1, config.MAX_SEQ_LEN, config.D_MODEL)
        )
        nn.init.trunc_normal_(self.position_encoding, std=0.02)

        # Transformer blocks
        self.layers = nn.ModuleList([
            CandleTransformerBlock(config) for _ in range(config.N_LAYERS)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.D_MODEL)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_volatility(self, candle_tokens: torch.Tensor) -> torch.Tensor:
        """
        Estimate volatility from token embeddings.
        Uses the high-low spread bins (tokens at positions 2 and 3 in each candle group).

        Returns: (batch, seq_len, 1)
        """
        B, L = candle_tokens.shape
        tokens_per_candle = 8  # from tokenizer config

        # Get high and low bin token IDs
        # In each group of 8 tokens: [tf, close, high, low, vol, pattern, ind1, ind2]
        high_positions = torch.arange(2, L, tokens_per_candle, device=candle_tokens.device)
        low_positions = torch.arange(3, L, tokens_per_candle, device=candle_tokens.device)

        # Clamp to valid range
        high_positions = high_positions[high_positions < L]
        low_positions = low_positions[low_positions < L]

        min_len = min(len(high_positions), len(low_positions))
        high_positions = high_positions[:min_len]
        low_positions = low_positions[:min_len]

        high_tok = candle_tokens[:, high_positions].float()
        low_tok = candle_tokens[:, low_positions].float()

        volatility = (high_tok - low_tok).abs().unsqueeze(-1)  # (B, N_candles, 1)

        # Expand back to full sequence length
        vol_full = torch.zeros(B, L, 1, device=candle_tokens.device)
        candle_indices = torch.arange(min_len, device=candle_tokens.device) * tokens_per_candle
        for ci, vi in zip(candle_indices, range(min_len)):
            end = min(ci + tokens_per_candle, L)
            if ci < L:
                vol_val = volatility[:, vi, :].unsqueeze(1)  # (B, 1, 1)
                vol_full[:, ci:end, :] = vol_val.expand(B, end - ci, 1)

        return vol_full

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) — token IDs from CandleStickTokenizer
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, d_model) — contextualized hidden states
        """
        B, L = input_ids.shape

        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids)  # (B, L, D)
        x = x + self.position_encoding[:, :L, :]
        x = self.embedding_dropout(x)

        # Compute volatility from raw tokens
        volatility = self._compute_volatility(input_ids)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Zero out padding positions
            x = x * attention_mask.unsqueeze(-1)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, volatility)

        x = self.final_norm(x)

        # Re-apply mask to ensure padding positions stay zero
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        return x

    def get_last_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get hidden state of the last non-padding token for each sequence."""
        hidden = self.forward(input_ids, attention_mask)

        if attention_mask is not None:
            # Find last non-padding position for each sequence
            lengths = attention_mask.sum(dim=1).long() - 1  # (B,)
            lengths = lengths.clamp(min=0)
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_indices, lengths]  # (B, D)
        else:
            return hidden[:, -1, :]  # (B, D)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
