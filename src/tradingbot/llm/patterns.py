"""
CandleStick Transformer — Candlestick Pattern Recognition
Detects 25+ candlestick patterns from OHLCV data.
Each pattern is assigned a unique token ID.
"""

from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


class PatternID(IntEnum):
    """Pattern token IDs. 0-15 reserved, patterns start at 16."""
    NONE = 0

    # Single candle patterns (16-31)
    DOJI = 16
    LONG_LEGGED_DOJI = 17
    DRAGONFLY_DOJI = 18
    GRAVESTONE_DOJI = 19
    HAMMER = 20
    INVERTED_HAMMER = 21
    SHOOTING_STAR = 22
    HANGING_MAN = 23
    MARUBOZU_BULL = 24
    MARUBOZU_BEAR = 25
    SPINNING_TOP = 26
    BELT_HOLD_BULL = 27
    BELT_HOLD_BEAR = 28

    # Double candle patterns (32-47)
    ENGULFING_BULL = 32
    ENGULFING_BEAR = 33
    HARAMI_BULL = 34
    HARAMI_BEAR = 35
    PIERCING_LINE = 36
    DARK_CLOUD = 37
    TWEEZER_TOP = 38
    TWEEZER_BOTTOM = 39
    KICKING_BULL = 40
    KICKING_BEAR = 41
    MEETING_LINES_BULL = 42
    MEETING_LINES_BEAR = 43

    # Triple candle patterns (48-63)
    MORNING_STAR = 48
    EVENING_STAR = 49
    THREE_WHITE_SOLDIERS = 50
    THREE_BLACK_CROWS = 51
    THREE_INSIDE_UP = 52
    THREE_INSIDE_DOWN = 53
    THREE_OUTSIDE_UP = 54
    THREE_OUTSIDE_DOWN = 55
    ABANDONED_BABY_BULL = 56
    ABANDONED_BABY_BEAR = 57

    COUNT = 64  # total pattern slots


def _body(candle: np.ndarray) -> float:
    """Body size: |close - open|"""
    o, h, l, c = candle[0], candle[1], candle[2], candle[3]
    return abs(c - o)


def _range(candle: np.ndarray) -> float:
    """Full range: high - low"""
    return candle[1] - candle[2]


def _upper_shadow(candle: np.ndarray) -> float:
    """Upper shadow: high - max(open, close)"""
    o, h, c = candle[0], candle[1], candle[3]
    return h - max(o, c)


def _lower_shadow(candle: np.ndarray) -> float:
    """Lower shadow: min(open, close) - low"""
    o, l, c = candle[0], candle[2], candle[3]
    return min(o, c) - l


def _is_bullish(candle: np.ndarray) -> bool:
    return candle[3] > candle[0]  # close > open


def _is_bearish(candle: np.ndarray) -> bool:
    return candle[3] < candle[0]  # close < open


def _body_midpoint(candle: np.ndarray) -> float:
    return (candle[0] + candle[3]) / 2.0


def _avg_body(candles: np.ndarray) -> float:
    """Average body size over a window of candles. Shape: (N, 6) -> (OHLCV + timestamp)"""
    bodies = np.abs(candles[:, 3] - candles[:, 0])
    return np.mean(bodies) if len(bodies) > 0 else 0.0


# ─── Single Candle Patterns ───


def detect_doji(c: np.ndarray, threshold: float = 0.05) -> bool:
    """Doji: body is tiny relative to range."""
    r = _range(c)
    if r == 0:
        return True
    return _body(c) / r < threshold


def detect_long_legged_doji(c: np.ndarray) -> bool:
    """Long-legged doji: doji with long shadows on both sides."""
    if not detect_doji(c, threshold=0.05):
        return False
    r = _range(c)
    if r == 0:
        return False
    return _upper_shadow(c) / r > 0.3 and _lower_shadow(c) / r > 0.3


def detect_dragonfly_doji(c: np.ndarray) -> bool:
    """Dragonfly doji: doji with long lower shadow, no upper shadow."""
    if not detect_doji(c, threshold=0.05):
        return False
    r = _range(c)
    if r == 0:
        return False
    return _lower_shadow(c) / r > 0.6 and _upper_shadow(c) / r < 0.1


def detect_gravestone_doji(c: np.ndarray) -> bool:
    """Gravestone doji: doji with long upper shadow, no lower shadow."""
    if not detect_doji(c, threshold=0.05):
        return False
    r = _range(c)
    if r == 0:
        return False
    return _upper_shadow(c) / r > 0.6 and _lower_shadow(c) / r < 0.1


def detect_hammer(c: np.ndarray) -> bool:
    """Hammer: small body at top, long lower shadow (2x+ body), little upper shadow."""
    b = _body(c)
    if b == 0:
        return False
    ls = _lower_shadow(c)
    us = _upper_shadow(c)
    return ls >= 2.0 * b and us < b * 0.5


def detect_inverted_hammer(c: np.ndarray) -> bool:
    """Inverted hammer: small body at bottom, long upper shadow."""
    b = _body(c)
    if b == 0:
        return False
    us = _upper_shadow(c)
    ls = _lower_shadow(c)
    return us >= 2.0 * b and ls < b * 0.5


def detect_shooting_star(c: np.ndarray) -> bool:
    """Shooting star: same shape as inverted hammer but in uptrend."""
    return detect_inverted_hammer(c)


def detect_hanging_man(c: np.ndarray) -> bool:
    """Hanging man: same shape as hammer but in uptrend."""
    return detect_hammer(c)


def detect_marubozu_bull(c: np.ndarray) -> bool:
    """Bullish marubozu: long green body, no shadows."""
    if not _is_bullish(c):
        return False
    r = _range(c)
    if r == 0:
        return False
    return _body(c) / r > 0.9


def detect_marubozu_bear(c: np.ndarray) -> bool:
    """Bearish marubozu: long red body, no shadows."""
    if not _is_bearish(c):
        return False
    r = _range(c)
    if r == 0:
        return False
    return _body(c) / r > 0.9


def detect_spinning_top(c: np.ndarray) -> bool:
    """Spinning top: small body with shadows on both sides."""
    b = _body(c)
    r = _range(c)
    if r == 0:
        return False
    body_ratio = b / r
    us = _upper_shadow(c) / r
    ls = _lower_shadow(c) / r
    return 0.1 < body_ratio < 0.35 and us > 0.15 and ls > 0.15


def detect_belt_hold_bull(c: np.ndarray) -> bool:
    """Bullish belt hold: opens at low, closes near high, bullish."""
    if not _is_bullish(c):
        return False
    return _lower_shadow(c) < _body(c) * 0.1


def detect_belt_hold_bear(c: np.ndarray) -> bool:
    """Bearish belt hold: opens at high, closes near low, bearish."""
    if not _is_bearish(c):
        return False
    return _upper_shadow(c) < _body(c) * 0.1


# ─── Double Candle Patterns ───


def detect_engulfing_bull(candles: np.ndarray) -> bool:
    """Bullish engulfing: bearish candle followed by larger bullish candle that engulfs it."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not _is_bearish(prev) or not _is_bullish(curr):
        return False
    return curr[0] <= prev[3] and curr[3] >= prev[0]  # curr open <= prev close AND curr close >= prev open


def detect_engulfing_bear(candles: np.ndarray) -> bool:
    """Bearish engulfing: bullish candle followed by larger bearish candle that engulfs it."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not _is_bullish(prev) or not _is_bearish(curr):
        return False
    return curr[0] >= prev[3] and curr[3] <= prev[0]


def detect_harami_bull(candles: np.ndarray) -> bool:
    """Bullish harami: large bearish candle followed by small bullish candle inside it."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not _is_bearish(prev) or not _is_bullish(curr):
        return False
    return curr[0] > prev[3] and curr[3] < prev[0] and _body(curr) < _body(prev) * 0.5


def detect_harami_bear(candles: np.ndarray) -> bool:
    """Bearish harami: large bullish candle followed by small bearish candle inside it."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not _is_bullish(prev) or not _is_bearish(curr):
        return False
    return curr[0] < prev[3] and curr[3] > prev[0] and _body(curr) < _body(prev) * 0.5


def detect_piercing_line(candles: np.ndarray) -> bool:
    """Piercing line: bearish candle, then bullish candle that opens below low and closes above midpoint."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not _is_bearish(prev) or not _is_bullish(curr):
        return False
    return curr[0] < prev[2] and curr[3] > _body_midpoint(prev) and curr[3] < prev[0]


def detect_dark_cloud(candles: np.ndarray) -> bool:
    """Dark cloud cover: bullish candle, then bearish candle that opens above high and closes below midpoint."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not _is_bullish(prev) or not _is_bearish(curr):
        return False
    return curr[0] > prev[1] and curr[3] < _body_midpoint(prev) and curr[3] > prev[0]


def detect_tweezer_top(candles: np.ndarray) -> bool:
    """Tweezer top: two candles with matching highs at resistance."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    r = _range(prev)
    if r == 0:
        return False
    return abs(prev[1] - curr[1]) / r < 0.02 and _is_bullish(prev) and _is_bearish(curr)


def detect_tweezer_bottom(candles: np.ndarray) -> bool:
    """Tweezer bottom: two candles with matching lows at support."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    r = _range(prev)
    if r == 0:
        return False
    return abs(prev[2] - curr[2]) / r < 0.02 and _is_bearish(prev) and _is_bullish(curr)


def detect_kicking_bull(candles: np.ndarray) -> bool:
    """Bullish kicking: bearish marubozu followed by bullish marubozu with gap up."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not detect_marubozu_bear(prev) or not detect_marubozu_bull(curr):
        return False
    return curr[0] > prev[0]  # gap up


def detect_kicking_bear(candles: np.ndarray) -> bool:
    """Bearish kicking: bullish marubozu followed by bearish marubozu with gap down."""
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    if not detect_marubozu_bull(prev) or not detect_marubozu_bear(curr):
        return False
    return curr[0] < prev[0]  # gap down


# ─── Triple Candle Patterns ───


def detect_morning_star(candles: np.ndarray) -> bool:
    """Morning star: bearish, small body (star), bullish recovery."""
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not _is_bearish(c1) or not _is_bullish(c3):
        return False
    if _body(c2) > _body(c1) * 0.3:
        return False
    return c3[3] > _body_midpoint(c1)


def detect_evening_star(candles: np.ndarray) -> bool:
    """Evening star: bullish, small body (star), bearish drop."""
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not _is_bullish(c1) or not _is_bearish(c3):
        return False
    if _body(c2) > _body(c1) * 0.3:
        return False
    return c3[3] < _body_midpoint(c1)


def detect_three_white_soldiers(candles: np.ndarray) -> bool:
    """Three white soldiers: three consecutive bullish candles, each closing higher."""
    if len(candles) < 3:
        return False
    for i in range(-3, 0):
        if not _is_bullish(candles[i]):
            return False
    return (candles[-3][3] < candles[-2][3] < candles[-1][3] and
            candles[-3][0] < candles[-2][0] < candles[-1][0])


def detect_three_black_crows(candles: np.ndarray) -> bool:
    """Three black crows: three consecutive bearish candles, each closing lower."""
    if len(candles) < 3:
        return False
    for i in range(-3, 0):
        if not _is_bearish(candles[i]):
            return False
    return (candles[-3][3] > candles[-2][3] > candles[-1][3] and
            candles[-3][0] > candles[-2][0] > candles[-1][0])


def detect_three_inside_up(candles: np.ndarray) -> bool:
    """Three inside up: bearish, harami bullish, confirmation bullish."""
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not _is_bearish(c1) or not _is_bullish(c2) or not _is_bullish(c3):
        return False
    return c2[0] > c1[3] and c2[3] < c1[0] and c3[3] > c1[0]


def detect_three_inside_down(candles: np.ndarray) -> bool:
    """Three inside down: bullish, harami bearish, confirmation bearish."""
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not _is_bullish(c1) or not _is_bearish(c2) or not _is_bearish(c3):
        return False
    return c2[0] < c1[3] and c2[3] > c1[0] and c3[3] < c1[0]


def detect_abandoned_baby_bull(candles: np.ndarray) -> bool:
    """Bullish abandoned baby: bearish, doji gap down, bullish gap up."""
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not _is_bearish(c1) or not detect_doji(c2) or not _is_bullish(c3):
        return False
    return c2[1] < c1[2] and c3[0] > c2[1]  # gap down then gap up


def detect_abandoned_baby_bear(candles: np.ndarray) -> bool:
    """Bearish abandoned baby: bullish, doji gap up, bearish gap down."""
    if len(candles) < 3:
        return False
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    if not _is_bullish(c1) or not detect_doji(c2) or not _is_bearish(c3):
        return False
    return c2[2] > c1[1] and c3[0] < c2[2]  # gap up then gap down


# ─── Main Detection API ───

# Registry: (detector_function, needs_n_candles)
PATTERN_REGISTRY = [
    # Single candle (needs 1)
    (PatternID.DOJI, detect_doji, 1),
    (PatternID.LONG_LEGGED_DOJI, detect_long_legged_doji, 1),
    (PatternID.DRAGONFLY_DOJI, detect_dragonfly_doji, 1),
    (PatternID.GRAVESTONE_DOJI, detect_gravestone_doji, 1),
    (PatternID.HAMMER, detect_hammer, 1),
    (PatternID.INVERTED_HAMMER, detect_inverted_hammer, 1),
    (PatternID.SHOOTING_STAR, detect_shooting_star, 1),
    (PatternID.HANGING_MAN, detect_hanging_man, 1),
    (PatternID.MARUBOZU_BULL, detect_marubozu_bull, 1),
    (PatternID.MARUBOZU_BEAR, detect_marubozu_bear, 1),
    (PatternID.SPINNING_TOP, detect_spinning_top, 1),
    (PatternID.BELT_HOLD_BULL, detect_belt_hold_bull, 1),
    (PatternID.BELT_HOLD_BEAR, detect_belt_hold_bear, 1),

    # Double candle (needs 2)
    (PatternID.ENGULFING_BULL, detect_engulfing_bull, 2),
    (PatternID.ENGULFING_BEAR, detect_engulfing_bear, 2),
    (PatternID.HARAMI_BULL, detect_harami_bull, 2),
    (PatternID.HARAMI_BEAR, detect_harami_bear, 2),
    (PatternID.PIERCING_LINE, detect_piercing_line, 2),
    (PatternID.DARK_CLOUD, detect_dark_cloud, 2),
    (PatternID.TWEEZER_TOP, detect_tweezer_top, 2),
    (PatternID.TWEEZER_BOTTOM, detect_tweezer_bottom, 2),
    (PatternID.KICKING_BULL, detect_kicking_bull, 2),
    (PatternID.KICKING_BEAR, detect_kicking_bear, 2),

    # Triple candle (needs 3)
    (PatternID.MORNING_STAR, detect_morning_star, 3),
    (PatternID.EVENING_STAR, detect_evening_star, 3),
    (PatternID.THREE_WHITE_SOLDIERS, detect_three_white_soldiers, 3),
    (PatternID.THREE_BLACK_CROWS, detect_three_black_crows, 3),
    (PatternID.THREE_INSIDE_UP, detect_three_inside_up, 3),
    (PatternID.THREE_INSIDE_DOWN, detect_three_inside_down, 3),
    (PatternID.ABANDONED_BABY_BULL, detect_abandoned_baby_bull, 3),
    (PatternID.ABANDONED_BABY_BEAR, detect_abandoned_baby_bear, 3),
]


def detect_patterns(candles: np.ndarray) -> List[int]:
    """
    Detect all candlestick patterns in the given candle sequence.

    Args:
        candles: OHLCV array of shape (N, 6) where columns are
                 [open, high, low, close, volume, timestamp]

    Returns:
        List of detected pattern IDs (as ints)
    """
    if len(candles) == 0:
        return []

    detected = []
    for pattern_id, detector, n_needed in PATTERN_REGISTRY:
        if len(candles) < n_needed:
            continue
        window = candles[-n_needed:]
        try:
            if detector(window) if n_needed > 1 else detector(window[0]):
                detected.append(int(pattern_id))
        except (IndexError, ZeroDivisionError):
            continue

    return detected


def detect_pattern_for_candle(candles: np.ndarray) -> int:
    """
    Get the single most significant pattern for the latest candle.
    Returns the pattern ID, or PatternID.NONE if no pattern found.
    Prefers triple > double > single patterns (more significant).
    """
    patterns = detect_patterns(candles)
    if not patterns:
        return int(PatternID.NONE)

    # Prefer higher-numbered patterns (triple > double > single)
    return max(patterns)
