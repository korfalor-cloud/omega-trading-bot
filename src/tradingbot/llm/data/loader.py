"""
CandleStick Transformer — CSV/Data File Loader
Loads OHLCV data from CSV files and other local formats.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


class CSVLoader:
    """Load OHLCV candlestick data from CSV files."""

    # Common column name mappings
    COLUMN_MAPPINGS = {
        # timestamp
        "timestamp": "timestamp", "time": "timestamp", "date": "timestamp",
        "datetime": "timestamp", "Date": "timestamp", "Datetime": "timestamp",
        "Timestamp": "timestamp",
        # open
        "open": "open", "Open": "open", "OPEN": "open",
        # high
        "high": "high", "High": "high", "HIGH": "high",
        # low
        "low": "low", "Low": "low", "LOW": "low",
        # close
        "close": "close", "Close": "close", "CLOSE": "close",
        # volume
        "volume": "volume", "Volume": "volume", "vol": "volume",
        "Vol": "volume", "VOLUME": "volume",
    }

    def load(
        self,
        file_path: Union[str, Path],
        timestamp_col: Optional[str] = None,
        open_col: Optional[str] = None,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        close_col: Optional[str] = None,
        volume_col: Optional[str] = None,
    ) -> np.ndarray:
        """
        Load OHLCV data from a CSV file.

        Args:
            file_path: Path to CSV file
            timestamp_col: Name of timestamp column (auto-detected if None)
            open_col: Name of open column (auto-detected if None)
            high_col: Name of high column
            low_col: Name of low column
            close_col: Name of close column
            volume_col: Name of volume column

        Returns:
            Array of shape (N, 6) — [open, high, low, close, volume, timestamp_ms]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(path)

        if df.empty:
            return np.zeros((0, 6), dtype=np.float64)

        # Auto-detect column names
        col_map = self._detect_columns(df, timestamp_col, open_col, high_col,
                                        low_col, close_col, volume_col)

        # Extract OHLCV
        candles = np.zeros((len(df), 6), dtype=np.float64)

        candles[:, 0] = df[col_map["open"]].values.astype(np.float64)
        candles[:, 1] = df[col_map["high"]].values.astype(np.float64)
        candles[:, 2] = df[col_map["low"]].values.astype(np.float64)
        candles[:, 3] = df[col_map["close"]].values.astype(np.float64)
        candles[:, 4] = df[col_map["volume"]].values.astype(np.float64)

        # Parse timestamps
        if "timestamp" in col_map:
            ts_col = col_map["timestamp"]
            candles[:, 5] = self._parse_timestamps(df[ts_col])
        else:
            # Use row index as timestamp
            candles[:, 5] = np.arange(len(df), dtype=np.float64)

        # Drop rows with NaN in OHLCV
        valid = ~np.any(np.isnan(candles[:, :5]), axis=1)
        candles = candles[valid]

        return candles

    def _detect_columns(
        self, df: pd.DataFrame,
        timestamp_col, open_col, high_col, low_col, close_col, volume_col,
    ) -> dict:
        """Auto-detect column names from DataFrame."""
        result = {}

        col_names = list(df.columns)

        # Map provided or auto-detected columns
        for target, provided in [
            ("timestamp", timestamp_col),
            ("open", open_col),
            ("high", high_col),
            ("low", low_col),
            ("close", close_col),
            ("volume", volume_col),
        ]:
            if provided and provided in col_names:
                result[target] = provided
            else:
                # Try common names
                for candidate in col_names:
                    mapped = self.COLUMN_MAPPINGS.get(candidate)
                    if mapped == target and target not in result:
                        result[target] = candidate
                        break

        # Validate required columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(result.keys())
        if missing:
            available = list(df.columns)
            raise ValueError(
                f"Could not find columns for: {missing}. "
                f"Available columns: {available}. "
                f"Provide explicit column names."
            )

        return result

    def _parse_timestamps(self, series: pd.Series) -> np.ndarray:
        """Parse various timestamp formats to milliseconds."""
        # Try numeric first (unix timestamps)
        try:
            values = series.values.astype(np.float64)
            # If values are in seconds (not ms), multiply by 1000
            if values.max() < 1e12:
                values = values * 1000.0
            return values
        except (ValueError, TypeError):
            pass

        # Try datetime parsing
        try:
            dt = pd.to_datetime(series)
            return (dt.astype(np.int64) // 10**6).values.astype(np.float64)
        except Exception:
            pass

        # Fallback: use index
        return np.arange(len(series), dtype=np.float64)
