"""Configuration Manager — YAML-based config with hot-reload.

Implements:
- YAML config file loading
- Environment variable overrides
- Hot-reload on file change
- Config validation
- Default config generation
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "system": {
        "name": "Omega Trading Bot",
        "version": "1.0.0",
        "log_level": "INFO",
        "data_dir": "data",
        "db_path": "data/omega.db",
    },
    "exchange": {
        "name": "binance",
        "api_key": "",
        "api_secret": "",
        "testnet": True,
        "rate_limit": 1200,
    },
    "trading": {
        "mode": "paper",  # paper, live
        "initial_capital": 100000,
        "max_positions": 10,
        "max_position_pct": 0.10,
        "default_stop_loss": 0.02,
        "default_take_profit": 0.04,
    },
    "risk": {
        "max_drawdown": 0.15,
        "max_daily_loss": 0.03,
        "max_leverage": 3.0,
        "max_concentration": 0.40,
        "circuit_breaker_enabled": True,
    },
    "evolution": {
        "enabled": True,
        "population_size": 50,
        "mutation_rate": 0.15,
        "crossover_rate": 0.7,
        "elite_pct": 0.1,
        "max_generations": 1000,
        "eval_interval_hours": 24,
    },
    "strategies": {
        "active": ["trend_following", "mean_reversion", "momentum"],
        "max_active": 5,
        "min_sharpe": 0.5,
        "min_trades": 20,
    },
    "monitoring": {
        "alerts_enabled": True,
        "telegram_enabled": False,
        "telegram_token": "",
        "telegram_chat_id": "",
        "dashboard_port": 8080,
    },
}


class ConfigManager:
    """Configuration management with hot-reload."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: dict = {}
        self._last_modified: float = 0
        self._callbacks: list[callable] = []
        self._load()

    def _load(self) -> None:
        """Load config from file or create default."""
        path = Path(self.config_path)

        if path.exists():
            try:
                import yaml
                with open(path) as f:
                    self._config = yaml.safe_load(f) or {}
                self._last_modified = path.stat().st_mtime
                logger.info(f"Loaded config from {self.config_path}")
            except ImportError:
                # Try JSON fallback
                json_path = path.with_suffix(".json")
                if json_path.exists():
                    with open(json_path) as f:
                        self._config = json.load(f)
                    self._last_modified = json_path.stat().st_mtime
                else:
                    self._config = dict(DEFAULT_CONFIG)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self._config = dict(DEFAULT_CONFIG)
        else:
            self._config = dict(DEFAULT_CONFIG)
            self._save()

        # Apply env overrides
        self._apply_env_overrides()

    def _save(self) -> None:
        """Save current config to file."""
        path = Path(self.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        except ImportError:
            # Fallback to JSON
            json_path = path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(self._config, f, indent=2)
            path = json_path

        self._last_modified = path.stat().st_mtime if path.exists() else 0

    def _apply_env_overrides(self) -> None:
        """Override config with environment variables.

        Format: OMEGA_SECTION_KEY=value
        Example: OMEGA_EXCHANGE_API_KEY=xxx
        """
        prefix = "OMEGA_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) == 2:
                section, param = parts
                if section in self._config:
                    # Try to parse as number/bool
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    else:
                        try:
                            value = float(value)
                            if value == int(value):
                                value = int(value)
                        except ValueError:
                            pass
                    self._config[section][param] = value

    def check_reload(self) -> bool:
        """Check if config file has changed and reload if needed."""
        path = Path(self.config_path)
        if not path.exists():
            return False

        mtime = path.stat().st_mtime
        if mtime > self._last_modified:
            self._load()
            for cb in self._callbacks:
                try:
                    cb(self._config)
                except Exception as e:
                    logger.error(f"Config callback error: {e}")
            return True
        return False

    def get(self, section: str, key: str = None, default: any = None) -> any:
        """Get config value."""
        sect = self._config.get(section, {})
        if key is None:
            return sect
        return sect.get(key, default)

    def set(self, section: str, key: str, value: any) -> None:
        """Set config value."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        self._save()

    def get_all(self) -> dict:
        """Get full config."""
        return dict(self._config)

    def on_change(self, callback: callable) -> None:
        """Register a callback for config changes."""
        self._callbacks.append(callback)

    def generate_default(self) -> None:
        """Generate default config file."""
        self._config = dict(DEFAULT_CONFIG)
        self._save()
