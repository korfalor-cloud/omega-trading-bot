"""Tests for credential storage — config loading, env overrides, persistence."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingbot.config import ExchangeConfig, OmegaConfig, load_config
from tradingbot.infrastructure.config_manager import ConfigManager, DEFAULT_CONFIG


class TestExchangeCredentialStorage:
    """Test credential storage in ExchangeConfig."""

    def test_empty_by_default(self):
        cfg = ExchangeConfig(name="binance")
        assert cfg.api_key == ""
        assert cfg.api_secret == ""

    def test_round_trip_values(self):
        cfg = ExchangeConfig(name="binance", api_key="key_abc", api_secret="secret_xyz", passphrase="phrase_123")
        data = cfg.model_dump()
        assert data["api_key"] == "key_abc"
        assert data["api_secret"] == "secret_xyz"
        assert data["passphrase"] == "phrase_123"


class TestConfigManagerCredentialPersistence:
    """Test that credentials persist through config file save/load cycles."""

    @pytest.fixture
    def config_path(self):
        path = tempfile.mktemp(suffix=".yaml")
        yield path
        for ext in [".yaml", ".json"]:
            p = Path(path).with_suffix(ext)
            if p.exists():
                os.unlink(p)

    def test_save_and_load_api_key(self, config_path):
        cm = ConfigManager(config_path)
        cm.set("exchange", "api_key", "saved_key")
        cm.set("exchange", "api_secret", "saved_secret")

        cm2 = ConfigManager(config_path)
        assert cm2.get("exchange", "api_key") == "saved_key"
        assert cm2.get("exchange", "api_secret") == "saved_secret"

    def test_overwrite_credentials(self, config_path):
        cm = ConfigManager(config_path)
        cm.set("exchange", "api_key", "old_key")
        cm.set("exchange", "api_key", "new_key")
        cm2 = ConfigManager(config_path)
        assert cm2.get("exchange", "api_key") == "new_key"

    def test_clear_credentials(self, config_path):
        cm = ConfigManager(config_path)
        cm.set("exchange", "api_key", "some_key")
        cm.set("exchange", "api_key", "")
        cm2 = ConfigManager(config_path)
        assert cm2.get("exchange", "api_key") == ""


class TestDefaultCredentialConfig:
    """Test that DEFAULT_CONFIG has safe empty credential values."""

    def test_default_config_has_exchange_section(self):
        assert "exchange" in DEFAULT_CONFIG
        assert "api_key" in DEFAULT_CONFIG["exchange"]
        assert "api_secret" in DEFAULT_CONFIG["exchange"]

    def test_default_telegram_has_token_keys(self):
        assert "telegram_token" in DEFAULT_CONFIG["monitoring"]
        assert "telegram_chat_id" in DEFAULT_CONFIG["monitoring"]

    def test_default_testnet_is_true(self):
        assert DEFAULT_CONFIG["exchange"]["testnet"] is True


class TestEnvCredentialLoading:
    """Test loading credentials from environment variables."""

    @pytest.fixture
    def config_path(self):
        path = tempfile.mktemp(suffix=".yaml")
        yield path
        for ext in [".yaml", ".json"]:
            p = Path(path).with_suffix(ext)
            if p.exists():
                os.unlink(p)

    def test_env_overrides_exchange_key(self, config_path):
        cm = ConfigManager(config_path)
        with patch.dict(os.environ, {"OMEGA_EXCHANGE_API_KEY": "env_exchange_key"}):
            cm._apply_env_overrides()
            assert cm.get("exchange", "api_key") == "env_exchange_key"

    def test_env_does_not_affect_unprefixed(self, config_path):
        cm = ConfigManager(config_path)
        original = cm.get("exchange", "api_key")
        with patch.dict(os.environ, {"EXCHANGE_API_KEY": "should_not_apply"}):
            cm._apply_env_overrides()
            assert cm.get("exchange", "api_key") == original

    def test_multiple_env_overrides(self, config_path):
        cm = ConfigManager(config_path)
        env = {
            "OMEGA_EXCHANGE_API_KEY": "multi_key",
            "OMEGA_TRADING_MODE": "live",
        }
        with patch.dict(os.environ, env):
            cm._apply_env_overrides()
            assert cm.get("exchange", "api_key") == "multi_key"
            assert cm.get("trading", "mode") == "live"


class TestMultipleExchangeCredentials:
    """Test managing credentials for multiple exchanges."""

    def test_multiple_exchanges_in_config(self):
        exchanges = {
            "binance": ExchangeConfig(name="binance", api_key="b_key", api_secret="b_secret"),
            "bybit": ExchangeConfig(name="bybit", api_key="y_key", api_secret="y_secret"),
        }
        config = OmegaConfig(exchanges=exchanges)
        assert config.exchanges["binance"].api_key == "b_key"
        assert config.exchanges["bybit"].api_key == "y_key"

    def test_exchanges_independent(self):
        b = ExchangeConfig(name="binance", api_key="b_key")
        y = ExchangeConfig(name="bybit", api_key="y_key")
        assert b.api_key != y.api_key
