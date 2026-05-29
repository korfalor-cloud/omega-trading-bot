"""Tests for authentication and API key management via configuration."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingbot.config import ExchangeConfig, MonitoringConfig, OmegaConfig, load_config
from tradingbot.infrastructure.config_manager import ConfigManager


class TestExchangeCredentials:
    """Test API key / secret configuration in ExchangeConfig."""

    def test_default_credentials_empty(self):
        cfg = ExchangeConfig(name="binance")
        assert cfg.api_key == ""
        assert cfg.api_secret == ""
        assert cfg.passphrase == ""

    def test_custom_credentials(self):
        cfg = ExchangeConfig(name="binance", api_key="my_key", api_secret="my_secret", passphrase="my_phrase")
        assert cfg.api_key == "my_key"
        assert cfg.api_secret == "my_secret"
        assert cfg.passphrase == "my_phrase"

    def test_testnet_default_true(self):
        cfg = ExchangeConfig(name="binance")
        assert cfg.testnet is True

    def test_testnet_live(self):
        cfg = ExchangeConfig(name="binance", testnet=False)
        assert cfg.testnet is False

    def test_rate_limit_default(self):
        cfg = ExchangeConfig(name="binance")
        assert cfg.rate_limit == 6000


class TestMonitoringTokens:
    """Test monitoring token configuration."""

    def test_default_telegram_empty(self):
        cfg = MonitoringConfig()
        assert cfg.telegram_bot_token == ""
        assert cfg.telegram_chat_id == ""

    def test_custom_telegram(self):
        cfg = MonitoringConfig(telegram_bot_token="bot123", telegram_chat_id="chat456")
        assert cfg.telegram_bot_token == "bot123"
        assert cfg.telegram_chat_id == "chat456"


class TestEnvOverrides:
    """Test that environment variables override configuration values."""

    def test_env_prefix(self):
        with patch.dict(os.environ, {"OMEGA_MODE": "live"}):
            config = OmegaConfig()
            assert config.mode == "live"

    def test_env_log_level(self):
        with patch.dict(os.environ, {"OMEGA_LOG_LEVEL": "DEBUG"}):
            config = OmegaConfig()
            assert config.log_level == "DEBUG"


class TestConfigManagerEnvOverrides:
    """Test ConfigManager environment variable overrides."""

    @pytest.fixture
    def config_path(self):
        path = tempfile.mktemp(suffix=".yaml")
        yield path
        for ext in [".yaml", ".json"]:
            p = Path(path).with_suffix(ext)
            if p.exists():
                os.unlink(p)

    def test_env_override_api_key(self, config_path):
        cm = ConfigManager(config_path)
        with patch.dict(os.environ, {"OMEGA_EXCHANGE_API_KEY": "env_key_123"}):
            cm._apply_env_overrides()
            assert cm.get("exchange", "api_key") == "env_key_123"

    def test_env_override_boolean(self, config_path):
        cm = ConfigManager(config_path)
        with patch.dict(os.environ, {"OMEGA_RISK_CIRCUIT_BREAKER_ENABLED": "false"}):
            cm._apply_env_overrides()
            assert cm.get("risk", "circuit_breaker_enabled") is False

    def test_env_override_numeric(self, config_path):
        cm = ConfigManager(config_path)
        with patch.dict(os.environ, {"OMEGA_TRADING_INITIAL_CAPITAL": "200000"}):
            cm._apply_env_overrides()
            assert cm.get("trading", "initial_capital") == 200000


class TestApiKeyRotation:
    """Test that API keys can be updated at runtime."""

    @pytest.fixture
    def config_path(self):
        path = tempfile.mktemp(suffix=".yaml")
        yield path
        for ext in [".yaml", ".json"]:
            p = Path(path).with_suffix(ext)
            if p.exists():
                os.unlink(p)

    def test_update_api_key(self, config_path):
        cm = ConfigManager(config_path)
        cm.set("exchange", "api_key", "new_key")
        assert cm.get("exchange", "api_key") == "new_key"

    def test_update_api_secret(self, config_path):
        cm = ConfigManager(config_path)
        cm.set("exchange", "api_secret", "new_secret")
        assert cm.get("exchange", "api_secret") == "new_secret"

    def test_credentials_persist_across_reload(self, config_path):
        cm = ConfigManager(config_path)
        cm.set("exchange", "api_key", "persisted_key")
        # Create new manager from same path
        cm2 = ConfigManager(config_path)
        assert cm2.get("exchange", "api_key") == "persisted_key"
