"""
Configuration Manager for PolySight
Handles settings via UI (config.json) and .env file with proper priority.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Config file path
CONFIG_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = CONFIG_DIR / "config.json"


class ConfigManager:
    """
    Manages application configuration with priority:
    1. config.json (UI에서 저장한 값)
    2. .env 파일
    3. 기본값
    """

    _instance = None
    _config: Dict[str, Any] = {}

    # Default configuration
    DEFAULTS = {
        "elastic_url": "",
        "elastic_api_key": "",
        "hf_token": "",
        "jina_api_key": "",
        "jina_mode": "local",  # "local" or "api"
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from all sources"""
        # Start with defaults
        self._config = self.DEFAULTS.copy()

        # Load .env file
        load_dotenv()

        # Override with .env values
        env_mapping = {
            "elastic_url": "ELASTIC_CLOUD_SERVERLESS_URL",
            "elastic_api_key": "ELASTIC_API_KEY",
            "hf_token": "HF_TOKEN",
            "jina_api_key": "JINA_API_KEY",
        }

        for config_key, env_key in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value and not env_value.startswith("your-"):
                self._config[config_key] = env_value

        # Override with config.json (highest priority)
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    json_config = json.load(f)
                    for key, value in json_config.items():
                        if value and str(value).strip():
                            self._config[key] = value
                logger.info(f"Loaded config from {CONFIG_FILE}")
            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}")

        # Auto-detect jina_mode based on API key
        if self._config.get("jina_api_key"):
            self._config["jina_mode"] = "api"

        logger.debug(f"Config loaded: {self._masked_config()}")

    def _masked_config(self) -> Dict[str, Any]:
        """Return config with sensitive values masked"""
        masked = {}
        sensitive_keys = ["elastic_api_key", "hf_token", "jina_api_key"]
        for key, value in self._config.items():
            if key in sensitive_keys and value:
                masked[key] = value[:8] + "..." if len(value) > 8 else "***"
            else:
                masked[key] = value
        return masked

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value (in memory)"""
        self._config[key] = value

        # Auto-update jina_mode
        if key == "jina_api_key":
            self._config["jina_mode"] = "api" if value else "local"

    def save(self):
        """Save current configuration to config.json"""
        try:
            # Only save non-empty values
            to_save = {k: v for k, v in self._config.items() if v}
            with open(CONFIG_FILE, "w") as f:
                json.dump(to_save, f, indent=2)
            logger.info(f"Config saved to {CONFIG_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def reload(self):
        """Reload configuration from all sources"""
        self._load_config()

    @property
    def elastic_url(self) -> str:
        return self._config.get("elastic_url", "")

    @property
    def elastic_api_key(self) -> str:
        return self._config.get("elastic_api_key", "")

    @property
    def hf_token(self) -> str:
        return self._config.get("hf_token", "")

    @property
    def jina_api_key(self) -> str:
        return self._config.get("jina_api_key", "")

    @property
    def jina_mode(self) -> str:
        """Returns 'local' or 'api'"""
        return self._config.get("jina_mode", "local")

    @property
    def is_elastic_configured(self) -> bool:
        """Check if Elasticsearch is properly configured"""
        return bool(self.elastic_url and self.elastic_api_key)

    @property
    def is_jina_api_configured(self) -> bool:
        """Check if Jina API is configured"""
        return bool(self.jina_api_key)

    def get_status(self) -> Dict[str, Any]:
        """Get configuration status for UI display"""
        return {
            "elastic": {
                "configured": self.is_elastic_configured,
                "url": self.elastic_url[:30] + "..." if len(self.elastic_url) > 30 else self.elastic_url,
                "has_api_key": bool(self.elastic_api_key),
            },
            "jina": {
                "mode": self.jina_mode,
                "api_configured": self.is_jina_api_configured,
            },
            "hf": {
                "configured": bool(self.hf_token),
            }
        }


# Singleton instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the singleton ConfigManager instance"""
    return config
