"""
Configuration management for ambient-subconscious.

This module handles loading and validating configuration from YAML files.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager.

    Loads configuration from YAML file and provides typed access to settings.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested keys with dot notation, e.g., "agents.audio.enabled"

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Supports nested keys with dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save to (if None, uses original path)
        """
        save_path = Path(path) if path else self.config_path

        try:
            with open(save_path, 'w') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

    # Convenience properties for common settings

    @property
    def spacetimedb_host(self) -> str:
        return self.get('spacetimedb.host', 'http://127.0.0.1:3000')

    @property
    def spacetimedb_module(self) -> str:
        return self.get('spacetimedb.module', 'ambient-listener')

    @property
    def spacetimedb_auth_token(self) -> Optional[str]:
        return self.get('spacetimedb.auth_token')

    @property
    def spacetimedb_svelte_api_url(self) -> str:
        return self.get('spacetimedb.svelte_api_url', 'http://localhost:5174')

    @property
    def enabled_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all enabled agents."""
        agents = self.get('agents', {})
        return {
            name: config
            for name, config in agents.items()
            if config.get('enabled', False)
        }

    @property
    def audio_agent_config(self) -> Dict[str, Any]:
        return self.get('agents.audio', {})

    @property
    def webcam_agent_config(self) -> Dict[str, Any]:
        return self.get('agents.webcam', {})

    @property
    def screen_capture_agent_config(self) -> Dict[str, Any]:
        return self.get('agents.screen_capture', {})

    @property
    def clct_enabled(self) -> bool:
        return self.get('subconscious.clct_heartbeat.enabled', False)

    @property
    def clct_config(self) -> Dict[str, Any]:
        return self.get('subconscious.clct_heartbeat', {})

    @property
    def executive_enabled(self) -> bool:
        return self.get('executive.enabled', False)

    @property
    def executive_config(self) -> Dict[str, Any]:
        return self.get('executive', {})

    @property
    def logging_level(self) -> str:
        return self.get('logging.level', 'INFO')

    @property
    def logging_file(self) -> str:
        return self.get('logging.file', 'logs/ambient_subconscious.log')

    @property
    def storage_base_path(self) -> Path:
        return Path(self.get('storage.base_path', 'data'))

    @property
    def audio_path(self) -> Path:
        return Path(self.get('storage.audio_path', 'data/audio'))

    @property
    def frames_path(self) -> Path:
        return Path(self.get('storage.frames_path', 'data/frames'))

    @property
    def sessions_path(self) -> Path:
        return Path(self.get('storage.sessions_path', 'data/sessions'))

    @property
    def models_path(self) -> Path:
        return Path(self.get('storage.models_path', 'models'))

    @property
    def data_dir(self) -> Path:
        """Alias for storage_base_path."""
        return self.storage_base_path

    def ensure_directories(self) -> None:
        """Create all configured storage directories if they don't exist."""
        directories = [
            self.storage_base_path,
            self.audio_path,
            self.frames_path,
            self.sessions_path,
            self.models_path,
            Path(self.logging_file).parent,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global config instance (singleton pattern)
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Config instance
    """
    global _global_config

    if _global_config is None:
        _global_config = Config(config_path)

    return _global_config


def setup_logging(config: Optional[Config] = None) -> None:
    """
    Setup logging from configuration.

    Args:
        config: Config instance (if None, uses global config)
    """
    if config is None:
        config = get_config()

    # Ensure log directory exists
    log_file = Path(config.logging_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging_level),
        format=config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(config.logging_file),
            logging.StreamHandler()
        ]
    )

    logger.info("Logging configured")
