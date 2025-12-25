"""
Configuration Management for Data Science Master System.

Provides unified configuration loading and management with:
- Multiple sources: YAML, JSON, Environment variables, .env files
- Environment-specific overrides (dev, staging, prod)
- Secrets management integration
- Type validation with Pydantic
- Hot reloading support
- Default values and fallbacks

Example:
    >>> from data_science_master_system.core import ConfigManager
    >>> config = ConfigManager()
    >>> config.load("configs/base.yaml")
    >>> db_url = config.get("database.url")
    >>> config.set("model.n_estimators", 100)
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from copy import deepcopy

import yaml

from data_science_master_system.core.exceptions import ConfigError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ConfigManager:
    """
    Unified configuration manager supporting multiple sources.
    
    Supports:
        - YAML files
        - JSON files
        - Environment variables
        - .env files
        - Programmatic overrides
        
    Configuration is loaded in layers with later sources overriding earlier ones.
    
    Example:
        >>> config = ConfigManager()
        >>> config.load("base.yaml")       # Load base config
        >>> config.load("local.yaml")      # Override with local
        >>> config.load_env()              # Override with env vars
        >>> db_host = config.get("database.host", default="localhost")
    """
    
    _instance: Optional["ConfigManager"] = None
    
    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigManager":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, env_prefix: str = "DSMS_") -> None:
        """
        Initialize the configuration manager.
        
        Args:
            env_prefix: Prefix for environment variables
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self._config: Dict[str, Any] = {}
        self._env_prefix = env_prefix
        self._loaded_files: List[Path] = []
        self._initialized = True
        
        logger.debug("ConfigManager initialized", env_prefix=env_prefix)
    
    def load(self, path: Union[str, Path], required: bool = True) -> "ConfigManager":
        """
        Load configuration from a file.
        
        Supports YAML (.yaml, .yml) and JSON (.json) files.
        
        Args:
            path: Path to configuration file
            required: Whether to raise error if file not found
            
        Returns:
            Self for chaining
            
        Raises:
            ConfigError: If file not found (when required)
        """
        path = Path(path)
        
        if not path.exists():
            if required:
                raise ConfigError(
                    f"Configuration file not found: {path}",
                    context={"path": str(path)},
                )
            logger.warning(f"Optional config file not found: {path}")
            return self
        
        try:
            if path.suffix in (".yaml", ".yml"):
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            elif path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                raise ConfigError(
                    f"Unsupported config format: {path.suffix}",
                    context={"path": str(path), "suffix": path.suffix},
                )
            
            self._deep_merge(self._config, data)
            self._loaded_files.append(path)
            logger.info(f"Loaded configuration from {path}")
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigError(
                f"Failed to parse configuration file: {path}",
                context={"path": str(path), "error": str(e)},
            )
        
        return self
    
    def load_env(
        self,
        prefix: Optional[str] = None,
        dotenv_path: Optional[Union[str, Path]] = None,
    ) -> "ConfigManager":
        """
        Load configuration from environment variables.
        
        Environment variables are converted to nested config using double underscores.
        Example: DSMS_DATABASE__HOST -> database.host
        
        Args:
            prefix: Override the default prefix
            dotenv_path: Path to .env file to load first
            
        Returns:
            Self for chaining
        """
        # Load .env file if specified
        if dotenv_path:
            dotenv_path = Path(dotenv_path)
            if dotenv_path.exists():
                try:
                    from dotenv import load_dotenv
                    load_dotenv(dotenv_path)
                    logger.info(f"Loaded .env from {dotenv_path}")
                except ImportError:
                    logger.warning("python-dotenv not installed, skipping .env loading")
        
        prefix = prefix or self._env_prefix
        env_config: Dict[str, Any] = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested key
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                
                # Convert value type
                value = self._convert_env_value(value)
                
                # Build nested dict
                current = env_config
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
        
        if env_config:
            self._deep_merge(self._config, env_config)
            logger.debug(f"Loaded {len(env_config)} env vars with prefix {prefix}")
        
        return self
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert string environment value to appropriate type."""
        # Boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False
        
        # None
        if value.lower() in ("null", "none", ""):
            return None
        
        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # JSON (for lists/dicts)
        if value.startswith(("[", "{")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        return value
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries, modifying base in place."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)
        return base
    
    def get(
        self,
        key: str,
        default: Optional[T] = None,
        required: bool = False,
    ) -> Optional[T]:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "database.host")
            default: Default value if key not found
            required: Raise error if key not found
            
        Returns:
            Configuration value or default
            
        Raises:
            ConfigError: If required key not found
            
        Example:
            >>> db_host = config.get("database.host", default="localhost")
            >>> api_key = config.get("api.key", required=True)
        """
        parts = key.split(".")
        value: Any = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                if required:
                    raise ConfigError(
                        f"Required configuration key not found: {key}",
                        context={"key": key},
                    )
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> "ConfigManager":
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "database.host")
            value: Value to set
            
        Returns:
            Self for chaining
        """
        parts = key.split(".")
        current = self._config
        
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        
        current[parts[-1]] = value
        return self
    
    def get_section(self, key: str) -> Dict[str, Any]:
        """
        Get a configuration section as a dictionary.
        
        Args:
            key: Section key (e.g., "database")
            
        Returns:
            Section as dictionary or empty dict
        """
        value = self.get(key, default={})
        if not isinstance(value, dict):
            return {}
        return deepcopy(value)
    
    def as_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return deepcopy(self._config)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {path}")
    
    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=indent)
        logger.info(f"Saved configuration to {path}")
    
    def validate(self, schema: Type) -> Any:
        """
        Validate configuration against a Pydantic model.
        
        Args:
            schema: Pydantic model class
            
        Returns:
            Validated Pydantic model instance
            
        Example:
            >>> from pydantic import BaseModel
            >>> class DatabaseConfig(BaseModel):
            ...     host: str
            ...     port: int = 5432
            >>> db_config = config.get_section("database")
            >>> validated = config.validate_section("database", DatabaseConfig)
        """
        try:
            return schema(**self._config)
        except Exception as e:
            raise ConfigError(
                f"Configuration validation failed: {e}",
                context={"schema": schema.__name__, "error": str(e)},
            )
    
    def reset(self) -> "ConfigManager":
        """Reset configuration to empty state."""
        self._config = {}
        self._loaded_files = []
        logger.debug("Configuration reset")
        return self
    
    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __repr__(self) -> str:
        return f"ConfigManager(loaded_files={len(self._loaded_files)}, keys={len(self._config)})"


# =============================================================================
# Convenience Functions
# =============================================================================

def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return ConfigManager.get_instance()


def load_config(
    base_path: Optional[Union[str, Path]] = None,
    env: Optional[str] = None,
    load_env_vars: bool = True,
) -> ConfigManager:
    """
    Load configuration with common patterns.
    
    Loads in order:
        1. Base configuration (if provided)
        2. Environment-specific configuration (if env provided)
        3. Local overrides (configs/local.yaml if exists)
        4. Environment variables (if load_env_vars)
    
    Args:
        base_path: Path to base configuration file
        env: Environment name (dev, staging, prod)
        load_env_vars: Whether to load environment variables
        
    Returns:
        Configured ConfigManager instance
    """
    config = ConfigManager()
    
    if base_path:
        config.load(base_path)
    
    if env:
        # Look for environment-specific config
        base = Path(base_path).parent if base_path else Path("configs")
        env_path = base / f"{env}.yaml"
        config.load(env_path, required=False)
    
    # Local overrides
    if base_path:
        local_path = Path(base_path).parent / "local.yaml"
        config.load(local_path, required=False)
    
    if load_env_vars:
        config.load_env()
    
    return config
