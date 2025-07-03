# distributed_training/config.py
"""
Configuration management for Distributed Training Framework.

Supports loading from:
1. YAML config file (config.yaml)
2. Environment variables (.env file)
3. Direct environment variables
4. Default values

Priority (highest to lowest):
1. Environment variables
2. .env file
3. config.yaml file
4. Default values
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for distributed training"""
    
    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Path to YAML config file (default: config.yaml)
            env_file: Path to .env file (default: .env)
        """
        self.config_file = config_file or "config.yaml"
        self.env_file = env_file or ".env"
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from all sources"""
        # 1. Start with defaults
        self._config = self._get_defaults()
        
        # 2. Load from YAML config file
        self._load_yaml_config()
        
        # 3. Load from .env file
        self._load_env_file()
        
        # 4. Override with environment variables
        self._load_env_variables()
        
        logger.debug(f"Configuration loaded: {self._config}")
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'network': {
                'gpu_host': '192.168.1.100',
                'cpu_host': '192.168.1.200',
                'port': 29500,
                'timeout': 30,
                'retry_attempts': 3
            },
            'training': {
                'batch_size': 64,
                'num_workers': 4,
                'epochs': 10,
                'learning_rate': 0.001
            },
            'data': {
                'shuffle': True,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 4
            },
            'gpu': {
                'device': 'auto',
                'mixed_precision': False,
                'compile_model': False
            },
            'monitoring': {
                'progress_bars': True,
                'log_level': 'INFO',
                'save_logs': False,
                'log_file': 'training.log'
            },
            'colab': {
                'ngrok_auth_token': None,
                'tunnel_region': 'us'
            },
            'advanced': {
                'flow_control': True,
                'queue_size_limit': 200,
                'memory_threshold': 0.8,
                'heartbeat_interval': 5
            }
        }
    
    def _load_yaml_config(self):
        """Load configuration from YAML file"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self._merge_config(yaml_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        else:
            logger.debug(f"Config file {config_path} not found, using defaults")
    
    def _load_env_file(self):
        """Load configuration from .env file"""
        env_path = Path(self.env_file)
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                logger.info(f"Loaded environment variables from {env_path}")
            except Exception as e:
                logger.warning(f"Failed to load .env file {env_path}: {e}")
    
    def _load_env_variables(self):
        """Load configuration from environment variables"""
        env_mapping = {
            'GPU_HOST': ['network', 'gpu_host'],
            'CPU_HOST': ['network', 'cpu_host'],
            'TRAINING_PORT': ['network', 'port'],
            'BATCH_SIZE': ['training', 'batch_size'],
            'NUM_WORKERS': ['training', 'num_workers'],
            'EPOCHS': ['training', 'epochs'],
            'LEARNING_RATE': ['training', 'learning_rate'],
            'GPU_DEVICE': ['gpu', 'device'],
            'MIXED_PRECISION': ['gpu', 'mixed_precision'],
            'LOG_LEVEL': ['monitoring', 'log_level'],
            'PROGRESS_BARS': ['monitoring', 'progress_bars'],
            'NGROK_AUTH_TOKEN': ['colab', 'ngrok_auth_token'],
            'NGROK_REGION': ['colab', 'tunnel_region'],
            'FLOW_CONTROL': ['advanced', 'flow_control'],
            'QUEUE_SIZE_LIMIT': ['advanced', 'queue_size_limit']
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_value(value)
                self._set_nested_value(config_path, converted_value)
    
    def _convert_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert string value to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string if no conversion applies
        return value
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration into existing config"""
        for key, value in new_config.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value
    
    def _set_nested_value(self, path: list, value: Any):
        """Set a nested configuration value"""
        config = self._config
        for key in path[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[path[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'network.gpu_host')"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_network_config(self) -> Dict[str, Any]:
        """Get network configuration"""
        return self._config.get('network', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self._config.get('training', {})
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration"""
        return self._config.get('gpu', {})
    
    def get_colab_config(self) -> Dict[str, Any]:
        """Get Colab configuration"""
        return self._config.get('colab', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dict-like setting"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


# Global configuration instance
_global_config = None

def get_config(config_file: Optional[str] = None, env_file: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_file, env_file)
    return _global_config

def reload_config(config_file: Optional[str] = None, env_file: Optional[str] = None):
    """Reload global configuration"""
    global _global_config
    _global_config = Config(config_file, env_file)
    return _global_config

# Convenience functions
def get_gpu_host() -> str:
    """Get GPU host from configuration"""
    return get_config().get('network.gpu_host', '192.168.1.100')

def get_cpu_host() -> str:
    """Get CPU host from configuration"""
    return get_config().get('network.cpu_host', '192.168.1.200')

def get_port() -> int:
    """Get communication port from configuration"""
    return get_config().get('network.port', 29500)

def get_batch_size() -> int:
    """Get batch size from configuration"""
    return get_config().get('training.batch_size', 64)

def get_num_workers() -> int:
    """Get number of workers from configuration"""
    return get_config().get('training.num_workers', 4)