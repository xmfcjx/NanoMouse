import yaml
import os
from pathlib import Path

_CONFIG = None

def _get_config_path() -> str:
    config_dir = Path(__file__).parent
    return str(config_dir / "default.yaml")

def load_config() -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    
    config_path = _get_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG

def get_config(key: str = None, default=None):
    config = load_config()
    if key is None:
        return config
    
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value

class Config:
    @property
    def model(self) -> dict:
        return get_config("model", {})
    
    @property
    def embedding(self) -> dict:
        return get_config("embedding", {})
    
    @property
    def rag(self) -> dict:
        return get_config("rag", {})
    
    @property
    def agent(self) -> dict:
        return get_config("agent", {})
    
    @property
    def retriever(self) -> dict:
        return get_config("retriever", {})
    
    @property
    def paths(self) -> dict:
        return get_config("paths", {})

config = Config()
