from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
import os
import importlib
from functools import wraps
from inspect import signature


def parse_config_from_yaml(config_path: str) -> DictConfig:
    """Parse config from yaml file."""
    config = OmegaConf.load(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    config = OmegaConf.to_container(config, resolve=True)

    if config.get("experiment_name") is None:
        config["experiment_name"] = config_name
    return config


def parse_config_from_cli():
    """Parse config from command line arguments."""
    config = OmegaConf.from_cli()

    if "config" in config:
        base_config = OmegaConf.load(config.config)
        config = OmegaConf.merge(base_config, config)

    config_name = os.path.splitext(os.path.basename(config.config))[0]
    config = OmegaConf.to_container(config, resolve=True)

    if config.get("experiment_name") is None:
        config["experiment_name"] = config_name
    return config


def init_from_config(config: Dict[str, Any], reload: bool = False):
    """Initialize object from config."""
    cfg = config.copy()
    target_key = "_target_"
    assert target_key in cfg, f"Key {target_key} is required for object initialization!"

    module, cls = cfg.pop(target_key).rsplit(".", 1)
    module = importlib.import_module(module)
    if reload:
        module = importlib.reload(module)
    return getattr(module, cls)(**cfg)


class ConfigMixin:
    def __init_subclass__(cls):
        super().__init_subclass__()
        if "__init__" in cls.__dict__:
            setattr(cls, "__init__", cls.get_initial_args_wrapper(cls.__init__))

    @staticmethod
    def get_initial_args_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            default_args = {
                key: val.default
                for key, val in signature(func).parameters.items()
                if key not in ("self", "args", "kwargs")
            }
            for key, override_val in zip(default_args, args):
                default_args[key] = override_val
            default_args.update(kwargs)

            self._default_args = OmegaConf.create(default_args)
            return func(self, *args, **kwargs)

        return wrapper

    @property
    def config(self):
        cls_name = ".".join([self.__class__.__module__, self.__class__.__qualname__])
        cfg = OmegaConf.create(dict(_target_=cls_name))
        if hasattr(self, "_default_args"):
            cfg.update(self._default_args.copy())
        return cfg
    
    def register_to_config(self, **kwargs):
        if hasattr(self, "_default_args"):
            self._default_args.update(kwargs.copy())
        else:
            self._default_args = OmegaConf.create(kwargs.copy())

    @classmethod
    def from_config(cls, config: dict):
        config = config.copy()
        cls_name = cls_name = ".".join(
            [cls.__class__.__module__, cls.__class__.__qualname__]
        )
        cfg_target = config.pop("_target_")
        assert cfg_target == cls_name, f"Expected target {cls_name}, got {cfg_target}"
        return cls(**config)
