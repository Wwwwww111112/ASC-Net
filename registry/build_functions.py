from typing import Optional, Union

import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.registry import Registry, build_from_cfg
from monai.transforms import Compose


def build_model_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
) -> 'nn.Module':
    """
    Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.
    :param cfg: model config
    :param registry: model registry
    :param default_args: default arguments for the build function
    :return: nn.Module
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(_cfg, registry, default_args) for _cfg in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_logger_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
):
    """
    Build loggers from config dict(s).
    :param cfg: logger config
    :param registry: logger registry
    :param default_args: default arguments for the build function
    :return: PyTorch-lightning logger(s)
    """
    if isinstance(cfg, list):
        loggers = []
        for _cfg in cfg:
            loggers.append(build_from_cfg(_cfg, registry, default_args))
        return loggers
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_callback_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
):
    """
    Build callbacks from config dict(s).
    :param cfg: callback config
    :param registry: callback registry
    :param default_args: default arguments for the build function
    :return: pytorch-lightning callback(s)
    """
    if isinstance(cfg, list):
        callbacks = [build_from_cfg(_cfg, registry, default_args) for _cfg in cfg]
        return callbacks
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_transform_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
):
    """
    Build transforms from config dict(s).
    :param cfg: transform config
    :param registry: transform registry
    :param default_args: default arguments for the build function
    :return: Compose transform(s) from MONAI
    """
    if isinstance(cfg, list):
        transforms = [
            build_from_cfg(_cfg, registry, default_args) for _cfg in cfg
        ]
        return Compose(transforms)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_metric_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
):
    """
    Build metrics from config dict(s).
    :param cfg: metric config
    :param registry: metric registry
    :param default_args: default arguments for the build function
    :return: Metric name(s) and metric(s)
    """
    keys = []
    metrics = []

    def get_metric_key(args):
        if args.get('type') == 'ConfusionMatrixMetric':
            key = args.get('metric_name')
            if isinstance(key, list):
                return key
            else:
                return [key]
        else:
            return [args.get('type')]

    if isinstance(cfg, list):
        for _cfg in cfg:
            metrics.append(build_from_cfg(_cfg, registry, default_args))
            keys.extend(get_metric_key(_cfg))
        return keys, metrics
    else:
        return get_metric_key(cfg), [build_from_cfg(cfg, registry, default_args)]
