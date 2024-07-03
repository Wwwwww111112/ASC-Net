from mmengine.registry import Registry

from .build_functions import (build_callback_from_cfg, build_logger_from_cfg,
                              build_metric_from_cfg, build_model_from_cfg,
                              build_transform_from_cfg)

# manage PyTorch-lightning Trainer
TRAINERS = Registry('trainer', locations=['medlab.core'])
# manage PyTorch-lightning Callbacks
CALLBACKS = Registry('callback', build_func=build_callback_from_cfg, locations=['medlab.core.callbacks'])
# manage PyTorch-lightning loggers
LOGGERS = Registry('logger', build_func=build_logger_from_cfg, locations=['medlab.core.loggers'])
# manage different vision tasks based on LightningModule
TASKS = Registry('task', locations=['medlab.tasks'])

# manage datasets
DATASETS = Registry('dataset', locations=['medlab.datasets'])
# manage MONAI transforms
TRANSFORMS = Registry('transform', build_func=build_transform_from_cfg, locations=['medlab.core.transforms'])
# manage MONAI inferers
INFERERS = Registry('inferer', locations=['medlab.core.inferers'])

# manage PyTorch models
MODELS = Registry('model', build_func=build_model_from_cfg, locations=['medlab.models'])
LOSSES = Registry('loss', locations=['medlab.core.losses'])
OPTIMIZERS = Registry('optimizer', locations=['medlab.core.optimizers'])
LR_SCHEDULERS = Registry('learning rate scheduler', locations=['medlab.core.lr_schedulers'])
METRICS = Registry('metric', build_func=build_metric_from_cfg, locations=['medlab.core.metrics'])
