from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from medlab.registry import CALLBACKS

CALLBACKS.register_module(name='ModelCheckpoint', module=ModelCheckpoint)
CALLBACKS.register_module(name='LearningRateMonitor', module=LearningRateMonitor)
CALLBACKS.register_module(name='EarlyStopping', module=EarlyStopping)
