from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from medlab.registry import LOGGERS

LOGGERS.register_module(name='TensorBoardLogger', module=TensorBoardLogger)
LOGGERS.register_module(name='CSVLogger', module=CSVLogger)
