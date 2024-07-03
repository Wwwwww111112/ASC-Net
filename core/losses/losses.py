from monai.losses import DeepSupervisionLoss, DiceCELoss, DiceLoss
from torch.nn import (BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, L1Loss,
                      MSELoss)

from medlab.registry import LOSSES

LOSSES.register_module(name='BCELoss', module=BCELoss)
LOSSES.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
LOSSES.register_module(name='BCEWithLogitsLoss', module=BCEWithLogitsLoss)
LOSSES.register_module(name='MSELoss', module=MSELoss)
LOSSES.register_module(name='L1Loss', module=L1Loss)
LOSSES.register_module(name='DiceLoss', module=DiceLoss)
LOSSES.register_module(name='DiceCELoss', module=DiceCELoss)
LOSSES.register_module(name='DeepSupervisionLoss', module=DeepSupervisionLoss)
