from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, LambdaLR,
                                      MultiStepLR, PolynomialLR,
                                      ReduceLROnPlateau, StepLR)

from medlab.registry import LR_SCHEDULERS

LR_SCHEDULERS.register_module(name='MultiStepLR', module=MultiStepLR)
LR_SCHEDULERS.register_module(name='StepLR', module=StepLR)
LR_SCHEDULERS.register_module(name='CosineAnnealingLR', module=CosineAnnealingLR)
LR_SCHEDULERS.register_module(name='CosineAnnealingWarmRestarts', module=CosineAnnealingWarmRestarts)
LR_SCHEDULERS.register_module(name='PolynomialLR', module=PolynomialLR)
LR_SCHEDULERS.register_module(name='LambdaLR', module=LambdaLR)
LR_SCHEDULERS.register_module(name='ReduceLROnPlateau', module=ReduceLROnPlateau)
