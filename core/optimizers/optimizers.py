from torch.optim import SGD, Adam, AdamW, RMSprop

from medlab.registry import OPTIMIZERS

OPTIMIZERS.register_module(name='SGD', module=SGD)
OPTIMIZERS.register_module(name='Adam', module=Adam)
OPTIMIZERS.register_module(name='AdamW', module=AdamW)
OPTIMIZERS.register_module(name='RMSprop', module=RMSprop)
