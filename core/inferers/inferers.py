from monai.inferers import SimpleInferer, SlidingWindowInferer

from medlab.registry import INFERERS

INFERERS.register_module(name='SimpleInferer', module=SimpleInferer)
INFERERS.register_module(name='SlidingWindowInferer', module=SlidingWindowInferer)
