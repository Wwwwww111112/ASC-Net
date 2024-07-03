from timm import create_model

from medlab.registry import MODELS

MODELS.register_module(name='timm_model', module=create_model)
