from torchvision.models._api import get_model_builder, list_models

from medlab.registry import MODELS

for model_name in list_models():
    MODELS.register_module(name=model_name + '_torchvision', module=get_model_builder(model_name))
