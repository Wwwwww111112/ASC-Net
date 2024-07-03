from monai.networks.nets import (UNETR, AHNet, AttentionUnet,
                                 BasicUNetPlusPlus, DenseNet, EfficientNetBN,
                                 LocalNet, RegUNet, ResNet, SegResNet, SENet,
                                 SwinUNETR, UNet, ViT, VNet, densenet121,
                                 densenet169, densenet201, densenet264,
                                 resnet10, resnet18, resnet34, resnet50,
                                 resnet101, resnet152, senet154, seresnet50,
                                 seresnet101, seresnet152, seresnext50,
                                 seresnext101)

from medlab.registry import MODELS

MODELS.register_module(name='resnet_monai', module=ResNet)
MODELS.register_module(name='resnet10_monai', module=resnet10)
MODELS.register_module(name='resnet18_monai', module=resnet18)
MODELS.register_module(name='resnet34_monai', module=resnet34)
MODELS.register_module(name='resnet50_monai', module=resnet50)
MODELS.register_module(name='resnet101_monai', module=resnet101)
MODELS.register_module(name='resnet152_monai', module=resnet152)
MODELS.register_module(name='densenet_monai', module=DenseNet)
MODELS.register_module(name='densenet121_monai', module=densenet121)
MODELS.register_module(name='densenet169_monai', module=densenet169)
MODELS.register_module(name='densenet201_monai', module=densenet201)
MODELS.register_module(name='densenet264_monai', module=densenet264)
MODELS.register_module(name='senet_monai', module=SENet)
MODELS.register_module(name='senet154_monai', module=senet154)
MODELS.register_module(name='seresnet50_monai', module=seresnet50)
MODELS.register_module(name='seresnet101_monai', module=seresnet101)
MODELS.register_module(name='seresnet152_monai', module=seresnet152)
MODELS.register_module(name='seresnext50_monai', module=seresnext50)
MODELS.register_module(name='seresnext101_monai', module=seresnext101)
MODELS.register_module(name='efficientnet_monai', module=EfficientNetBN)
MODELS.register_module(name='vit_monai', module=ViT)

MODELS.register_module(name='unet_monai', module=UNet)
MODELS.register_module(name='vnet_monai', module=VNet)
MODELS.register_module(name='swinunetr_monai', module=SwinUNETR)
MODELS.register_module(name='unetr_monai', module=UNETR)
MODELS.register_module(name='ahnet_monai', module=AHNet)
MODELS.register_module(name='unetplusplus_monai', module=BasicUNetPlusPlus)
MODELS.register_module(name='attentionunet_monai', module=AttentionUnet)
MODELS.register_module(name='regunet_monai', module=RegUNet)
MODELS.register_module(name='localnet_monai', module=LocalNet)
MODELS.register_module(name='segresnet_monai', module=SegResNet)
