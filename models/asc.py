import torch
from timm import create_model
from torch import nn

from medlab.registry import MODELS
@MODELS.register_module()
class ASC(nn.Module):
    def __init__(self, num_classes, drop_out=0.4, img_backbone='inception_v3.tv_in1k', shape_backbone='vgg16_bn.tv_in1k', backbone_pretrained=False, feature_dim=1000, attn_dim=256):
        super().__init__()

        self.img_backbone = create_model(model_name=img_backbone, pretrained=backbone_pretrained, num_classes=feature_dim)
        self.shape_backbone = create_model(model_name=shape_backbone, pretrained=backbone_pretrained, num_classes=feature_dim, in_chans=2)

        # self.shape_attention = ShapeAttentionNet(feature_dim, attn_dim)
        self.shape_attention = ShapeAttentionTransformer(in_channels=feature_dim, attn_dim=attn_dim)
        self.cls = nn.Sequential(
            nn.Linear(3*feature_dim, feature_dim//2), 
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(feature_dim//2, feature_dim//4),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(feature_dim//4, num_classes)
        )

        self._initialize_weights()

    def forward(self, img, cistern, cerebellum):

        img_feat = self.img_backbone(img)
        shape_feat = self.shape_backbone(torch.cat([cistern, cerebellum], dim=1))
        shape_attn = self.shape_attention(shape_feat)
        shape_attn = img_feat * shape_attn
        feat = torch.cat([img_feat, shape_attn, shape_feat], dim=1)
        out = self.cls(feat)

        return out
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ShapeAttentionTransformer(nn.Module):

    def __init__(self, in_channels, attn_dim) -> None:
        super().__init__()

        self.transformer_attn = nn.TransformerEncoderLayer(d_model=in_channels, nhead=1, dim_feedforward=attn_dim)
        
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Used transformer-based attention
        out = self.transformer_attn(x)
    
        out = out.view(out.size(0), out.size(1), 1, 1)
      
        out = self.global_avg_pooling(out)  
      
        out = out.view(out.size(0), -1)  
       
        out = self.fc(out)
        out = self.sigmoid(out)
        # print(out.shape)
        return out


if __name__ == '__main__':
    model = ASC(2)
    # print(model(torch.randn(4, 3, 224, 224), torch.randn(4, 1, 224, 224), torch.randn(4, 1, 224, 224)).shape)
    print(model(torch.randn(4, 3, 224, 224), torch.randn(4, 1, 224, 224), torch.randn(4, 1, 224, 224)).shape)