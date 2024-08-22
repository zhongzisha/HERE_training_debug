
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from transformers import ResNetModel, BeitModel, BitModel, ConvNextModel, CvtModel, DeiTModel, \
    DinatModel, DPTModel, EfficientFormerModel, GLPNModel, MobileNetV1Model, ImageGPTModel, \
    LevitModel, MobileNetV1Model, MobileNetV2Model, MobileViTModel, NatModel, PoolFormerModel, \
    SwinModel, Swinv2Model, ViTModel, ViTHybridModel, ViTMAEModel, ViTMSNModel, CLIPModel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoImageProcessor, CLIPProcessor
from torchvision import transforms
import os
import torch
import torch.nn as nn
from model import AttentionModel, Attn_Net_Gated
from utils import get_svs_prefix, _assertLevelDownsamplesV2, new_web_annotation
from common import HF_MODELS_DICT, BACKBONE_DICT
from dataset import PatchDatasetV2

from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch.utils.tensorboard import SummaryWriter
import torchinfo


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        dropout = 0.25
        self.backbone = backbone

        if backbone == 'PLIP':
            self.backbone_model = CLIPModel.from_pretrained('vinid/plip')
        elif backbone == 'ProvGigaPath':
            self.backbone_model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        elif backbone == 'CONCH':
            from conch.open_clip_custom import create_model_from_pretrained 
            self.backbone_model, image_processor = create_model_from_pretrained('conch_ViT-B-16','./CONCH_weights_pytorch_model.bin')
        elif backbone == 'UNI':
            self.backbone_model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            self.backbone_model.load_state_dict(torch.load("./UNI_pytorch_model.bin", map_location="cpu"), strict=True)
        self.attention_net = nn.Sequential(*[
            nn.Linear(BACKBONE_DICT[backbone], 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            Attn_Net_Gated(L=256, D=256, dropout=dropout, n_classes=1)
        ])

    def forward(self, x, attention_only=False):
        if self.backbone == 'ProvGigaPath':
            x = self.backbone_model(x)
        elif self.backbone == 'CONCH':
            x = self.backbone_model.encode_image(x, proj_contrast=False, normalize=False)
        elif self.backbone == 'PLIP':
            x = self.backbone_model.get_image_features(x)
        elif self.backbone == 'UNI':
            x = self.backbone_model(x)
        x_path = x.unsqueeze(0)
        x = self.attention_net(x_path)
        return x


fp = open('/data/zhongz2/temp29/flops_test.txt', 'w')
x = torch.randn(1, 3, 224, 224)
total_flops = {}
for backbone in ['PLIP', 'CONCH', 'ProvGigaPath', 'UNI']:
    model = Model(backbone)
    model.eval()
    with torch.no_grad():
        y = model(x)

    flops = FlopCountAnalysis(model, x)
    txt = flop_count_table(flops)
    total_flops[backbone] = flops.total()

    fp.write(backbone + " ="*20 + "\n")
    fp.writelines(txt)
    fp.write('\n\n\n')
fp.close()







