
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
import ResNet as ResNet


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
        elif backbone == 'Yottixel':
            # DenseNet121
            self.backbone_model = timm.create_model('densenet121', pretrained=True)
        elif backbone == 'SISH':
            # DenseNet121
            self.backbone_model = timm.create_model('densenet121', pretrained=True)
        elif backbone == 'MobileNetV3':
            self.backbone_model = timm.create_model('mobilenetv3_large_100', pretrained=True) 
        elif backbone == 'HIPT':
            # ViT-small
            import vision_transformer as vits
            self.backbone_model = vits.__dict__['vit_small'](patch_size=16)  # 256x256
            state_dict = torch.load('./HIPT_vit256_small_dino.pth', map_location="cpu")
        elif backbone == 'CLIP':
            # openai/clip-vit-base-patch32
            self.backbone_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        elif backbone == 'RetCCL':
            self.backbone_model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            pretext_model = torch.load('./RetCCL_best_ckpt.pth')
            self.backbone_model.fc = nn.Identity()
            self.backbone_model.load_state_dict(pretext_model, strict=True)

        # self.attention_net = nn.Sequential(*[
        #     nn.Linear(BACKBONE_DICT[backbone], 256), 
        #     nn.ReLU(), 
        #     nn.Dropout(dropout),
        #     Attn_Net_Gated(L=256, D=256, dropout=dropout, n_classes=1)
        # ])

    def forward(self, x, attention_only=False):
        if self.backbone == 'ProvGigaPath':
            x = self.backbone_model(x)
        elif self.backbone == 'CONCH':
            x = self.backbone_model.encode_image(x, proj_contrast=False, normalize=False)
        elif self.backbone in ['PLIP', 'CLIP']:
            x = self.backbone_model.get_image_features(x)
        elif self.backbone == 'UNI':
            x = self.backbone_model(x)
        elif self.backbone in ['Yottixel', 'SISH', 'MobileNetV3', 'HIPT', 'RetCCL']:
            x = self.backbone_model(x)
        # x_path = x.unsqueeze(0)
        # x = self.attention_net(x_path)
        return x


fp = open('/data/zhongz2/temp29/flops_test.txt', 'w')
x = torch.randn(1, 3, 224, 224)
x_yottixel = torch.randn(1, 3, 1000, 1000)
total_flops = {}
for backbone in ['PLIP', 'CONCH', 'ProvGigaPath', 'UNI', 'Yottixel', 'SISH', 'MobileNetV3', 'HIPT', 'CLIP', 'RetCCL']:
    print(backbone)
    model = Model(backbone)
    model.eval()
    with torch.no_grad():
        y = model(x_yottixel) if backbone == 'Yottixel' else model(x)

        pytorch_total_params = sum(p.numel() for p in model.parameters())

    flops = FlopCountAnalysis(model, x)
    txt = flop_count_table(flops)
    total_flops[backbone] = (pytorch_total_params, flops.total())

    fp.write(backbone + " ="*20 + "\n")
    fp.writelines(txt)
    fp.write('\n\n\n')
fp.close()
print('total_flops')
print(total_flops)

# {'PLIP': (151277313, 4413615360), 'CONCH': (395232769, 17738386944), 'ProvGigaPath': (1134953984, 228217640448), 'UNI': (303350784, 61603111936), 'Yottixel': (7978856, 2865546752), 'SISH': (7978856, 2865546752), 'MobileNetV3': (5483032, 225436416), 'HIPT': (21665664, 4607954304), 'CLIP': (151277313, 4413615360), 'RetCCL': (23508032, 4109464576)}



