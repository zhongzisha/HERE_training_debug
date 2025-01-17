
import sys,os,json,glob
import pandas as pd
import numpy as np
import tarfile
import io
import gc
import re
import faiss
from sklearn.metrics import pairwise_distances, confusion_matrix, classification_report
from collections import Counter
import base64
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import timm
import pickle
import time
from PIL import Image, ImageDraw, ImageFont

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('agg')
import random
import pymysql
import pyarrow.parquet as pq
from transformers import CLIPModel, CLIPProcessor
import psutil
# print(psutil.virtual_memory().used/1024/1024/1024, "GB")
import h5py
import openslide
import cv2

DATA_DIR = f'/data/zhongz2/CPTAC'
# project_names = ['TCGA-COMBINED', 'KenData_20240814', 'ST']  # do not change the order
# project_start_ids = {'TCGA-COMBINED': 0, 'KenData_20240814': 159011314, 'ST': 281115587}
# backbones = ['HERE_CONCH', 'HERE_UNI'] # choices: 'HERE_CONCH', 'HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_UNI'

# cancer_types = ['AML', 'BRCA', 'CCRCC', 'CM', 'COAD', 'GBM', 'HNSCC', 'LSCC', 'LUAD', 'OV', 'PDA', 'SAR', 'UCEC']
# project_names = ['CPTAC'] + [f'CPTAC_{cancer_type}' for cancer_type in cancer_types] # do not change the order
# project_start_ids = {'CPTAC': 0}
# for i, cancer_type in enumerate(cancer_types):
#     project_start_ids[f'CPTAC_{cancer_type}'] = i + 1

project_names = ['CPTAC']
project_start_ids = {'CPTAC': 0}
backbones = ['HERE_CONCH'] # choices: 'HERE_CONCH', 'HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_UNI'
DEVICE = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu') 


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)  # 1 x num_patches x 256
        b = self.attention_b(x)  # 1 x num_patches x 256
        A = a.mul(b)  # 1 x num_patches x 256
        A = self.attention_c(A)  # N x n_tasks, num_patches x 512
        return A, x


"""
/data/zhongz2/results_histo256_generated7fp_hf_TCGA-ALL2_32_2gpus/adam_RegNormNone_Encoderimagenetmobilenetv3_CLSLOSSweighted_ce_accum4_wd1e-4_reguNone1e-4/split_3/snapshot_22.pt
"""

# survival not shared, all other shared
class AttentionModel_bak(nn.Module):
    def __init__(self):
        super().__init__()
        fc = [nn.Linear(1280, 256)]
        self.attention_net = nn.Sequential(*fc)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, label=None, instance_eval=False, return_features=False, attention_only=False):

        x_path = x.squeeze(0)  # 1 x num_patches x 512  1 x 10000 x 512 --> all 256x256 pacthces

        return self.attention_net(x_path)


BACKBONE_DICT = {
    'CLIP': 512,
    'PLIP': 512,
    'MobileNetV3': 1280,
    'mobilenetv3': 1280,
    'ProvGigaPath': 1536,
    'CONCH': 512
}

# survival not shared, all other shared
class AttentionModel(nn.Module):
    def __init__(self, backbone='PLIP'):
        super().__init__()

        self.classification_dict = {
            'CDH1_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'GATA3_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'PIK3CA_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
            'TP53_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'KRAS_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
            'ARID1A_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'PTEN_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'BRAF_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
            'APC_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'ATRX_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'IDH1_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss_Or_Switch', 'Other']
        }
        self.regression_list = [
            'Cytotoxic_T_Lymphocyte',
            'TIDE_CAF',
            'TIDE_Dys',
            'TIDE_M2',
            'TIDE_MDSC',
            'HALLMARK_ADIPOGENESIS_sum',
            'HALLMARK_ALLOGRAFT_REJECTION_sum',
            'HALLMARK_ANDROGEN_RESPONSE_sum',
            'HALLMARK_ANGIOGENESIS_sum',
            'HALLMARK_APICAL_JUNCTION_sum',
            'HALLMARK_APICAL_SURFACE_sum',
            'HALLMARK_APOPTOSIS_sum',
            'HALLMARK_BILE_ACID_METABOLISM_sum',
            'HALLMARK_CHOLESTEROL_HOMEOSTASIS_sum',
            'HALLMARK_COAGULATION_sum',
            'HALLMARK_COMPLEMENT_sum',
            'HALLMARK_DNA_REPAIR_sum',
            'HALLMARK_E2F_TARGETS_sum',
            'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION_sum',
            'HALLMARK_ESTROGEN_RESPONSE_EARLY_sum',
            'HALLMARK_ESTROGEN_RESPONSE_LATE_sum',
            'HALLMARK_FATTY_ACID_METABOLISM_sum',
            'HALLMARK_G2M_CHECKPOINT_sum',
            'HALLMARK_GLYCOLYSIS_sum',
            'HALLMARK_HEDGEHOG_SIGNALING_sum',
            'HALLMARK_HEME_METABOLISM_sum',
            'HALLMARK_HYPOXIA_sum',
            'HALLMARK_IL2_STAT5_SIGNALING_sum',
            'HALLMARK_IL6_JAK_STAT3_SIGNALING_sum',
            'HALLMARK_INFLAMMATORY_RESPONSE_sum',
            'HALLMARK_INTERFERON_ALPHA_RESPONSE_sum',
            'HALLMARK_INTERFERON_GAMMA_RESPONSE_sum',
            'HALLMARK_KRAS_SIGNALING_DN_sum',
            'HALLMARK_KRAS_SIGNALING_UP_sum',
            'HALLMARK_MITOTIC_SPINDLE_sum',
            'HALLMARK_MTORC1_SIGNALING_sum',
            'HALLMARK_MYC_TARGETS_V1_sum',
            'HALLMARK_MYC_TARGETS_V2_sum',
            'HALLMARK_MYOGENESIS_sum',
            'HALLMARK_NOTCH_SIGNALING_sum',
            'HALLMARK_OXIDATIVE_PHOSPHORYLATION_sum',
            'HALLMARK_P53_PATHWAY_sum',
            'HALLMARK_PANCREAS_BETA_CELLS_sum',
            'HALLMARK_PEROXISOME_sum',
            'HALLMARK_PI3K_AKT_MTOR_SIGNALING_sum',
            'HALLMARK_PROTEIN_SECRETION_sum',
            'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY_sum',
            'HALLMARK_SPERMATOGENESIS_sum',
            'HALLMARK_TGF_BETA_SIGNALING_sum',
            'HALLMARK_TNFA_SIGNALING_VIA_NFKB_sum',
            'HALLMARK_UNFOLDED_PROTEIN_RESPONSE_sum',
            'HALLMARK_UV_RESPONSE_DN_sum',
            'HALLMARK_UV_RESPONSE_UP_sum',
            'HALLMARK_WNT_BETA_CATENIN_SIGNALING_sum',
            'HALLMARK_XENOBIOTIC_METABOLISM_sum'
        ]

        self.attention_net = nn.Sequential(*[
            nn.Linear(BACKBONE_DICT[backbone], 256), 
            nn.ReLU(), 
            nn.Dropout(0.25),
            Attn_Net_Gated(L=256, D=256, dropout=0.25, n_classes=1)
        ])
        self.rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25)])

        classifiers = {}
        for k, labels in self.classification_dict.items():
            classifiers[k] = nn.Linear(256, len(labels))
        self.classifiers = nn.ModuleDict(classifiers)
        regressors = {}
        for k in self.regression_list:
            regressors[k] = nn.Linear(256, 1)
        self.regressors = nn.ModuleDict(regressors)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, label=None, instance_eval=False, return_features=False, attention_only=False):
        x_path = x.squeeze(0)  # 1 x num_patches x 512  1 x 10000 x 512 --> all 256x256 pacthces

        A, h = self.attention_net(x_path)  # num_patches x num_tasks, num_patches x 512
        A = torch.transpose(A, 1, 0)  # num_tasks x num_patches
        # A_raw = A  # 1 x num_patches
        if attention_only:
            return {'A_raw': A}

        results_dict = {}
        A = F.softmax(A, dim=1)  # num_tasks x num_patches, normalized
        h = torch.mm(A, h)  # A: num_tasks x num_patches, h_path: num_patches x 256  --> num_tasks x 256
        results_dict['global_feat'] = h
        results_dict['A'] = A
        h = self.rho(h)

        for k, classifier in self.classifiers.items():
            logits_k = classifier(h[0].unsqueeze(0))
            results_dict[k + '_logits'] = logits_k

        for k, regressor in self.regressors.items():
            values_k = regressor(h[0].unsqueeze(0)).squeeze(1)
            results_dict[k + '_logits'] = values_k

        return results_dict


def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def load_model_config_from_hf(model_id: str):
    cached_file = f'{DATA_DIR}/assets/ProvGigaPath/config.json'

    hf_config = load_cfg_from_json(cached_file)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:  # deprecated name for 'label_names'
            pretrained_cfg['label_names'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg

    # NOTE currently discarding parent config as only arch name and pretrained_cfg used in timm right now
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'

    # model should be created with base config num_classes if its exist
    if 'num_classes' in hf_config:
        pretrained_cfg['num_classes'] = hf_config['num_classes']

    # label meta-data in base config overrides saved pretrained_cfg on load
    if 'label_names' in hf_config:
        pretrained_cfg['label_names'] = hf_config.pop('label_names')
    if 'label_descriptions' in hf_config:
        pretrained_cfg['label_descriptions'] = hf_config.pop('label_descriptions')

    model_args = hf_config.get('model_args', {})
    model_name = hf_config['architecture']
    return pretrained_cfg, model_name, model_args

from timm.layers import set_layer_config
from timm.models import is_model, model_entrypoint, load_checkpoint

def split_model_name_tag(model_name: str, no_tag: str = ''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag

from urllib.parse import urlsplit

def parse_model_name(model_name: str):
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def create_model():
    model_name = 'hf_hub:prov-gigapath/prov-gigapath'
    model_source, model_name = parse_model_name(model_name)
    pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_name)
    kwargs = {}
    if model_args:
        for k, v in model_args.items():
            kwargs.setdefault(k, v)
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=None, exportable=None, no_jit=None):
        model = create_fn(
            pretrained=False,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
            **kwargs
        )
    load_checkpoint(model, f'{DATA_DIR}/assets/ProvGigaPath/pytorch_model.bin')

    return model

print('before loading backbones ', psutil.virtual_memory().used/1024/1024/1024, "GB")
models_dict = {}
for search_backbone in backbones:
    models_dict[search_backbone] = {}
    if search_backbone == 'HERE_PLIP':
        models_dict[search_backbone]['feature_extractor'] = CLIPModel.from_pretrained(f"{DATA_DIR}/assets/vinid_plip")
        models_dict[search_backbone]['image_processor_or_transform'] = CLIPProcessor.from_pretrained(f"{DATA_DIR}/assets/vinid_plip")
        models_dict[search_backbone]['attention_model'] = AttentionModel()
        models_dict[search_backbone]['state_dict'] = torch.load(f"{DATA_DIR}/assets/snapshot_66_HERE_PLIP.pt", map_location=DEVICE) #, weights_only=True)
    elif search_backbone == 'HERE_CONCH':
        from conch.open_clip_custom import create_model_from_pretrained
        models_dict[search_backbone]['feature_extractor'], models_dict[search_backbone]['image_processor_or_transform'] = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=f'{DATA_DIR}/assets/CONCH_weights_pytorch_model.bin')
        models_dict[search_backbone]['attention_model'] = AttentionModel(backbone='CONCH')
        models_dict[search_backbone]['state_dict'] = torch.load(f"{DATA_DIR}/assets/snapshot_53_HERE_CONCH.pt", map_location=DEVICE) #, weights_only=True)
    elif search_backbone == 'HERE_ProvGigaPath':
        models_dict[search_backbone]['feature_extractor'] = create_model()
        models_dict[search_backbone]['image_processor_or_transform'] = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        models_dict[search_backbone]['attention_model'] = AttentionModel(backbone='ProvGigaPath')
        models_dict[search_backbone]['state_dict'] = torch.load(f"{DATA_DIR}/assets/snapshot_39_HERE_ProvGigaPath.pt", map_location=DEVICE) #, weights_only=True)
    elif search_backbone == 'HERE_UNI':
        models_dict[search_backbone]['feature_extractor'] = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        # models_dict[search_backbone]['feature_extractor'].load_state_dict(torch.load(f"{DATA_DIR}/assets/UNI_pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
        models_dict[search_backbone]['feature_extractor'].load_state_dict(torch.load(f"{DATA_DIR}/assets/UNI_pytorch_model.bin", map_location=DEVICE), strict=True)
        models_dict[search_backbone]['image_processor_or_transform'] = create_transform(**resolve_data_config(models_dict[search_backbone]['feature_extractor'].pretrained_cfg, model=models_dict[search_backbone]['feature_extractor']))
        models_dict[search_backbone]['attention_model'] = AttentionModel(backbone='UNI')
        models_dict[search_backbone]['state_dict'] = torch.load(f"{DATA_DIR}/assets/snapshot_58_HERE_UNI.pt", map_location=DEVICE)#, weights_only=True)
    
    models_dict[search_backbone]['feature_extractor'].to(DEVICE)
    models_dict[search_backbone]['feature_extractor'].eval()
    models_dict[search_backbone]['attention_model'].load_state_dict(models_dict[search_backbone]['state_dict']['MODEL_STATE'], strict=False)
    models_dict[search_backbone]['attention_model'].to(DEVICE)
    models_dict[search_backbone]['attention_model'].eval()

print('after loading backbones ', psutil.virtual_memory().used/1024/1024/1024, "GB")

faiss_indexes = {}
# faiss_types = ['faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8', 'faiss_IndexFlatL2']
faiss_types = ['faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8']
for backbone in backbones:
    faiss_indexes[backbone] = {}
    for faiss_type in faiss_types:
        faiss_indexes[backbone][faiss_type] = {}
        for project_name in project_names:
            faiss_bin_filename = f"{DATA_DIR}/assets/faiss_bins/all_data_feat_before_attention_feat_{faiss_type}_{project_name}_{backbone}.bin"
            if os.path.exists(faiss_bin_filename):
                faiss_indexes[backbone][faiss_type][project_name] = faiss.read_index(faiss_bin_filename)

print('after loading faiss indexes ', psutil.virtual_memory().used/1024/1024/1024, "GB")

# with open(f'{DATA_DIR}/assets/randomly_1000_data_with_PLIP_ProvGigaPath_CONCH_20240814.pkl', 'rb') as fp: # with HERE_ProvGigaPath, with CONCH, random normal distribution
#     randomly_1000_data = pickle.load(fp)
randomly_1000_data = {}
for method in ['HERE_CONCH']: #['HERE_ProvGigaPath', 'HERE_CONCH', 'HERE_PLIP', 'HERE_UNI']:
    if method not in randomly_1000_data:
        randomly_1000_data[method] = {}
    for project_name in project_names: #['KenData_20240814', 'ST_20240903', 'TCGA-COMBINED']:
        if project_name == 'ST_20240903':
            version = 'V20240908'
        elif project_name[:5] == 'CPTAC':
            version = 'V20240908'
        else:
            version = 'V6'
        filename = f'{DATA_DIR}/assets/randomly_background_samples_for_train_{project_name}_{method}{version}.pkl'
        if project_name in randomly_1000_data[method] or not os.path.exists(filename):
            print('wrong')
            import pdb
            pdb.set_trace()
            continue
        with open(filename, 'rb') as fp:
            data1 = pickle.load(fp)
        if project_name == 'TCGA-COMBINED' or project_name == 'KenData_20240814':
            embeddings = data1[method][project_name]['embeddings']
            randomly_1000_data[method][project_name] = embeddings[np.random.randint(0, len(embeddings), 10000), :]
        else:
            randomly_1000_data[method][project_name] = data1[method][project_name]['embeddings']
    # randomly_1000_data[method]['ALL'] = np.concatenate([
    #     vv for kk, vv in randomly_1000_data[method].items() 
    # ])
    randomly_1000_data[method]['ALL'] = randomly_1000_data[method]['CPTAC']


font = ImageFont.truetype("Gidole-Regular.ttf", size=36)
print('before loading lmdb ', psutil.virtual_memory().used/1024/1024/1024, "GB")



def knn_search_images_by_faiss(query_embedding, k=10, search_project="ALL", search_method='faiss', search_backbone='HERE_PLIP'):
    if search_project == 'ALL':
        Ds, Is = {}, {}
        for iiiii, project_name in enumerate(project_names):
            Di, Ii = faiss_indexes[search_backbone][search_method][project_name].search(query_embedding, k)
            Di = np.array([dd for dd, ii in zip(Di[0], Ii[0]) if ii>=0])
            beginid = project_start_ids[project_name]
            Ii = [beginid+ii for ii in Ii[0] if ii>=0]
            Ds[project_name] = Di
            Is[project_name] = Ii

        D = np.concatenate(list(Ds.values()))
        I = np.concatenate(list(Is.values()))
        if 'HNSW' in search_method or 'IndexFlatL2' in search_method:
            inds = np.argsort(D)[:k]
        else:  # IP or cosine similarity, the larger, the better
            inds = np.argsort(D)[::-1][:k]
        return D[inds], I[inds]

    else:
        Di, Ii = faiss_indexes[search_backbone][search_method][search_project].search(query_embedding, k)

        Di = np.array([dd for dd, ii in zip(Di[0], Ii[0]) if ii>=0])
        beginid = project_start_ids[search_project]
        Ii = np.array([beginid+ii for ii in Ii[0] if ii>=0])
        return Di, Ii


def compute_mean_std_cosine_similarity_from_random1000_bak(query_embedding, search_project='ALL', search_backbone='HERE_PLIP'):
    distances = 1 - pairwise_distances(query_embedding.reshape(1, -1),
                                       randomly_1000_data[search_backbone][search_project if search_project in randomly_1000_data[search_backbone].keys() else 'ALL'],
                                       metric='cosine')[0]
    return np.mean(distances), np.std(distances), distances
def compute_mean_std_cosine_similarity_from_random1000(query_embedding, search_project='ALL', search_backbone='HERE_PLIP'):
    distances = pairwise_distances(query_embedding.reshape(1, -1), randomly_1000_data[search_backbone][search_project if search_project in randomly_1000_data[search_backbone].keys() else 'ALL'])[0]
    return np.mean(distances), np.std(distances), distances


def get_image_patches(image, sizex=256, sizey=256):
    # w = 2200
    # h = 2380
    # sizex, sizey = 256, 256
    w, h = image.size
    if w < sizex:
        image1 = Image.new(image.mode, (sizey, h), (0, 0, 0))
        image1.paste(image, ((sizex - w) // 2, 0))
        image = image1
    w, h = image.size
    if h < sizey:
        image1 = Image.new(image.mode, (w, sizex), (0, 0, 0))
        image1.paste(image, (0, (sizey - h) // 2))
        image = image1
    w, h = image.size
    # creating new Image object
    image_shown = image.copy()
    img1 = ImageDraw.Draw(image_shown)

    num_x = np.floor(w / sizex)
    num_y = np.floor(h / sizey)
    box_w = int(num_x * sizex)
    box_y = int(num_y * sizey)
    startx = w // 2 - box_w // 2
    starty = h // 2 - box_y // 2
    patches = []
    r = 5
    patch_coords = []
    for x1 in range(startx, w, sizex):
        x2 = x1 + sizex
        if x2 > w:
            continue
        for y1 in range(starty, h, sizey):
            y2 = y1 + sizey
            if y2 > h:
                continue
            img1.line((x1, y1, x1, y2), fill="white", width=1)
            img1.line((x1, y2, x2, y2), fill="white", width=1)
            img1.line((x2, y2, x2, y1), fill="white", width=1)
            img1.line((x2, y1, x1, y1), fill="white", width=1)
            cx, cy = x1 + sizex // 2, y1 + sizey // 2
            patch_coords.append((cx, cy))
            img1.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 0, 0, 0))
            patches.append(image.crop((x1, y1, x2, y2)))
    return patches, patch_coords, image_shown



def get_query_embedding(img_urls, resize=0, search_backbone='HERE_PLIP'):
    image_patches_all = []
    patch_coords_all = []
    image_shown_all = []
    minWorH = 1e8
    sizex, sizey = 256, 256
    # if 'CONCH' in search_backbone:
    #     sizex, sizey = 512, 512
    for img_url in img_urls:
        if img_url[:4] == 'http':
            image = Image.open(img_url.replace(
                'https://hidare-dev.ccr.cancer.gov/', '')).convert('RGB')
        elif img_url[:4] == 'data':
            image_data = re.sub('^data:image/.+;base64,', '', img_url)
            image = Image.open(io.BytesIO(
                base64.b64decode(image_data))).convert('RGB')
        else:
            image = Image.open(img_url).convert('RGB')

        W, H = image.size        
        minWorH = min(min(W, H), minWorH)
        if 0 < resize:
            resize_scale = 1. / 2**resize
            newW, newH = int(W*resize_scale), int(H*resize_scale)
            minWorH = min(min(newW, newH), minWorH)
            image = image.resize((newW, newH))
        if search_backbone == 'ProvGigaPath':
            image = image.resize((256, 256))
        patches, patch_coords, image_shown = get_image_patches(image, sizex=sizex, sizey=sizey)
        image_patches_all.append(patches)
        patch_coords_all.append(patch_coords)
        image_shown_all.append(image_shown)

    image_patches = [
        patch for patches in image_patches_all for patch in patches]
    image_urls_all = {}
    results_dict = {}
    with torch.no_grad():

        if search_backbone in ['HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_CONCH', 'HERE_UNI']:
            if search_backbone == 'HERE_PLIP':
                images = models_dict[search_backbone]['image_processor_or_transform'](images=image_patches, return_tensors='pt')['pixel_values']
                images = images.to(DEVICE)
                feat_after_encoder_feat = models_dict[search_backbone]['feature_extractor'].get_image_features(images).detach()
                # extract feat_before_attention_feat
                embedding = feat_after_encoder_feat @ models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.weight'].T + \
                    models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.bias']
                # get the attention scores
                results_dict = models_dict[search_backbone]['attention_model'](feat_after_encoder_feat.unsqueeze(0))
            elif search_backbone == 'HERE_ProvGigaPath':
                images = torch.stack([models_dict[search_backbone]['image_processor_or_transform'](example) for example in image_patches])
                images = images.to(DEVICE)
                feat_after_encoder_feat = models_dict[search_backbone]['feature_extractor'](images).detach()
                # extract feat_before_attention_feat
                embedding = feat_after_encoder_feat @ models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.weight'].T + \
                    models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.bias']
                # get the attention scores
                results_dict = models_dict[search_backbone]['attention_model'](feat_after_encoder_feat.unsqueeze(0))
            elif search_backbone == 'HERE_CONCH':
                images = torch.stack([models_dict[search_backbone]['image_processor_or_transform'](example) for example in image_patches])
                images = images.to(DEVICE)
                feat_after_encoder_feat = models_dict[search_backbone]['feature_extractor'].encode_image(images, proj_contrast=False, normalize=False).detach()
                # extract feat_before_attention_feat
                embedding = feat_after_encoder_feat @ models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.weight'].T + \
                    models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.bias']
                # get the attention scores
                results_dict = models_dict[search_backbone]['attention_model'](feat_after_encoder_feat.unsqueeze(0))
            elif search_backbone == 'HERE_UNI':
                images = torch.stack([models_dict[search_backbone]['image_processor_or_transform'](example) for example in image_patches])
                images = images.to(DEVICE)
                feat_after_encoder_feat = models_dict[search_backbone]['feature_extractor'](images).detach()
                # extract feat_before_attention_feat
                embedding = feat_after_encoder_feat @ models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.weight'].T + \
                    models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.bias']
                # get the attention scores
                results_dict = models_dict[search_backbone]['attention_model'](feat_after_encoder_feat.unsqueeze(0))
            # weighted the features using attention scores
            embedding = torch.mm(results_dict['A'], embedding)
            # embedding = results_dict['global_feat'].detach().numpy()
            embedding = embedding.detach().cpu().numpy()

            if len(image_patches_all) > 1:
                atten_scores = np.split(results_dict['A'].detach().cpu().numpy()[0], np.cumsum(
                    [len(patches) for patches in image_patches_all])[:-1])
            else:
                atten_scores = [results_dict['A'].detach().cpu().numpy()[0]]
            for ii, atten_scores_ in enumerate(atten_scores):
                I1 = ImageDraw.Draw(image_shown_all[ii])
                for jj, score in enumerate(atten_scores_):
                    I1.text(patch_coords_all[ii][jj], "{:.4f}".format(
                        score), fill=(0, 255, 255), font=font)

                img_byte_arr = io.BytesIO()
                image_shown_all[ii].save(img_byte_arr, format='JPEG')
                image_urls_all[str(ii)] = "data:image/jpeg;base64, " + \
                    base64.b64encode(img_byte_arr.getvalue()).decode()
        else:
            return None, image_urls_all, results_dict, minWorH

    embedding = embedding.reshape(1, -1)
    embedding /= np.linalg.norm(embedding)
    return embedding, image_urls_all, results_dict, minWorH



def new_web_annotation2(cluster_label, min_dist, x, y, w, h, annoid_str):
    anno = {
        "type": "Annotation",
        "body": [{
            "type": "TextualBody",
            "value": "{}".format(min_dist),
            "purpose": "tagging"
        }],
        "target": {
            "source": "http://localhost:3000/",
            "selector": {
                "type": "FragmentSelector",
                "conformsTo": "http://www.w3.org/TR/media-frags/",
                "value": f"xywh=pixel:{x},{y},{w},{h}"
            }
        },
        "@context": "http://www.w3.org/ns/anno.jsonld",
        "id": annoid_str
    }
    return anno


def main():

    with open('/data/zhongz2/CPTAC/allsvs/allsvs.txt', 'r') as fp:
        filenames = [line.strip() for line in fp.readlines()]
    df = pd.DataFrame(filenames, columns=['orig_filename'])
    df['svs_prefix'] = [os.path.splitext(os.path.basename(f))[0] for f in df['orig_filename'].values]
    df['cancer_type'] = [f.split('/')[-2] for f in df['orig_filename'].values]
    clinical = df
    all_svs_prefixes = df['svs_prefix'].values
    all_labels_dict = dict(zip(df['svs_prefix'], df['cancer_type'])) # svs_prefix: cancer_type

    font2                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 5
    fontColor              = (45, 255, 255)
    thickness              = 5
    lineType               = 5
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font1 = ImageFont.truetype(font_path, size=28)

    topn = 5

    assert len(project_names) == 1

    # files = sorted(glob.glob('/mnt/hidare-efs/data_20240208/jiang_exp1/png/*.png'))
    # files = sorted(glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/png/*.png'))
    # # print('files', files)

    # files = glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4/png/*.png')
    # files = glob.glob('/data/zhongz2/HERE101_20x/*.tif')
    # files = glob.glob('/data/zhongz2/CPTAC/patches_256/CONCH/heatmap_files/*/patch1024/top0.png')
    files = glob.glob('/data/zhongz2/CPTAC/yottixel_bobs/patches/*/patch1024/top0.png')

    files = [f for f in files if '_r2.png' not in f and '_r4.png' not in f]

    search_backbone = 'HERE_CONCH'
    search_method = 'faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8'
    search_project = project_names[0]

    save_root = f'/data/zhongz2/CPTAC/search_from_CPTAC/{search_backbone}/{search_method}_mut/Yottixel_results/'
    save_root = f'/data/zhongz2/CPTAC/search_from_CPTAC/{search_backbone}/{search_method}_mut_intersection/Yottixel_results/'
    save_root = f'/data/zhongz2/CPTAC/search_from_CPTAC/{search_backbone}/{search_method}_mut_intersection/Yottixel_results_gpu/'
    os.makedirs(os.path.join(save_root, 'retrieved_patches'), exist_ok=True)


    cache_dir = '/data/zhongz2/CPTAC/caches'
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = '{}/{}.pkl'.format(cache_dir, search_project)

    if not os.path.exists(cache_filename):
        
        project_items = []
        all_scales = {}
        image_table = [] 
        all_coords = []
        all_svs_prefix_ids = []
        all_project_ids = []

        for proj_id, project_name in enumerate(project_names):
            project_items.append((proj_id, project_name))
            # h5filenames = sorted(glob.glob(os.path.join(f'/data/zhongz2/{project_name}_256/patches/*.h5')))
            h5filenames = sorted(glob.glob(os.path.join(f'/data/zhongz2/{project_name}_256/patches_intersection/*.h5')))

            for svs_prefix_id, h5filename in enumerate(h5filenames):
                
                svs_prefix = os.path.basename(h5filename).replace('.h5', '')

                with h5py.File(h5filename, 'r') as file:
                    coords = file['coords'][()].astype(np.int32)
                svs_prefix_ids = svs_prefix_id * np.ones((len(coords), 1), dtype=np.int32)
                project_ids = proj_id * np.ones((len(coords), 1), dtype=np.int32) 
                all_coords.append(coords)
                all_svs_prefix_ids.append(svs_prefix_ids)
                all_project_ids.append(project_ids)

                key = '{}_{}'.format(project_name, svs_prefix)
                if key in all_scales:
                    scale = all_scales[key]['scale'][0]
                    patch_size_vis_level = all_scales[key]['patch_size_vis_level']
                else:
                    scale = 1.0
                    patch_size_vis_level = 256
                # note = all_notes[svs_prefix] if svs_prefix in all_notes else svs_prefix
                note = clinical[clinical['svs_prefix']==svs_prefix].to_xml()
                # if 'TCGA' == svs_prefix[:4] and svs_prefix in case_uuids:
                #     note = f'Link: <a href=\"https://portal.gdc.cancer.gov/cases/{case_uuids[svs_prefix]}\" target=\"_blank\">{svs_prefix}</a>\n\n' + note
                external_link = ''
                if project_name == 'ST':
                    external_link = ST_df.loc[ST_df['ID']==svs_prefix, 'Source'].values[0]
                elif project_name == 'TCGA-COMBINED':
                    external_link = f'https://portal.gdc.cancer.gov/cases/{case_uuids[svs_prefix]}' if svs_prefix in case_uuids else ''
                image_table.append((proj_id, svs_prefix_id, svs_prefix, scale, patch_size_vis_level, external_link, note))
        
        all_coords = np.concatenate(all_coords)
        all_svs_prefix_ids = np.concatenate(all_svs_prefix_ids)
        all_project_ids = np.concatenate(all_project_ids)

        with open(cache_filename, 'wb') as fp:
            pickle.dump({
                'project_items':project_items,
                'all_scales':all_scales,
                'image_table':image_table ,
                'all_coords': all_coords,
                'all_svs_prefix_ids':all_svs_prefix_ids,
                'all_project_ids':all_project_ids,
            }, fp)
    else:
        with open(cache_filename, 'rb') as fp:
            tmpdata = pickle.load(fp)
            image_table = tmpdata['image_table']
            all_coords = tmpdata['all_coords']
            all_svs_prefix_ids = tmpdata['all_svs_prefix_ids']
            all_project_ids = tmpdata['all_project_ids']
            del tmpdata
        image_table = pd.DataFrame(image_table, columns=['proj_id', 'svs_prefix_id', 'svs_prefix', 'scale', 'patch_size_vis_level', 'external_link', 'note'])

    all_results = []
    all_results_per_slide = []

    k0 = 200
    for i, f in enumerate(files):

        print('begin ', f)
        # query_prefix = os.path.basename(f).replace('.png', '')
        # query_prefix = os.path.splitext(os.path.basename(f))[0]
        query_prefix = f.split('/')[-3]


        # save_dir = os.path.join(save_root, query_prefix)
        # os.makedirs(save_dir, exist_ok=True)

        # save_filename = os.path.join(save_dir, f'{query_prefix}.pkl')
        # if os.path.exists(save_filename):
        #     continue

        kk=k0
        while True:
            params = {
                'k': kk,
                'search_project': search_project,
                'search_feature': 'before_attention',
                'search_method': search_method,
                'socketid': '',
                'img_urls': [f],
                'filenames': [f],
                'resize': 0,
                'search_backbone': search_backbone
            }
                
            start = time.perf_counter()

            query_embedding, images_shown_urls, results_dict, minWorH = \
                get_query_embedding(params['img_urls'], resize=int(float(params['resize'])), search_backbone=params['search_backbone'])  # un-normalized
            
            query_embedding = query_embedding.reshape(1, -1)

            coxph_html_dict = {}

            query_embedding /= np.linalg.norm(query_embedding)  # l2norm normalized

            D, I = knn_search_images_by_faiss(query_embedding,
                                            k=params['k'], search_project=search_project,
                                            search_method=search_method,
                                            search_backbone=params['search_backbone'])
            
            random1000_mean, random1000_std, random1000_dists = compute_mean_std_cosine_similarity_from_random1000(
                query_embedding, search_project=search_project, search_backbone=params['search_backbone'])

            final_response = {}
            final_response1 = {}

            index_rank_scores = np.arange(1, 1+len(D))[::-1]

            for ii, (score, ind) in enumerate(zip(D, I)):

                # rowid, x, y, svs_prefix_id, proj_id, scale, patch_size_vis_level, slide_name, external_link, note = infos[ind]
                x, y = all_coords[ind]
                proj_id, svs_prefix_id, slide_name, scale, patch_size_vis_level, external_link, note = \
                    image_table[(image_table['proj_id']==all_project_ids[ind, 0]) & (image_table['svs_prefix_id']==all_svs_prefix_ids[ind, 0])].values[0]

                if slide_name == query_prefix:
                    continue

                scale = float(scale)
                patch_size_vis_level = int(patch_size_vis_level)
                if len(note) == 0:
                    note = 'No clinical information. '

                item = {'_score': score,
                        '_zscore': (score - random1000_mean) / random1000_std,
                        '_pvalue': len(np.where(random1000_dists <= score)[0]) / len(random1000_dists)}
                project_name = project_names[proj_id]
                x0, y0 = int(x), int(y)
                image_id = '{}_{}_x{:d}_y{:d}'.format(proj_id, svs_prefix_id, x, y)
                image_name = '{}_x{}_y{}'.format(slide_name, x, y)

                if clinical is not None:
                    cancer_type = clinical[clinical['svs_prefix']==slide_name]['cancer_type'].values[0]
                else:
                    cancer_type = 'None'

                if 'ST_' in project_name:
                    has_gene = '1'
                else:
                    has_gene = '0'

                # image_id_bytes = image_id.encode('ascii')
                # img_bytes = None
                # for i in range(len(txns[project_name])):
                #     if img_bytes is None:
                #         img_bytes = txns[project_name][i].get(image_id_bytes)

                # if img_bytes is None:
                #     print('no img_bytes')
                #     continue

                # im = Image.open(io.BytesIO(img_bytes))
                # buffer = io.BytesIO()
                # im.save(buffer, format="jpeg")
                # encoded_image = base64.b64encode(buffer.getvalue()).decode()
                # img_url = "data:image/jpeg;base64, " + encoded_image
                img_url = ''

                x = int(float(scale) * float(image_name.split('_')[-2].replace('x', '')))
                y = int(float(scale) * float(image_name.split('_')[-1].replace('y', '')))

                if slide_name in final_response:
                    final_response[slide_name]['images'].append(
                        {'img_url': img_url, 'x': x, 'y': y, 'x0': x0, 'y0': y0, 'image_name': image_name,
                        'has_gene': has_gene})
                    final_response[slide_name]['annotations'].append(
                        new_web_annotation2(0, "{:.3f}, z{:.3f}, p{:.3f}".format(item['_score'], item['_zscore'],
                                                                                item['_pvalue']),
                                            x, y, patch_size_vis_level, patch_size_vis_level, ""))
                    final_response[slide_name]['scores'].append(float(item['_score']))
                    final_response[slide_name]['zscores'].append(
                        float(item['_zscore']))
                    final_response[slide_name]['note'] = note
                    final_response[slide_name]['external_link'] = external_link
                    final_response1[slide_name]['index_rank_scores'].append(
                        index_rank_scores[ii]
                    )
                else:
                    final_response[slide_name] = {}
                    final_response[slide_name]['project_name'] = project_name if 'KenData' not in project_name else "NCIData"
                    final_response[slide_name]['cancer_type'] = cancer_type
                    final_response[slide_name]['images'] = []
                    final_response[slide_name]['images'].append(
                        {'img_url': img_url, 'x': x, 'y': y, 'x0': x0, 'y0': y0, 'image_name': image_name,
                        'has_gene': has_gene})
                    final_response[slide_name]['annotations'] = []
                    final_response[slide_name]['annotations'].append(
                        new_web_annotation2(0, "{:.3f}, z{:.3f}, p{:.3f}".format(item['_score'], item['_zscore'],
                                                                                item['_pvalue']),
                                            x, y, patch_size_vis_level, patch_size_vis_level, ""))
                    final_response[slide_name]['scores'] = []
                    final_response[slide_name]['scores'].append(float(item['_score']))
                    final_response[slide_name]['zscores'] = []
                    final_response[slide_name]['zscores'].append(
                        float(item['_zscore']))
                    final_response[slide_name]['note'] = note
                    final_response[slide_name]['external_link'] = external_link
                    final_response1[slide_name] = {}
                    final_response1[slide_name]['index_rank_scores'] = [index_rank_scores[ii]]

            end = time.perf_counter()
            time_elapsed_ms = (end - start) * 1000

            zscore_sum_list = []
            for k in final_response.keys():
                final_response[k]['min_score'] = float(min(final_response[k]['scores']))
                final_response[k]['max_score'] = float(max(final_response[k]['scores']))
                final_response[k]['min_zscore'] = float(min(final_response[k]['zscores']))
                final_response[k]['max_zscore'] = float(max(final_response[k]['zscores']))
                zscore_sum = float(sum(final_response[k]['zscores']))
                # final_response[k]['zscore_sum'] = zscore_sum
                final_response[k]['zscore_sum'] = float(sum(final_response1[k]['index_rank_scores']))
                # zscore_sum_list.append(abs(zscore_sum))
                # zscore_sum_list.append(len(final_response[k]['zscores']))
                zscore_sum_list.append(sum(final_response1[k]['index_rank_scores']))
                final_response[k]['_random1000_mean'] = float(random1000_mean)
                final_response[k]['_random1000_std'] = float(random1000_std)
                final_response[k]['_time_elapsed_ms'] = float(time_elapsed_ms)

            kk += 100
            if len(final_response) >= topn:
                break
            

        sort_inds = np.argsort(zscore_sum_list)[::-1].tolist()
        allkeys = list(final_response.keys())
        ranks = {rank: allkeys[ind] for rank, ind in enumerate(sort_inds)} # sorted by zscore_sum descend order

        # prediction
        if params['search_backbone'] in ['HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_CONCH', 'HERE_UNI']:
            table_str = [
                '<table border="1"><tr><th>task</th><th>prediction</th></tr>']
            for k,v in models_dict[params['search_backbone']]['attention_model'].classification_dict.items():
                Y_prob_k = F.softmax(results_dict[k + '_logits'], dim=1).detach().cpu().numpy()[0]
                table_str.append(
                    '<tr><td>{}</td><td>{}: {:.3f}</td></tr>'.format(k.replace('_cls', ''), v[1], Y_prob_k[1]))
            for k in models_dict[params['search_backbone']]['attention_model'].regression_list:
                table_str.append(
                    '<tr><td>{}</td><td>{:.3f}</td></tr>'.format(k, results_dict[k + '_logits'].item()))
            table_str.append('</table>')
            pred_str = ''.join(table_str)
        else:
            pred_str = 'None'

        gc.collect()

        # all_results[query_prefix] = {'coxph_html_dict': coxph_html_dict, 'response': final_response, 'ranks': ranks, 'pred_str': pred_str,
        #     'images_shown_urls': images_shown_urls, 'minWorH': minWorH}

        results = []
        patches_dir = '/data/zhongz2/CPTAC/patches_256/patches'
        svs_dir = '/data/zhongz2/CPTAC/svs'

        # top_n_labels = [all_labels_dict[ind] for ind in sort_inds[:topn]]
        # top_n_dists = dists_grouped[sort_inds[:topn]]
        # majority_vote_pred = Counter(top_n_labels).most_common(1)[0][0]

        # gt = all_labels_dict[all_svs_prefixes_reverse[query_prefix]] if query_prefix in all_svs_prefixes_reverse else None
        # if gt is None:
        #     continue

        # all_results.append((query_prefix, gt, majority_vote_pred, top_n_labels, top_n_dists))

        # output retrieved patches
        top_n_labels = []
        top_n_dists = []
        retrived_images = [np.array(Image.open(f).convert('RGB').resize((1000, 1000)))]

        for ri, (svs_prefix, item) in enumerate(final_response.items()):

            if ri >= topn:
                break

            # save_dir = os.path.join(save_root, query_prefix, f'top-{ri}-{svs_prefix}')
            # os.makedirs(save_dir, exist_ok=True)

            slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))
            objective_power = int(slide.properties['openslide.objective-power'])
            patch_size_20x = int((objective_power/20.)*1000)
            patch_level = 0

            coords = np.array([[v['x0'], v['y0']] for v in item['images']])
            dists_in_slide = np.array(item['scores'])
            cancer_type = item['cancer_type']
            top_n_labels.append(cancer_type)
            top_n_dists.append(dists_in_slide.min())
            all_results_per_slide.append((query_prefix, svs_prefix, ri, dists_in_slide.min(), dists_in_slide, coords))

            for ii in range(min(10, len(coords))):
                x, y = coords[ii]
                im = np.array(slide.read_region(location=(x, y), level=patch_level, size=(patch_size_20x, patch_size_20x)).convert('RGB'))

                cv2.putText(im,'{}: {:.3f}'.format(ri, dists_in_slide[ii]), 
                    (8, 512), 
                    font2, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

                # cv2.imwrite(os.path.join(save_dir, 'top-{:05d}.jpg'.format(ii)), im[:,:,::-1])  # RGB --> BGR
                retrived_images.append(im)

                break
        
        retrived_images = np.concatenate(retrived_images, axis=1)
        cv2.imwrite(os.path.join(save_root, 'retrieved_patches', query_prefix+'.jpg'), retrived_images[:,:,::-1])

        gt = all_labels_dict[query_prefix] if query_prefix in all_labels_dict else None
        if gt is None:
            continue

        majority_vote_pred = Counter(top_n_labels).most_common(1)[0][0]
        all_results.append((query_prefix, gt, majority_vote_pred, top_n_labels, top_n_dists))

        if i % 100 == 0:
            print(i)


    all_results1 = pd.DataFrame(all_results, columns=['svs_prefix', 'labelStr', 'predStr', 'mvPred', 'mvDist'])
    labels_dict = {v: i for i,v in enumerate(df['cancer_type'].unique())}

    all_results1['gt'] = all_results1['labelStr'].map(labels_dict)
    all_results1['pred'] = all_results1['predStr'].map(labels_dict)

    y_true = all_results1['gt'].values
    y_pred = all_results1['pred'].values
    labels = np.unique(list(labels_dict.values()))

    c_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(y_true, y_pred, output_dict=False)

    with open(os.path.join(save_root, 'all_results.pkl'), 'wb') as fp:
        pickle.dump({
            'all_results': all_results,
            'all_results_per_slide': all_results_per_slide,
            'report_text': report_text,
            'c_matrix': c_matrix,
            'y_true': y_true,
            'y_pred': y_pred,
            'labels_dict': labels_dict,
            'all_svs_prefixes': all_svs_prefixes
        }, fp)

    with open(os.path.join(save_root, 'classification_report.txt'), 'w') as fp:
        fp.writelines(report_text)




def get_all_data():

    import sys,os,glob,shutil
    import numpy as np
    
    results_dirs = {
        'Yottixel': '/data/zhongz2/CPTAC/yottixel_bobs/CPTAC/HERE_CONCH_results/Yottixel',
        'RetCCL': '/data/zhongz2/PSC/FEATURES/DATABASE/NCI/CPTAC/HERE_CONCH_results/RetCCL',
        'SISH_patch': '/data/zhongz2/PSC_SISH/FEATURES/DATABASE/MOSAICS/NCI/CPTAC/20x/HERE_CONCH_results/SISH_patch',
        'SISH_slide': '/data/zhongz2/PSC_SISH/FEATURES/DATABASE/MOSAICS/NCI/CPTAC/20x/HERE_CONCH_results/SISH_slide',
        'HERE_CONCH': '/data/zhongz2/CPTAC/search_from_CPTAC/HERE_CONCH/faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8'
    }

    for method, result_dir in results_dirs.items():
        f = os.path.join(result_dir, 'classification_report.txt')
        if not os.path.exists(f):
            continue
        with open(f, 'r') as fp:
            lines = fp.read()

        print(method)
        print(lines)


if __name__ == '__main__':
    main()








