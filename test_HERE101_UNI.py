


import sys,os,json,glob
import pandas as pd
import numpy as np
import tarfile
import io
import gc
import re
import faiss
from sklearn.metrics import pairwise_distances
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


# loading all packages here to start
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics
from uni.downstream.utils import concat_images


def main():

    files = sorted(glob.glob('/mnt/hidare-efs/data_20240208/jiang_exp1/png/*.png'))
    files = sorted(glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/png/*.png'))
    print('files', files)

    jinlin_df = pd.read_excel('/data/zhongz2/test_CONCH_ProvGigaPath_UNI/refined Ver0831.xlsx')

    save_dir = '/data/zhongz2/test_CONCH_ProvGigaPath_UNI/UNI'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # checkpoint_path = 'CONCH_weights_pytorch_model.bin'
    # model, preprocess = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=checkpoint_path, device=device)
    # _ = model.eval()
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load("./UNI_pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


if __name__ == '__main__':
    main()












