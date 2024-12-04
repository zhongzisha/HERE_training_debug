


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

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from conch.downstream.zeroshot_path import zero_shot_classifier, run_mizero, topj_pooling
from conch.downstream.wsi_datasets import WSIEmbeddingDataset


def main():

    files = sorted(glob.glob('/mnt/hidare-efs/data_20240208/jiang_exp1/png/*.png'))
    files = sorted(glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/png/*.png'))
    print('files', files)

    jinlin_df = pd.read_excel('/data/zhongz2/test_CONCH_ProvGigaPath_UNI/refined Ver0831.xlsx')
    classnames = [v.replace('\xa0','') for v in jinlin_df['key_point'].values if not isinstance(v, float)]
    classnames = np.unique(classnames).tolist()
    n_classes = len(classnames)
    classnames_text = [[v] for v in classnames]

    save_dir = '/data/zhongz2/test_CONCH_ProvGigaPath_UNI/CONCH'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'CONCH_weights_pytorch_model.bin'
    model, preprocess = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=checkpoint_path, device=device)
    _ = model.eval()

    # index_col = 'slide_id' # column with the slide ids
    # # target_col = 'OncoTreeCode' # column with the target labels
    # # label_map = {'LUAD': 0, 'LUSC': 1} # maps values in target_col to integers
    # target_col = None
    # label_map = None

    # # assuming the csv has a column for slide_id (index_col) and OncoTreeCode (target_col), adjust above as needed
    # df = pd.read_csv('path/to/csv')
    # # path to the extracted embeddings, assumes the embeddings are saved as .pt files, 1 file per slide
    # data_source = '/path/to/extracted-embeddings/' 

    # df = df[df[target_col].isin(label_map.keys())].reset_index(drop=True)

    # dataset = WSIEmbeddingDataset(data_source = data_source,
    #                             df=df,
    #                             index_col=index_col,
    #                             target_col=target_col,
    #                             label_map=label_map)
    # dataloader = DataLoader(dataset, 
    #                         batch_size=1, 
    #                         shuffle=False, 
    #                         num_workers=4)

    # idx_to_class = {v:k for k,v in dataloader.dataset.label_map.items()}
    # print("num samples: ", len(dataloader.dataset))
    # print(idx_to_class)

    prompt_file = './conch/prompts/nsclc_prompts_all_per_class.json'
    with open(prompt_file) as f:
        prompts = json.load(f)['0']
    # classnames = prompts['classnames']
    templates = prompts['templates']
    # n_classes = len(classnames)
    # classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
    # for class_idx, classname in enumerate(classnames_text):
    #     print(f'{class_idx}: {classname}')

    zeroshot_weights = zero_shot_classifier(model, classnames_text, templates, device=device)
    print(zeroshot_weights.shape)

    topj = (1,5,10,50,100)
    results = []
    for _, row in jinlin_df.iterrows():
        query_prefix = row['query']
        if not isinstance(row['key_point'], float):
            label = row['key_point'].replace('\xa0','')
        else:
            label = ''
        f = f'/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/png/{query_prefix}.png'
        image = Image.open(f)
        image = preprocess(image).unsqueeze(0)
        print(image.shape)

        with torch.inference_mode():
            image = image.to(device)
            image_embs = model.encode_image(image)

            image_features = model.visual.forward_project(image_embs)            
            image_features = F.normalize(image_features, dim=-1) 
            logits = image_features @ zeroshot_weights
            preds, pooled_logits = topj_pooling(logits, topj = topj)

            pred = preds[1].detach().cpu().numpy()[0]

            pred = classnames_text[pred][0]
        results.append((query_prefix, label, pred))

    results_df = pd.DataFrame(results, columns=['query_prefix', 'key_point', 'prediction'])
    results_df.to_csv(os.path.join(save_dir, 'results.csv'))


    # results, dump = run_mizero(model, zeroshot_weights, dataloader, device, \
    #     dump_results=True, metrics=['bacc', 'weighted_f1'])

    # best_j_idx = np.argmax(list(results['bacc'].values()))
    # best_j = list(results['bacc'].keys())[best_j_idx]
    # for metric, metric_dict in results.items():
    #     print(f"{metric}: {metric_dict[best_j]:.3f}")


if __name__ == '__main__':
    main()












