import sys, os, glob, shutil
import pdb
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from skimage import exposure
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import time
import h5py
import openslide
from sklearn.preprocessing import MinMaxScaler
import torch
from multiprocessing import Pool
import itertools
from natsort import natsorted
import idr_torch
import argparse
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoImageProcessor, CLIPProcessor
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import ResNetModel, BeitModel, BitModel, ConvNextModel, CvtModel, DeiTModel, \
    DinatModel, DPTModel, EfficientFormerModel, GLPNModel, MobileNetV1Model, ImageGPTModel, \
    LevitModel, MobileNetV1Model, MobileNetV2Model, MobileViTModel, NatModel, PoolFormerModel, \
    SwinModel, Swinv2Model, ViTModel, ViTHybridModel, ViTMAEModel, ViTMSNModel, CLIPModel
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import gc
import clip
import socket
from datetime import datetime
from common import HF_MODELS_DICT
from dataset import PatchDatasetV2
from utils import save_hdf5



def main():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, default='')  
    parser.add_argument('--data_slide_dir', type=str, default='')  
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default='') 
    parser.add_argument('--feat_dir', type=str, default='') 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='resnet18')
    args = parser.parse_args()

    model_name = args.model_name
    model_params = HF_MODELS_DICT[model_name] if model_name in HF_MODELS_DICT else None
    print('HF_MODELS_DICT:', HF_MODELS_DICT)

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None or csv_path == "None":
        DX_filenames = sorted(glob.glob(os.path.join(args.data_slide_dir, '*{}*'.format(args.slide_ext))))
        df = pd.DataFrame({'DX_filename': DX_filenames})
    else:
        if 'xlsx' in csv_path:
            df = pd.read_excel(csv_path, low_memory=False)
        else:
            df = pd.read_csv(csv_path, low_memory=False)

    if 'slide_id' not in df:
        slide_ids = [os.path.basename(f).replace(args.slide_ext, '') for f in df['DX_filename'].values]
        df['slide_id'] = slide_ids
    else:
        df['slide_id'] = df['slide_id'].astype(str)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    print('local_rank: {},{} '.format(socket.gethostname(), idr_torch.local_rank))
    torch.cuda.set_device(idr_torch.local_rank)
    device = torch.device('cuda')
    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], model_name, str(idr_torch.rank), str(idr_torch.local_rank))
    elif os.environ['CLUSTER_NAME'] == 'FRCE':
        local_temp_dir = os.path.join('/tmp/', os.environ['USER'], model_name, str(idr_torch.rank), str(idr_torch.local_rank))
    else:
        local_temp_dir = os.path.join(os.environ['HOME'], model_name, str(idr_torch.rank), str(idr_torch.local_rank))
    os.makedirs(local_temp_dir, exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    existed_prefixes = set([os.path.basename(f).replace('.pt', '') for f in dest_files])
    h5_prefixes = set([os.path.basename(f).replace('.h5', '') for f in glob.glob(os.path.join(args.data_h5_dir, 'patches',  '*.h5'))])

    drop_ids = []
    for ind, f in enumerate(df['DX_filename'].values):
        svs_prefix = os.path.basename(f).replace(args.slide_ext, '')
        if svs_prefix in existed_prefixes or svs_prefix not in h5_prefixes:
            drop_ids.append(ind)
    if len(drop_ids) > 0:
        df = df.drop(drop_ids)

    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    print('index_splits', index_splits)
    print('rank: ', idr_torch.rank)
    print('local rank:', idr_torch.local_rank)
    print('world_size: ', idr_torch.world_size)
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)
    print(idr_torch.rank, idr_torch.local_rank, idr_torch.world_size, sub_df)

    feature_tensors = {}
    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()
        return hook

    time.sleep(3)
    for index, slide_file_path in enumerate(sub_df['DX_filename'].values):
        svs_prefix = os.path.basename(slide_file_path).replace(args.slide_ext, '')
        slide_file_path = os.path.join(args.data_slide_dir, svs_prefix+ args.slide_ext)

        h5_file_path = os.path.join(args.data_h5_dir, 'patches', svs_prefix + '.h5')
        if not os.path.exists(h5_file_path):
            continue

        local_temp_dir1 = os.path.join(local_temp_dir, datetime.now().strftime(format='%Y%m%d%H%M%S'))
        os.makedirs(local_temp_dir1, exist_ok=True)

        local_svs_filename = os.path.join(local_temp_dir1, svs_prefix + args.slide_ext)
        os.system(f'cp -RL "{slide_file_path}" "{local_svs_filename}"')
        time.sleep(1)

        h5file = h5py.File(h5_file_path, 'r')
        dset = h5file['coords']
        coords = dset[:]
        patch_level = h5file['coords'].attrs['patch_level']
        patch_size = h5file['coords'].attrs['patch_size']

        # filter the coords
        slide = openslide.OpenSlide(local_svs_filename)
        time.sleep(1)
        print('extract features')

        config = None
        transform = None
        image_processor = None 
        if model_name == 'mobilenetv3':
            model = timm.create_model('mobilenetv3_large_100', pretrained=True)
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
            model.flatten.register_forward_hook(get_activation('after_flatten'))
        elif model_name == 'ProvGigaPath':
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif model_name == 'CONCH':
            from conch.open_clip_custom import create_model_from_pretrained
            model, image_processor = create_model_from_pretrained('conch_ViT-B-16','/data/zhongz2/HUGGINGFACE_HUB_CACHE/CONCH_weights/pytorch_model.bin')
        else:
            print('model_params: ', model_params)
            model = globals()[model_params[0]].from_pretrained(model_params[1])
            if 'PLIP' in model_name or 'CLIP' in model_name:
                image_processor = CLIPProcessor.from_pretrained(model_params[1])
            else:
                image_processor = AutoImageProcessor.from_pretrained(model_params[1])
        model = model.to(device)
        model.eval() 

        def collate_fn2(examples): 
            pixel_values = torch.stack([transform(example["pixel_values"]) for example in examples])
            coords = np.vstack([example["coords"] for example in examples])
            return pixel_values, coords

        def collate_fn(examples):
            pixel_values = image_processor(images=[example["pixel_values"] for example in examples], return_tensors='pt')
            coords = np.vstack([example["coords"] for example in examples])
            return pixel_values['pixel_values'], coords

        def collate_fn_CONCH(examples):
            pixel_values = torch.stack([image_processor(example["pixel_values"]) for example in examples])
            coords = np.vstack([example["coords"] for example in examples])
            return pixel_values, coords

        dataset = PatchDatasetV2(slide, coords, patch_level, patch_size)
        kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': False}
        if transform is not None:
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=collate_fn2)
        else:
            if model_name == 'CONCH':
                loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=collate_fn_CONCH)
            else:
                loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=collate_fn)

        output_path = os.path.join(local_temp_dir1, svs_prefix + '.h5')
        time_start = time.time()
        mode = 'w'
        for count, (images, coords) in enumerate(loader):
            with torch.no_grad():
                images = images.to(device)

                if model_name == 'mobilenetv3':
                    output = model(images)
                    features = feature_tensors.get('after_flatten_feat').cpu().numpy().reshape((len(coords), -1))
                elif model_name == 'ProvGigaPath':
                    features = model(images).detach().cpu().numpy().reshape((len(coords), -1))
                elif model_name == 'CONCH':
                    features = model.encode_image(images, proj_contrast=False, normalize=False).detach().cpu().numpy().reshape((len(coords), -1))
                elif model_params is not None and model_params[2] is not None:
                    features = model(images, return_dict=True)
                    features = features.get(model_params[2]).cpu().numpy().reshape((len(coords), -1))
                elif 'PLIP_Retrained' in model_name:
                    features = model.encode_image(images).detach().cpu().numpy().reshape((len(coords), -1))
                else:
                    features = model.get_image_features(images).detach().cpu().numpy().reshape((len(coords), -1)) # CLIP, PLIP

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_path, time_elapsed))
        del model
        gc.collect() 

        with h5py.File(output_path, "r") as file:
            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
            features = torch.from_numpy(features)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', svs_prefix + '.pt'))

        time.sleep(0.5)
        os.system(f'rm -rf "{output_path}"')

        slide.close()
        os.system(f'rm -rf "{local_svs_filename}"')
        time.sleep(1)

        torch.cuda.empty_cache()
        
        if os.path.isdir(local_temp_dir1):
            os.system(f'rm -rf "{local_temp_dir1}"')


    time.sleep(2)
    if os.path.isdir(local_temp_dir):
        os.system(f'rm -rf "{local_temp_dir}"')



if __name__ == '__main__':
    main()









