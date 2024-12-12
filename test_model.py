

import sys, os, glob, shutil, json
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
from timm.layers import set_layer_config
from timm.models import is_model, model_entrypoint, load_checkpoint
from urllib.parse import urlsplit
import gc
import clip
import socket
from common import HF_MODELS_DICT
from dataset import PatchDatasetV2
from utils import save_hdf5, visHeatmap, score2percentile, to_percentiles
from model import AttentionModel
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True



def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def load_model_config_from_hf(model_id: str):
    cached_file = '/data/zhongz2/HUGGINGFACE_HUB_CACHE/ProvGigaPath/config.json'

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


def split_model_name_tag(model_name: str, no_tag: str = ''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


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
    load_checkpoint(model, '/data/zhongz2/HUGGINGFACE_HUB_CACHE/ProvGigaPath/pytorch_model.bin')

    return model



def get_args():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, default='/data/zhongz2/tcga/TCGA-ALL2_256/patches')  
    parser.add_argument('--data_slide_dir', type=str, default='/data/zhongz2/tcga/TCGA-ALL2_256/svs')  
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default='./splits/test-0.csv') 
    parser.add_argument('--feat_dir', type=str, default=os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'feats')) 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='UNI')
    parser.add_argument('--start_idx', type=int, default=-1)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--generate_attention_heatmap', type=int, default=0)
    parser.add_argument("--hidare_checkpoint", type=str, default="/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backboneUNI_dropout0.25/split_3/snapshot_58.pt", help="Path to HiDARE setp2 checkpoint")
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    print('args', args)

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
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path, low_memory=False)

    if 'slide_id' not in df:
        slide_ids = [os.path.basename(f).replace(args.slide_ext, '') for f in df['DX_filename'].values]
        df['slide_id'] = slide_ids
    else:
        df['slide_id'] = df['slide_id'].astype(str)

    print('before', len(df))
    if args.end_idx > args.start_idx > -1:
        if  args.end_idx >= len(df):
            args.end_idx = len(df)
        df = df.iloc[np.arange(args.start_idx, args.end_idx)].reset_index(drop=True)
    print('after', len(df))

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pred_files'), exist_ok=True)
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
    existed_prefixes = set([os.path.basename(f).replace('.pt', '') for f in os.listdir(os.path.join(args.feat_dir, 'pred_files'))])
    h5_prefixes = set([os.path.basename(f).replace('.h5', '') for f in glob.glob(os.path.join(args.data_h5_dir,  '*.h5'))])

    drop_ids = []
    for ind, f in enumerate(df['DX_filename'].values):
        svs_prefix = os.path.basename(f).replace(args.slide_ext, '')
        if svs_prefix in existed_prefixes or svs_prefix not in h5_prefixes:
            drop_ids.append(ind)
    print('before0', len(df))
    if len(drop_ids) > 0:
        df = df.drop(drop_ids)
    df = df.reset_index(drop=True)
    if len(df) == 0:        
        print('all done')
        sys.exit(0)
    print('before', len(df))

    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    print('local rank:', idr_torch.local_rank)
    print('world_size: ', idr_torch.world_size)
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)
    if len(sub_df) == 0:
        print('all done')
        sys.exit(0)
    print(idr_torch.rank, sub_df['DX_filename'].values[0])

    feature_tensors = {}
    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()
        return hook


    config = None
    transform = None
    image_processor = None 
    if model_name == 'mobilenetv3':
        model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        model.flatten.register_forward_hook(get_activation('after_flatten'))
    elif model_name == 'ProvGigaPath':
        # model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        model = create_model()  # timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
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
        model, image_processor = create_model_from_pretrained('conch_ViT-B-16','./CONCH_weights_pytorch_model.bin')
    elif model_name == 'UNI':
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load("./UNI_pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    else:
        print('model_params: ', model_params)
        model = globals()[model_params[0]].from_pretrained(model_params[1])
        if 'PLIP' in model_name or 'CLIP' in model_name:
            image_processor = CLIPProcessor.from_pretrained(model_params[1])
        else:
            image_processor = AutoImageProcessor.from_pretrained(model_params[1])
    model = model.to(device)
    model.eval() 

    if 'CONCH' in args.model_name:
        attention_model = AttentionModel(backbone='CONCH')
    elif 'UNI' in args.model_name:
        attention_model = AttentionModel(backbone='UNI')
    elif 'ProvGigaPath' in args.model_name:
        attention_model = AttentionModel(backbone='ProvGigaPath')
    elif 'PLIP' in args.model_name:
        attention_model = AttentionModel(backbone='PLIP')
    else:
        raise ValueError("error model name")

    state_dict = torch.load(args.hidare_checkpoint, map_location='cpu', weights_only=True)
    attention_model.load_state_dict(state_dict['MODEL_STATE'], strict=False)
    attention_model = attention_model.to(device)
    attention_model.eval()

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
        
    time.sleep(3)
    for index, row in sub_df.iterrows():

        svs_prefix = os.path.basename(row['DX_filename']).replace(args.slide_ext, '')

        h5_file_path = os.path.join(args.data_h5_dir, svs_prefix + '.h5')

        if not os.path.exists(h5_file_path):
            continue

        final_feat_filename = os.path.join(args.feat_dir, 'pt_files', svs_prefix + '.pt')
        # if os.path.exists(final_feat_filename):
        #     continue

        local_temp_dir1 = os.path.join(local_temp_dir, str(np.random.randint(1e5) + np.random.randint(1e10)))
        os.makedirs(local_temp_dir1, exist_ok=True)
        
        h5file = h5py.File(h5_file_path, 'r')
        dset = h5file['coords']
        all_coords = dset[:]
        patch_level = h5file['coords'].attrs['patch_level']
        patch_size = h5file['coords'].attrs['patch_size']

        svs_filename1 = os.path.realpath(os.path.join(args.data_slide_dir, svs_prefix+args.slide_ext))
        local_svs_filename = os.path.join(local_temp_dir1, os.path.basename(svs_filename1))
        os.system(f'cp -RL "{svs_filename1}" "{local_svs_filename}"')
        time.sleep(0.5)
        
        backbone_feature_filename = os.path.join(args.feat_dir, 'pt_files', svs_prefix + '.pt')
        if not os.path.exists(backbone_feature_filename):

            # filter the coords
            slide = openslide.open_slide(local_svs_filename)
            time.sleep(1)
            print('extract features')

            dataset = PatchDatasetV2(slide, all_coords, patch_level, patch_size)
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
                    elif model_name == 'UNI':
                        features = model(images).detach().cpu().numpy().reshape((len(coords), -1))
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

            with h5py.File(output_path, "r") as file:
                features = file['features'][:]
                print('features size: ', features.shape)
                print('coordinates size: ', file['coords'].shape)
                features = torch.from_numpy(features)
                torch.save(features, os.path.join(args.feat_dir, 'pt_files', svs_prefix + '.pt'))

            time.sleep(0.5)
            os.system(f'rm -rf "{output_path}"')

            # del model
            gc.collect() 
            slide.close()
        else:
            features = torch.load(backbone_feature_filename, weights_only=True)

        features = features.to(device)
        with torch.no_grad():
            results_dicts = attention_model(features, return_features=True)
        results_dicts = {k: v.detach().cpu().numpy() for k,v in results_dicts.items()}
        torch.save(results_dicts, os.path.join(args.feat_dir, 'pred_files', svs_prefix + '.pt'))
        torch.cuda.empty_cache()

        if args.generate_attention_heatmap:
            A = np.copy(results_dicts['A_raw'])[0]

            save_filename = '{}/pred_files/{}_big_attention_map.tif'.format(args.feat_dir, svs_prefix)
            if not os.path.exists(save_filename):
                slide = openslide.open_slide(local_svs_filename)
                img = visHeatmap(slide, scores=A, coords=all_coords,
                                vis_level=0, patch_size=(patch_size, patch_size),
                                convert_to_percentiles=True)
                print(type(img), img.size)
                img.save(save_filename)
                img_vips = pyvips.Image.new_from_array(img)
                # img_vips.dzsave(save_filename, tile_size=1024)
                img_vips.tiffsave(save_filename, compression="jpeg",
                    tile=True, tile_width=256, tile_height=256,
                    pyramid=True,  bigtiff=True)
                # img_vips.write_to_file(save_filename, tile=True, compression="jpeg", bigtiff=True, pyramid=True)
                # time.sleep(1)
                # del img, img_vips

        
        os.system(f'rm -rf "{local_svs_filename}"')
        time.sleep(1)

        if os.path.isdir(local_temp_dir1):
            os.system(f'rm -rf "{local_temp_dir1}"')

    time.sleep(2)
    if os.path.isdir(local_temp_dir):
        os.system(f'rm -rf "{local_temp_dir}"')


def softmax_stable(x):  # only 2-D
    x = np.exp(x - np.max(x, axis=1)[:, None])
    return x / x.sum(axis=1)[:, None]


def check_case_number():

    with open('/data/zhongz2/CPTAC/biospecimen.cohort.2024-12-01.json','r') as fp:
        bio = json.load(fp)
    m = {}
    for i in range(len(bio)):
        if len(bio[i]['samples'][0]['portions']) >0 and len(bio[i]['samples'][0]['portions'][0]['analytes']) > 0:
            m[bio[i]['submitter_id']]= bio[i]['samples'][0]['portions'][0]['analytes'][0]['aliquots'][0]['submitter_id']


def generate_CPTAC_dirs():
    import sys,os,glob,shutil
    import pandas as pd

    allsvs_dir = '/data/zhongz2/CPTAC_256/svs'
    allpatches_dir = '/data/zhongz2/CPTAC_256/patches'
    patch_prefixes = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(allpatches_dir, '*.h5'))]

    with open('/data/zhongz2/CPTAC/allsvs/allsvs.txt', 'r') as fp:
        lines = [line.strip().split('/')[-2:] for line in fp.readlines()]
    df = pd.DataFrame(lines, columns=['cancer_type', 'file_name'])

    for cancer_type in df['cancer_type'].unique():
        print(cancer_type)
        save_dir = f'/data/zhongz2/CPTAC_{cancer_type}_256'
        os.makedirs(os.path.join(save_dir, 'svs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'patches'), exist_ok=True)
        os.system('ln -sf "/data/zhongz2/CPTAC_256/feats" "{}/"'.format(save_dir))

        dff = df[df['cancer_type'] == cancer_type]
        for file_name in dff['file_name'].values:
            svs_prefix = os.path.splitext(file_name)[0]
            if svs_prefix not in patch_prefixes:
                continue
            os.system('ln -sf "/data/zhongz2/CPTAC_256/svs/{}.svs" "{}"'.format(svs_prefix, os.path.join(save_dir, 'svs', svs_prefix+'.svs')))
            os.system('ln -sf "/data/zhongz2/CPTAC_256/patches/{}.h5" "{}"'.format(svs_prefix, os.path.join(save_dir, 'patches', svs_prefix+'.h5')))


def prepare_labels_forCPTAC():

    import os, glob,json
    import pandas as pd
    import numpy as np
    from common import CLASSIFICATION_DICT, REGRESSION_LIST, IGNORE_INDEX_DICT
    import torch
    from sklearn.metrics import confusion_matrix, f1_score, auc, roc_auc_score, roc_curve, classification_report, r2_score
    from scipy.stats import percentileofscore, pearsonr, spearmanr
    from matplotlib import pyplot as plt


    # clinical
    txt_files = glob.glob('/data/zhongz2/CPTAC/Clinical_meta_data_v1/*.txt')
    alldfs = []
    for f in txt_files:
        df = pd.read_csv(f, sep='\t')
        df.drop(0, inplace=True)
        alldfs.append(df)
    clinical = pd.concat(alldfs)

    # mutation effect
    oncokb_files = glob.glob('/data/zhongz2/CPTAC/Mutation_BCM_v1/*oncokb.txt')
    alldata = {}
    gene_dict = {}
    for filepath in oncokb_files:

        df = pd.read_csv(filepath, sep='\t', low_memory=False)
        df1 = df.drop_duplicates(subset=['gene_name'])
        gene_dict.update(dict(zip(df1['gene_name'], df1['Hugo_Symbol'])))
        barcodes = df['Tumor_Sample_Barcode'].unique()
        for barcode in barcodes:
            print(barcode)
            dff = df[df['Tumor_Sample_Barcode'] == barcode]
            dff = dff[['Hugo_Symbol', 'MUTATION_EFFECT']].drop_duplicates(subset=['Hugo_Symbol'])
            dff = dff.set_index('Hugo_Symbol')
            if barcode in alldata:
                print(barcode, ' is in there')
                import pdb
                pdb.set_trace()
            alldata[barcode] = dff.to_dict()['MUTATION_EFFECT']
            # for ii, gene_symbol in enumerate(dff.index.values):
            #     if gene_symbol not in alldata:
            #         alldata[gene_symbol] = {}
            #     alldata[gene_symbol][barcode] = dff.loc[gene_symbol, 'MUTATION_EFFECT']
    mutation_df = pd.DataFrame.from_dict(alldata, orient='index').transpose()
    mutation_df.index.name = 'Hugo_Symbol'
    mutation_df.to_csv('/data/zhongz2/CPTAC/Mutation_BCM_v1/mutation_effect.csv')

    gene_symbols = {
        'TP53': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'PIK3CA': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'PTEN': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'KRAS': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'ARID1A': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'BRAF': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'APC': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'IDH1': {'Gain_Or_Loss_Or_Unknown_Or_NaN': 0, 'switch': 1, 'Other': 2},
        'KMT2D': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'FBXW7': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CDKN2A': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'NF1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'RB1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'KMT2C': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ATRX': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CTNNB1': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'NRAS': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'FAT1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'PBRM1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'PIK3R1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ATM': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'RNF43': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'EGFR': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'KDM6A': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ARID2': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CDH1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'SETD2': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CTCF': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'EP300': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'NFE2L2': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'CIC': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'SMAD4': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'VHL': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'KMT2B': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'GATA3': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CREBBP': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ZFHX3': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ERBB2': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'MAP3K1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'JAK1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'NSD1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'STAG2': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'KMT2A': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'BRCA2': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'MGA': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'BAP1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'MSH6': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'HRAS': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'SPOP': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'B2M': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'NOTCH1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'BCORL1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CASP8': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2}
    }

    gene_data = {}
    for gene_symbol in gene_symbols.keys():
        df1 = mutation_df.loc[gene_symbol, :]
        if len(df1) > 0:
            gene_data[gene_symbol] = {k:str(v).lower() for k,v in df1.to_dict().items()}

    case_ids = mutation_df.columns.values
    gene_mut_labels = {}
    # create gene mutation labels
    for gene_symbol, labels_dict in gene_symbols.items():
        if gene_symbol not in gene_data:
            continue
        values = []
        labels_dict_reverse = {v: k.lower() for k, v in labels_dict.items()}
        for case_id in case_ids:
            if case_id in gene_data[gene_symbol]:
                label = gene_data[gene_symbol][case_id]
                if 'gain' in label: label = 'gain'
                if 'loss' in label: label = 'loss'
                if 'switch' in label: label = 'switch'
                if 'nan' in label: label = 'nan'
                if 'unknown' in label or 'inconclusive' in label: label = 'unknown'
                found = False
                for k, v in labels_dict_reverse.items():
                    if label in v:
                        values.append(k)
                        found = True
                        break
                if not found:
                    print(label)
                    values.append(0)
            else:
                values.append(2)

        if len(values) > 0 and len(np.unique(values)) > 1:
            gene_mut_labels['{}_cls'.format(gene_symbol)] = values
    gene_mut_labels = pd.DataFrame(gene_mut_labels, index=case_ids)
    gene_mut_labels.index.name = 'barcode'
    gene_mut_labels.to_csv('/data/zhongz2/CPTAC/gene_mutation_classification_labels.csv')


    # MsigDB Hallmark 50 gene sets
    with open('/data/zhongz2/HistoVAE/h.all.v2022.1.Hs.json', 'r') as fp:
        hallmark_dict = json.load(fp)
        hallmark_gene_symbols = []
        for k,v in hallmark_dict.items():
            hallmark_gene_symbols.extend(v['geneSymbols'])
        hallmark_gene_symbols = set(hallmark_gene_symbols)
    
    # gene set expression
    gene_exp_files = glob.glob('/data/zhongz2/CPTAC/RNA_BCM_v1/*_gene_RSEM_coding_UQ_1500_log2_Tumor.txt')
    alldfs = []
    for f in gene_exp_files:
        df = pd.read_csv(f, sep='\t', low_memory=False, index_col=0)
        alldfs.append(df)
    gene_exp_df = pd.concat(alldfs, axis=1)
    del alldfs
    gene_exp_df = gene_exp_df.sort_index()
    gene_prefixes = [v[:15] for v in gene_exp_df.index.values]
    gene_prefixes_df = pd.DataFrame(gene_prefixes, columns=['prefix'])
    gene_prefixes_df.index = gene_exp_df.index.values
    gene_prefixes_df = gene_prefixes_df.drop_duplicates(subset=['prefix'], keep='first')
    # gene_map
    ensembl = pd.read_csv('/home/zhongz2/ST_prediction/ensembl.tsv', sep='\t')
    ensembl = ensembl[ensembl['Ensembl ID(supplied by Ensembl)'].notna()]
    gene_prefixes_df['Hugo_Symbol'] = gene_prefixes_df['prefix'].map(dict(zip(ensembl['Ensembl ID(supplied by Ensembl)'], ensembl['Approved symbol'])))
    gene_prefixes_df = gene_prefixes_df[gene_prefixes_df['Hugo_Symbol'].notna()]
    gene_exp_df = gene_exp_df.loc[gene_prefixes_df.index]
    gene_exp_df.index = gene_prefixes_df['Hugo_Symbol'].values
    gene_exp_df.index.name = 'Hugo_Symbol'

    for v in list(hallmark_gene_symbols):
        if v not in gene_exp_df.index.values:
            print(v)

    for k,v in hallmark_dict.items():
        existed_symbols = set(v['geneSymbols']).intersection(gene_exp_df.index)
        ratio = len(existed_symbols) / float(len(v['geneSymbols']))
        print('{:<45}\t{}\t{}\t{:.2f}'.format(k, len(existed_symbols), len(v['geneSymbols']), ratio))

    gene_exp_df.to_csv('/data/zhongz2/CPTAC/gene_exp.csv')

    # data_tumor = gene_exp_df
    # data_normal = gene_exp_df
    # data = data_tumor.subtract(data_normal.mean(axis=1), axis=0)  # Y   # data_tumor
    # data = data.groupby(data.index).median()
    data = gene_exp_df

    newdata2 = {}
    for hall_key, hall_item_dict in hallmark_dict.items():
        gene_list = [v for v in hall_item_dict['geneSymbols'] if v in data.index.values]
        if len(gene_list) > 0:
            newdata2['{}_sum'.format(hall_key)] = data.loc[gene_list].mean()   # the difference between v5 and v6
    newdata2 = pd.DataFrame(newdata2)

    CTL = data.loc[['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1']].median()

    signature = pd.read_csv('/data/zhongz2/HistoVAE/TIDE_Results/Exclusion_scores/exclusion.signature', delimiter='\t')
    metrics = signature.apply(lambda v: data.corrwith(v))

    signature_dysfunction = pd.read_csv('/data/zhongz2/HistoVAE/run.summary.full.trans.gz', delimiter='\t',
                                        index_col=0).dropna().median(axis=1)
    signature_dysfunction = signature_dysfunction.to_frame()
    signature_dysfunction.index.name = 'gene_name'
    signature_dysfunction.columns = ['Dys']
    dysfunction_metrics = signature_dysfunction.apply(lambda v: data.corrwith(v))

    TIDE_scores = pd.concat([metrics.drop(['Mean'], axis=1), dysfunction_metrics], axis=1)

    CTL = pd.DataFrame(CTL)
    CTL.columns = ['Cytotoxic_T_Lymphocyte']
    gene_set_regression_labels = pd.concat([CTL, TIDE_scores, newdata2], axis=1)
    gene_set_regression_labels.index.name = 'barcode'

    hist_save_dir = os.path.join('/data/zhongz2/CPTAC/hists')
    os.makedirs(hist_save_dir, exist_ok=True)
    try:
        from scipy.stats import pearsonr
        import matplotlib.pyplot as plt

        for col in gene_set_regression_labels.columns:
            corr, pvalue = pearsonr(CTL.values.flatten(), gene_set_regression_labels[col].values)
            print('Corr ({}, {}) is {}(p={}) '.format('CTL', col, corr, pvalue))
            _ = plt.hist(gene_set_regression_labels[col].values, bins='auto')
            plt.savefig(os.path.join(hist_save_dir, '{}_hist.png'.format(col)))
            plt.close('all')
    except:
        print('ERROR: MsigDB hallmark histogram error, check it!')

    gene_set_regression_labels.to_csv('/data/zhongz2/CPTAC/gene_set_regression_labels.csv')

    all_labels = gene_mut_labels.merge(gene_set_regression_labels, left_index=True, right_index=True)
    all_labels.to_csv('/data/zhongz2/CPTAC/all_labels.csv')


def check_results_forCPTAC(model_name='UNI'):

    save_root = '/data/zhongz2/CPTAC/predictions'
    save_root = '/data/zhongz2/CPTAC/predictions_v2'
    os.makedirs('{}/per-cancer'.format(save_root), exist_ok=True)

    results_dir = f'/data/zhongz2/CPTAC/patches_256/{model_name}/pred_files'
    all_files = glob.glob(os.path.join(results_dir, '*.pt'))
    results = {}
    for f in all_files:
        svs_prefix = os.path.splitext(os.path.basename(f))[0]
        results_dict = torch.load(f)
        result = {}
        for k, v in CLASSIFICATION_DICT.items():
            result[k] = results_dict[k+'_logits']
        for k in REGRESSION_LIST:
            result[k] = results_dict[k+'_logits']
        results[svs_prefix] = result

    result_df = pd.DataFrame(results).T
    result_df.index.name = 'svs_prefix'
    result_df = result_df.reset_index(drop=False)

    with open('/data/zhongz2/CPTAC/allsvs/allsvs.txt', 'r') as fp:
        filenames = [line.strip() for line in fp.readlines()]
    df = pd.DataFrame(filenames, columns=['orig_filename'])
    df['svs_prefix'] = [os.path.splitext(os.path.basename(f))[0] for f in df['orig_filename'].values]
    df['cancer_type'] = [f.split('/')[-2] for f in df['orig_filename'].values]

    result_df1 = result_df.merge(df, left_on='svs_prefix', right_on='svs_prefix').reset_index(drop=True)

    all_labels = pd.read_csv('/data/zhongz2/CPTAC/all_labels.csv', index_col=0)
    # ['OV', 'BRCA', 'COAD', 'LUAD', 'CCRCC', 'UCEC', 'GBM', 'PDA', 'SAR', 'LSCC', 'CM', 'AML', 'HNSCC']
    all_scores = {}

    for cancer_type in result_df1['cancer_type'].unique():

        result_df2 = result_df1[result_df1['cancer_type'] == cancer_type].reset_index(drop=True)
        if len(result_df2) == 0:
            continue

        barcodes = []
        for svs_prefix in result_df2['svs_prefix'].values:
            found = False
            for v in all_labels.index.values:
                if v in svs_prefix:
                    found = True
                    break
            if found:
                barcodes.append(v)
            else:
                barcodes.append('')
            
        result_df2['barcode'] = barcodes

        result_df2 = result_df2[result_df2['barcode'].isin(all_labels.index)].reset_index(drop=True)
        labels = all_labels.loc[result_df2['barcode'].values]

        if len(result_df2) ==0 or len(labels) == 0:
            continue

        labels.to_csv(f'{save_root}/per-cancer/{cancer_type}_{model_name}_labels.csv')
        result_df2.to_csv(f'{save_root}/per-cancer/{cancer_type}_{model_name}_predictions.csv')

        results = []
        for k, v in CLASSIFICATION_DICT.items():
            if k not in labels.columns:
                results.append(0)
                continue

            valid_ind = ~labels[k].isin([np.nan, IGNORE_INDEX_DICT[k]])
            gt = labels.loc[valid_ind, k]
            a,b = np.unique(gt.values, return_counts=True)
            if len(a) == 0 or b[0] <= 5 or b[1] <= 5:
                results.append(0)
                continue

            try:
                logits = np.concatenate(result_df2.loc[np.where(valid_ind)[0], k].values)
                probs = softmax_stable(logits)
                probs = np.delete(probs, IGNORE_INDEX_DICT[k], axis=1)
                probs = softmax_stable(probs)
                # logits = np.delete(logits, IGNORE_INDEX_DICT[k], axis=1)
                # probs = softmax_stable(logits)
                preds = np.argmax(probs, axis=1)
                auc = roc_auc_score(y_true=gt, y_score=probs[:, 1], average='weighted', multi_class='ovo',
                                        labels=np.arange(2))
                # cm = confusion_matrix(y_true=gt, y_pred=preds)
                results.append(auc)
            except Exception as error:
                print(k, error)
                results.append(0)

        for k in REGRESSION_LIST:
            if k not in labels.columns and k[5:] not in labels.columns:
                results.append(0)
                continue
            logits = np.concatenate(result_df2.loc[:, k].values)
            if 'TIDE_' == k[:5]:
                k = k[5:]
            gt = labels.loc[:, k]
            r2score = r2_score(gt, logits)
            pearson_corr, pearsonr_pvalue = pearsonr(gt, logits)
            spearmanr_corr, spearmanr_pvalue = spearmanr(gt, logits)
            results.append(spearmanr_corr)

        all_scores[cancer_type] = np.array(results).reshape(1, -1)

    results = pd.DataFrame(np.concatenate([v for k,v in all_scores.items()]), columns=list(CLASSIFICATION_DICT.keys())+REGRESSION_LIST)
    results.index = list(all_scores.keys())

    results.to_csv(f'{save_root}/{model_name}_prediction_scores.csv')


def do_results():

    for model_name in ['UNI', 'ProvGigaPath', 'CONCH']:
        check_results_forCPTAC(model_name=model_name)

def check_results_forTCGA_v2(model_name='CONCH'):

    import os
    import pandas as pd
    import numpy as np
    from common import CLASSIFICATION_DICT, REGRESSION_LIST, IGNORE_INDEX_DICT, PAN_CANCER_SITES
    import torch
    from sklearn.metrics import confusion_matrix, f1_score, auc, roc_auc_score, roc_curve, classification_report, r2_score
    from scipy.stats import percentileofscore, pearsonr, spearmanr


    save_root = '/data/zhongz2/CPTAC/predictions'
    save_root = '/data/zhongz2/CPTAC/predictions_v2_TCGA'
    os.makedirs('{}/per-cancer'.format(save_root), exist_ok=True)

    best_splits = {
        'CONCH': 3,
        'UNI': 3,
        'ProvGigaPath': 1
    }
    best_epochs = {
        'CONCH': 53,
        'UNI': 58,
        'ProvGigaPath': 39
    }

    split = best_splits[model_name]
    all_labels = pd.read_csv(f'./splits/test-{split}.csv', low_memory=False)
    all_labels['cancer_type'] = all_labels['PanCancerSiteID'].map({site_id+1: site_name for site_id, site_name in enumerate(PAN_CANCER_SITES)})
    all_labels['svs_prefix'] = [os.path.splitext(os.path.basename(row['DX_filename']))[0] for _, row in all_labels.iterrows()]
    all_labels = all_labels.set_index('svs_prefix')

    results_dir = f'/data/zhongz2/download/TCGA_test{split}/{model_name}/pred_files'
    results = {}
    for svs_prefix, row in all_labels.iterrows():
        results_dict = torch.load(os.path.join(results_dir, svs_prefix+'.pt'))
        result = {}
        for k, v in CLASSIFICATION_DICT.items():
            result[k] = results_dict[k+'_logits']
        for k in REGRESSION_LIST:
            result[k] = results_dict[k+'_logits']
        results[svs_prefix] = result

    result_df = pd.DataFrame(results).T
    result_df.index.name = 'svs_prefix'
    result_df = result_df.reset_index(drop=False)
    result_df['cancer_type'] = all_labels.loc[result_df['svs_prefix'].values, 'cancer_type'].values
    result_df1 = result_df

    all_scores = {}

    for cancer_type in result_df1['cancer_type'].unique():

        result_df2 = result_df1[result_df1['cancer_type'] == cancer_type].reset_index(drop=True)
        if len(result_df2) == 0:
            continue

        barcodes = []
        for svs_prefix in result_df2['svs_prefix'].values:
            found = False
            for v in all_labels.index.values:
                if v in svs_prefix:
                    found = True
                    break
            if found:
                barcodes.append(v)
            else:
                barcodes.append('')
            
        result_df2['barcode'] = barcodes

        result_df2 = result_df2[result_df2['barcode'].isin(all_labels.index)].reset_index(drop=True)
        labels = all_labels.loc[result_df2['barcode'].values]

        if len(result_df2) ==0 or len(labels) == 0:
            continue

        labels.to_csv(f'{save_root}/per-cancer/{cancer_type}_{model_name}_labels.csv')
        result_df2.to_csv(f'{save_root}/per-cancer/{cancer_type}_{model_name}_predictions.csv')

        results = []
        for k, v in CLASSIFICATION_DICT.items():
            if k not in labels.columns:
                results.append(0)
                continue

            valid_ind = ~labels[k].isin([np.nan, IGNORE_INDEX_DICT[k]])
            gt = labels.loc[valid_ind, k]
            a,b = np.unique(gt.values, return_counts=True)
            if len(a) == 0: # or b[0] <= 5 or b[1] <= 5:
                results.append(0)
                continue

            try:
                logits = np.concatenate(result_df2.loc[np.where(valid_ind)[0], k].values)
                probs = softmax_stable(logits)
                probs = np.delete(probs, IGNORE_INDEX_DICT[k], axis=1)
                probs = softmax_stable(probs)
                # logits = np.delete(logits, IGNORE_INDEX_DICT[k], axis=1)
                # probs = softmax_stable(logits)
                preds = np.argmax(probs, axis=1)
                auc = roc_auc_score(y_true=gt, y_score=probs[:, 1], average='weighted', multi_class='ovo',
                                        labels=np.arange(2))
                # cm = confusion_matrix(y_true=gt, y_pred=preds)
                results.append(auc)
            except Exception as error:
                print(k, error)
                results.append(0)

        for k in REGRESSION_LIST:
            if k not in labels.columns and k[5:] not in labels.columns:
                results.append(0)
                continue
            try:
                logits = np.concatenate(result_df2.loc[:, k].values)
                # if 'TIDE_' == k[:5]:
                #     k = k[5:]
                gt = labels.loc[:, k]
                r2score = r2_score(gt, logits)
                pearson_corr, pearsonr_pvalue = pearsonr(gt, logits)
                spearmanr_corr, spearmanr_pvalue = spearmanr(gt, logits)
                results.append(spearmanr_corr)
            except Exception as error:
                print(k, error)
                results.append(0)

        all_scores[cancer_type] = np.array(results).reshape(1, -1)

    results = pd.DataFrame(np.concatenate([v for k,v in all_scores.items()]), columns=list(CLASSIFICATION_DICT.keys())+REGRESSION_LIST)
    results.index = list(all_scores.keys())

    results.to_csv(f'{save_root}/{model_name}_prediction_scores.csv')


def do_results_TCGA():

    for model_name in ['UNI', 'ProvGigaPath', 'CONCH']:
        check_results_forTCGA_v2(model_name=model_name)


def check_results_forTCGA():

    import os
    import pandas as pd
    import numpy as np
    from common import CLASSIFICATION_DICT, REGRESSION_LIST, IGNORE_INDEX_DICT
    import torch
    from sklearn.metrics import confusion_matrix, f1_score, auc, roc_auc_score, roc_curve, classification_report, r2_score
    from scipy.stats import percentileofscore, pearsonr, spearmanr

    for model_name in ['CONCH', 'UNI', 'ProvGigaPath']:
        scores = []
        for split in range(5):
            csv_path = f'./splits/test-{split}.csv'
            df = pd.read_csv(csv_path, low_memory=False)

            results_dir = f'/data/zhongz2/download/TCGA_test{split}/{model_name}/pred_files'
            results = []
            for _, row in df.iterrows():
                svs_prefix = os.path.splitext(os.path.basename(row['DX_filename']))[0]
                results_dict = torch.load(os.path.join(results_dir, svs_prefix+'.pt'))
                result = {}
                for k, v in CLASSIFICATION_DICT.items():
                    result[k] = results_dict[k+'_logits']
                for k in REGRESSION_LIST:
                    result[k] = results_dict[k+'_logits']
                results.append(result)

            result_df = pd.DataFrame(results)

            results = []
            for k, v in CLASSIFICATION_DICT.items():
                valid_ind = ~df[k].isin([np.nan, IGNORE_INDEX_DICT[k]])

                gt = df.loc[valid_ind, k]
                logits = np.concatenate(result_df.loc[valid_ind, k].values)
                probs = softmax_stable(logits)
                probs = np.delete(probs, IGNORE_INDEX_DICT[k], axis=1)
                probs = softmax_stable(probs)
                # logits = np.delete(logits, IGNORE_INDEX_DICT[k], axis=1)
                # probs = softmax_stable(logits)
                # preds = np.argmax(probs, axis=1)
                auc = roc_auc_score(y_true=gt, y_score=probs[:, 1], average='weighted', multi_class='ovo',
                                        labels=np.arange(2))
                # cm = confusion_matrix(y_true=gt, y_pred=preds)
                results.append(auc)
            
            for k in REGRESSION_LIST:
                gt = df.loc[:, k]
                logits = np.concatenate(result_df.loc[:, k].values)
                r2score = r2_score(gt, logits)
                pearson_corr, pearsonr_pvalue = pearsonr(gt, logits)
                spearmanr_corr, spearmanr_pvalue = spearmanr(gt, logits)
                results.append(spearmanr_corr)

            scores.append(results)   
        scores = pd.DataFrame(scores, columns=list(CLASSIFICATION_DICT.keys())+REGRESSION_LIST)

if __name__ == '__main__':
    main()









