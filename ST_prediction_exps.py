import sys,os,shutil,json,h5py,glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import openslide
import pickle
from sklearn.metrics import r2_score
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def test_vst_vis(): # fine
    sc.logging.print_versions()
    sc.set_figure_params(facecolor="white", figsize=(8, 8))
    sc.settings.verbosity = 3

    adata = sc.datasets.visium_sge(sample_id="CytAssist_11mm_FFPE_Human_Lung_Cancer")
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pl.spatial(adata, color="log1p_n_genes_by_counts", cmap="hsv", save=True)



def create_data():

    root = '/data/zhongz2/ST_20240903'
    df = pd.read_excel(f'{root}/ST_20240907.xlsx')
    svs_dir = os.path.join(root, 'svs')
    patches_dir = os.path.join(root, 'patches')
    gene_vst_dir = os.path.join(root, 'gene_vst')

    human_slide_ids = {
        '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0',
        '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0',
        '10x_CytAssist_FFPE_Protein_Expression_Human_Tonsil_2.1.0',
        '10x_CytAssist_Fresh_Frozen_Human_Breast_Cancer_2.0.1',
        '10x_Targeted_Visium_Human_BreastCancer_Immunology_1.2.0',
        '10x_V1_Breast_Cancer_Block_A_Section_1_1.1.0',
        '10x_V1_Breast_Cancer_Block_A_Section_2_1.1.0',
        '10x_Visium_FFPE_Human_Cervical_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Intestinal_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Ovarian_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_1.3.0',
        '10x_Visium_Human_Breast_Cancer_1.3.0',
        'ST1K4M_Human_Breast_10X_06092021_Visium',
        'ST1K4M_Human_Colon_10X_10052023_Visium_control_rep1',
        'ST1K4M_Human_Colon_10X_10052023_Visium_control_rep2',
        'ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep1',
        'ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep2',
        'ST1K4M_Human_Prostate_10X_06092021_Visium_cancer',
        'ST1K4M_Human_Prostate_10X_06092021_Visium_normal',
        'ST1K4M_Human_Prostate_10X_07122022_Visium'
    }
    df = df[df['slide_id'].isin(human_slide_ids)].reset_index(drop=True)

    # get common genes
    gene_names = {}
    for rowid, row in df.iterrows():
        svs_prefix = row['slide_id']
        vst_filename_db = os.path.join(root, 'vst_dir_db', svs_prefix+'_original_VST.db')
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        gene_names[svs_prefix] = [v for v in existing_columns if '__' != v[:2]]
    gene_names1 = list(gene_names.values())
    selected_gene_names = sorted(list(set(gene_names1[0]).intersection(*gene_names1[1:])))

    cache_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'cache_v1')
    os.makedirs(cache_dir, exist_ok=True)

    for rowid, row in df.iterrows():
        svs_prefix = row['slide_id']
        save_filename = os.path.join(cache_dir, svs_prefix+'.pkl')
        svs_filename = os.path.join(root, 'svs', svs_prefix+'.svs')
        vst_filename_db = os.path.join(root, 'vst_dir_db', svs_prefix+'_original_VST.db')
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        meta_columns = ['__barcode', '__spot_X', '__spot_Y', '__upperleft_X', '__upperleft_Y']

        query_columns = ['__upperleft_X', '__upperleft_Y'] + selected_gene_names
        vst_df = pd.read_parquet(vst_filename_db, columns=query_columns)

        spot_size = row['spot_size']
        patch_size = int(np.ceil(1.1 * spot_size)) # expand some area (10% here)
        st_patch_size = patch_size

        if False:
            with h5py.File(os.path.join(patches_dir, svs_prefix+'.h5'), 'r') as file:
                coords = file['coords'][()]
                print(file['coords'].attrs['patch_size'])

        slide = openslide.open_slide(svs_filename)
        data = []
        mean = np.zeros(3, dtype=np.float32)
        std = np.zeros(3, dtype=np.float32)
        for _, row1 in vst_df.iterrows():
            x,y = int(row1['__upperleft_X']), int(row1['__upperleft_Y'])
            gt = row1[selected_gene_names].values
            patch = slide.read_region(location=(x,y), level=0, size=(st_patch_size, st_patch_size)).convert('RGB')
            # data.append((patch, gt))
            patch = np.array(patch)
            mean += patch.mean((0, 1))
            std += patch.std((0, 1)) 
            mean1 = patch.mean((0, 1))
            std1 = patch.std((0, 1))
            if np.unique(std1).shape[0] == 1:
                continue
            patch1 = (patch - mean1)/std1
            data.append((patch1, gt))
        mean /= len(vst_df)
        std /= len(vst_df)

        with open(save_filename, 'wb') as fp:
            pickle.dump({'data': data, 'mean': mean, 'std': std}, fp)
        
        print(svs_prefix)



class PatchDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, label = self.data[idx]
        return self.transform(img), label


def main():

    max_epochs = 100
    device = torch.device('cuda:0')
    # data
    cache_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'cache')
    files = sorted(glob.glob(os.path.join(cache_dir, '*.pkl')))

    val_prefixes = [
        '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1'
    ]
    train_data = []
    val_data = []
    for f in files:
        svs_prefix = os.path.basename(f).replace('.pkl', '')
        with open(f, 'rb') as fp:
            data = pickle.load(fp)
        if svs_prefix in val_prefixes:
            val_data.extend(data['data'])
        else:
            train_data.extend(data['data'])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    
    train_dataset = PatchDataset(train_data, transform=train_transform)
    val_dataset = PatchDataset(val_data, transform=val_transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    # model
    num_classes = 778  # from selected_gene_names
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.xavier_uniform_(model.fc.weight)

    criterion = nn.MSELoss()
    lr = 1e-4
    weight_decay = 1e-4
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # params_1x are the parameters of the network body, i.e., of all layers except the FC layers
    params_1x = [param for name, param in net.named_parameters() if 'fc' not in str(name)]
    optimizer = torch.optim.Adam([{'params':params_1x}, {'params': net.fc.parameters(), 'lr': lr*10}], lr=lr, weight_decay=weight_decay)

    model.to(device)

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        train_scores = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            train_scores.extend([r2_score(labels[j], preds[j]) for j in range(len(preds))])
            break 

        model.eval()
        val_scores = []
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            # labels = labels.to(device)
            labels = labels.cpu().numpy()

            with torch.no_grad():
                preds = model(images).detach().cpu().numpy()

            val_scores.extend([r2_score(labels[j], preds[j]) for j in range(len(preds))])

        print(epoch, np.mean(train_scores), np.mean(val_scores))
            
