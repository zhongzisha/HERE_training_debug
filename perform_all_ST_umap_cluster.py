

import sys,os,glob,shutil,json
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import cluster, neighbors
from scipy.spatial import distance_matrix
import h5py

import dash
import dash_deck
from dash import html
import pydeck


root = '/data/zhongz2/ST_256'
df = pd.read_excel(f'{root}/all_20231117.xlsx')
patches_dir = f'{root}/patches'
prefixes = [os.path.basename(f).replace('.h5', '') for f in glob.glob(os.path.join(patches_dir, '*.h5'))]
invalid_slide_ids = [
    '10x_Parent_Visium_Human_Glioblastoma_1.2.0',
    '10x_Targeted_Visium_Human_BreastCancer_Immunology_1.2.0'
]
prefixes = [v for v in prefixes if v not in invalid_slide_ids]
df = df[df['slide_id'].isin(prefixes)].reset_index(drop=True)

slide_ids = df['slide_id'].to_dict()
colors = np.random.randint(0, 255, size=(len(slide_ids), 3), dtype=np.uint8)

for backbone in ['PLIP', 'ProvGigaPath', 'CONCH']:
    save_filename = f'ST_{backbone}_umap3d.csv'
    if os.path.exists(save_filename):
        continue

    feats_dir = f'{root}/feats/{backbone}/pt_files'

    X0 = []
    Y0 = []
    all_coords = []
    for row_ind, row in df.iterrows():
        x = torch.load(os.path.join(feats_dir, row['slide_id']+'.pt'), weights_only=True).cpu().numpy()
        X0.append(x)
        Y0.append(row_ind * np.ones((x.shape[0],), dtype=np.int32))

        with h5py.File(os.path.join(patches_dir, row['slide_id']+'.h5'), 'r') as file:
            all_coords.append(file['coords'][()])
        print(row['slide_id'])
    X0 = np.concatenate(X0)
    Y0 = np.concatenate(Y0)
    all_coords = np.concatenate(all_coords)

    is_reduced_sample = False
    kmeans0_Y = None
    if False:  # len(X0) > 30000:  # downsample the data
        print('too many samples, subsample it')
        # kmeans0 = cluster.KMeans(n_clusters=10000, n_init='auto', random_state=42)
        kmeans0 = cluster.MiniBatchKMeans(n_clusters=50000, n_init='auto', random_state=42, max_iter=100, batch_size=2048)
        kmeans0_Y = kmeans0.fit_predict(X0)
        X = kmeans0.cluster_centers_
        is_reduced_sample = True
    else:
        X = X0
        Y = Y0
        
    feature_normalization_type = 'none'
    dimension_reduction_method = 'umap3d'
    clustering_method = 'kmeans'
    clustering_distance_metric = 'euclidean'
    num_clusters = len(df) * 8

    if feature_normalization_type == 'meanstd':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif feature_normalization_type == 'minmax':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X

    if dimension_reduction_method == 'none':
        feats_embedding = X_scaled
        reducer = None
    elif dimension_reduction_method == 'pca3d':
        reducer = PCA(n_components=3)
        feats_embedding = reducer.fit_transform(X_scaled)
    elif dimension_reduction_method == 'umap3d':
        reducer = umap.UMAP(metric=clustering_distance_metric, n_components=3, n_jobs=-1)
        feats_embedding = reducer.fit_transform(X_scaled)
    else:
        raise ValueError('wrong dimension reduction method.')
    del X_scaled

    data = np.concatenate([Y0[:, None], feats_embedding, all_coords], axis=1)
    alldata_df = pd.DataFrame(data, columns=['slide_id', 'x', 'y', 'z', 'px', 'py'])
    alldata_df['r'] = alldata_df['slide_id'].map({i: v for i, v in enumerate(colors[:, 0])})
    alldata_df['g'] = alldata_df['slide_id'].map({i: v for i, v in enumerate(colors[:, 1])})
    alldata_df['b'] = alldata_df['slide_id'].map({i: v for i, v in enumerate(colors[:, 2])})
    alldata_df = alldata_df.astype({'slide_id':'int32','px':'int32','py':'int32'})
    alldata_df['slide_name'] = alldata_df['slide_id'].map(slide_ids)
    alldata_df.to_csv(save_filename, index=None)


# for backbone in "PLIP" "ProvGigaPath" "CONCH"; do python test_dash.py ${backbone}; done






















