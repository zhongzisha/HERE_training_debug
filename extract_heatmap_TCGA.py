


import sys,os,glob,shutil,json
import numpy as np
import pandas as pd
import openslide
import pyvips
import h5py
import torch
from scipy.stats import percentileofscore
from utils import save_hdf5, visHeatmap, score2percentile, to_percentiles
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pyvips
import idr_torch
import gc
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def clustering(X0, n_clusters=3, top_n=5):

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X0)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Get distances to centroids
    distances = kmeans.transform(X)  # shape: (n_samples, n_clusters)

    # For each cluster, find top N closest samples
    top_nearest_indices = []
    cluster_ids = []
    for cluster_id in range(kmeans.n_clusters):
        # Get distances to this centroid
        dists_to_cluster = distances[:, cluster_id]
        
        # Get indices of samples assigned to this cluster
        cluster_members = np.where(kmeans.labels_ == cluster_id)[0]
        
        # Sort those members by distance to centroid
        sorted_members = cluster_members[np.argsort(dists_to_cluster[cluster_members])]
        
        # Take top N
        top_n_ids = sorted_members[:min(top_n, len(sorted_members))]
        top_nearest_indices.append(top_n_ids)
        cluster_ids.append(np.ones_like(top_n_ids)*cluster_id)
    
    return np.concatenate(top_nearest_indices), np.concatenate(cluster_ids)


def main():

    prefix = sys.argv[1] # TCGA_trainval3 or TCGA_test3

    svs_dir = "/data/zhongz2/tcga/TCGA-ALL2_256/svs"
    patches_dir = "/data/zhongz2/tcga/TCGA-ALL2_256/patches"
    backbone = 'CONCH'
    feats_dir = f'/data/zhongz2/download/{prefix}/{backbone}/pt_files'
    preds_dir = f'/data/zhongz2/download/{prefix}/{backbone}/pred_files'

    save_root = preds_dir.replace('pred_files', 'heatmap_files')
    save_root = preds_dir.replace('pred_files', 'heatmap_files_check') # for Revision 2 20250411
    os.makedirs(save_root, exist_ok=True)
    save_root_top_patches = preds_dir.replace('pred_files', 'heatmap_top_patches')
    os.makedirs(save_root_top_patches, exist_ok=True)

    pt_files = glob.glob(os.path.join(preds_dir, '*.pt'))
    # existed_prefixes = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(save_root, '*.tif'))]
    existed_prefixes = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(save_root) if f[0]!='.']
    needtodo_files = [f for f in pt_files if os.path.splitext(os.path.basename(f))[0]+"_heatmap" not in existed_prefixes]

    print('existing files: ', len(existed_prefixes))
    print('needtodo', len(needtodo_files))
    indices = np.arange(len(needtodo_files))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    # print('index_splits', index_splits)
    needtodo_files = [needtodo_files[i] for i in index_splits[idr_torch.rank]]
    print(idr_torch.rank, len(needtodo_files))

    top_bottom = {
        'top':0.9,
        'bottom':0.1
    }
    for f in needtodo_files:

        svs_prefix = os.path.splitext(os.path.basename(f))[0]

        A_raw = torch.load(os.path.join(preds_dir, svs_prefix+'.pt'))['A_raw']

        with h5py.File(os.path.join(patches_dir, svs_prefix+'.h5'), 'r') as file:
            all_coords = file['coords'][:]
            patch_size = file['coords'].attrs['patch_size']
            patch_level = file['coords'].attrs['patch_level']

        slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))

        if True: # try:
            A = np.copy(A_raw)[0]
            attention_scores = to_percentiles(A)
            attention_scores /= 100

            save_filename = '{}/{}_heatmap.tif'.format(save_root, svs_prefix)
            img, attention_scores = visHeatmap(slide, scores=A, coords=all_coords,
                            vis_level=0, patch_size=(patch_size, patch_size),
                            convert_to_percentiles=True, return_scores=True)
            print(type(img), img.size)
            # img.save(save_filename)
            img_vips = pyvips.Image.new_from_array(img)
            # img_vips.dzsave(save_filename, tile_size=1024)
            img_vips.tiffsave(save_filename, compression="jpeg",
                tile=True, tile_width=256, tile_height=256,
                pyramid=True,  bigtiff=True)
            del img, img_vips
            gc.collect()

            save_dir = os.path.join(save_root_top_patches, svs_prefix)
            os.makedirs(save_dir, exist_ok=True)
            feats = torch.load(os.path.join(feats_dir, svs_prefix+'.pt'))
            for tb,tb_thres in top_bottom.items():
                if tb=='top':
                    red_inds = np.where(attention_scores>tb_thres)[0]
                else:
                    red_inds = np.where(attention_scores<tb_thres)[0]
                red_feats = feats[red_inds].cpu().numpy()
                red_top_inds, red_cluster_ids = clustering(red_feats)
                for ind, cid in zip(red_top_inds, red_cluster_ids):
                    x,y = all_coords[ind]
                    patch = slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert('RGB')
                    patch.save(os.path.join(save_dir, f'{tb}_c{cid}_patch_x{x}_y{y}.jpg'))

            # img_vips.write_to_file(save_filename, tile=True, compression="jpeg", bigtiff=True, pyramid=True)
            # time.sleep(1)
            # del img, img_vips
        # except:
        #     print(f'error {f}')


if __name__ == '__main__':
    main()