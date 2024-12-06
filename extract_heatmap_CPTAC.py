


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

def main():

    svs_dir = '/data/zhongz2/CPTAC/svs'
    patches_dir = '/data/zhongz2/CPTAC/patches_256/patches'
    backbone = 'CONCH'
    preds_dir = f'/data/zhongz2/CPTAC/patches_256/{backbone}/pred_files'

    save_root = f'/data/zhongz2/CPTAC/patches_256/{backbone}/heatmap_files'
    os.makedirs(save_root, exist_ok=True)

    patch_files = glob.glob(os.path.join(patches_dir, '*.h5'))
    existed_prefixes = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(save_root, '*.tif'))]
    needtodo_files = [f for f in patch_files if os.path.splitext(os.path.basename(f))[0] not in existed_prefixes]

    print('existing files: ', len(existed_prefixes))
    print('needtodo', len(needtodo_files))
    indices = np.arange(len(needtodo_files))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    # print('index_splits', index_splits)
    needtodo_files = [needtodo_files[i] for i in index_splits[idr_torch.rank]]
    print(idr_torch.rank, len(needtodo_files))

    num_patches = 8

    for f in needtodo_files:

        svs_prefix = os.path.splitext(os.path.basename(f))[0]

        A_raw = torch.load(os.path.join(preds_dir, svs_prefix+'.pt'))['A_raw']

        with h5py.File(f, 'r') as file:
            all_coords = file['coords'][:]
            patch_size = file['coords'].attrs['patch_size']
            patch_level = file['coords'].attrs['patch_level']

        slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))

        # select top patches
        A = np.copy(A_raw)[0]
        ref_scores = np.copy(A)
        for ind1 in range(len(A)):
            A[ind1] = percentileofscore(ref_scores, A[ind1])

        sort_ind = np.argsort(A)[::-1]
        num_ps = num_patches if len(sort_ind) > num_patches else len(sort_ind)
        selected_indices = sort_ind[:int(num_ps)]
        selected_indices1 = sort_ind[-int(num_ps):]

        for size in [256, 1024]:
            save_dir = os.path.join(save_root, svs_prefix, f'patch{size}')
            if os.path.exists(save_dir):
                continue
            os.makedirs(save_dir, exist_ok=True)

            for ri, ind in enumerate(selected_indices):
                x, y = all_coords[ind]
                cx, cy = int(x+patch_size//2), int(y+patch_size//2)
                x1, y1 = int(cx - size//2), int(cy - size//2)

                patch = slide.read_region((x1, y1), level=patch_level, size=(size, size)).convert('RGB')
                patch.save(os.path.join(save_dir, f'top{ri}.png'))
            

            for ri, ind in enumerate(selected_indices1):
                rii = len(selected_indices1) - ri - 1
                x, y = all_coords[ind]
                cx, cy = int(x+patch_size//2), int(y+patch_size//2)
                x1, y1 = int(cx - size//2), int(cy - size//2)

                patch = slide.read_region((x1, y1), level=patch_level, size=(size, size)).convert('RGB')
                patch.save(os.path.join(save_dir, f'bot{ri}.png'))

        try:

            A = np.copy(A_raw)[0]
            save_filename = '{}/{}.tif'.format(save_root, svs_prefix)
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
        except:
            print(f'error {f}')


if __name__ == '__main__':
    main()