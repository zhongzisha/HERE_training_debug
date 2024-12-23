


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

    prefix = sys.argv[1] # TCGA_trainval3 or TCGA_test3

    svs_dir = "/data/zhongz2/tcga/TCGA-ALL2_256/svs"
    patches_dir = "/data/zhongz2/tcga/TCGA-ALL2_256/patches"
    backbone = 'CONCH'
    preds_dir = f'/data/zhongz2/download/{prefix}/{backbone}/pred_files'

    save_root = preds_dir.replace('pred_files', 'heatmap_files')
    os.makedirs(save_root, exist_ok=True)

    pt_files = glob.glob(os.path.join(preds_dir, '*.pt'))
    # existed_prefixes = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(save_root, '*.tif'))]
    existed_prefixes = [f for f in os.listdir(save_root) if '.tif' not in f and f[0]!='.']
    needtodo_files = [f for f in pt_files if os.path.splitext(os.path.basename(f))[0] not in existed_prefixes]

    print('existing files: ', len(existed_prefixes))
    print('needtodo', len(needtodo_files))
    indices = np.arange(len(needtodo_files))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    # print('index_splits', index_splits)
    needtodo_files = [needtodo_files[i] for i in index_splits[idr_torch.rank]]
    print(idr_torch.rank, len(needtodo_files))


    for f in needtodo_files:

        svs_prefix = os.path.splitext(os.path.basename(f))[0]

        A_raw = torch.load(os.path.join(preds_dir, svs_prefix+'.pt'))['A_raw']

        with h5py.File(os.path.join(patches_dir, svs_prefix+'.h5'), 'r') as file:
            all_coords = file['coords'][:]
            patch_size = file['coords'].attrs['patch_size']
            patch_level = file['coords'].attrs['patch_level']

        slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))

        try:
            A = np.copy(A_raw)[0]
            save_filename = '{}/{}_heatmap.tif'.format(save_root, svs_prefix)
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