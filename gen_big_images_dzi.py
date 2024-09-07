import sys,os,glob
import openslide
import idr_torch
import time
import pyvips
import pickle
import h5py
import pandas as pd
import numpy as np
from utils import get_svs_prefix, visWSI
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():

    svs_dir = sys.argv[1]
    save_root = sys.argv[2]
    os.makedirs(save_root, exist_ok=True)

    files = glob.glob(os.path.join(svs_dir, '*.svs'))
    existed_prefixes = set([os.path.basename(f).replace('_big_orig.zip', '') for f in glob.glob(os.path.join(save_root, '*.zip'))])
    files = [f for f in files if os.path.basename(f).replace('.svs', '') not in existed_prefixes]

    indices = np.arange(len(files))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)

    print(idr_torch.rank, len(index_splits[idr_torch.rank]))

    for ind in index_splits[idr_torch.rank]:
        svs_filename = files[ind]

        local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank),
                                      str(idr_torch.local_rank))
        os.makedirs(local_temp_dir, exist_ok=True)

        svs_prefix = get_svs_prefix(svs_filename)
        svs_filename1 = os.path.realpath(svs_filename)
        local_svs_filename = os.path.join(local_temp_dir, os.path.basename(svs_filename1))
        if not os.path.exists(local_svs_filename):
            os.system(f'cp -RL "{svs_filename1}" "{local_svs_filename}"')

        slide = openslide.open_slide(local_svs_filename)
        dimension = slide.level_dimensions[1] if len(slide.level_dimensions) > 1 else slide.level_dimensions[0]
        if dimension[0] > 100000 or dimension[1] > 100000:
            vis_level = 2
        else:
            vis_level = 1
        if len(slide.level_dimensions) == 1:
            vis_level = 0

        save_filename = '{}/{}_big_orig.zip'.format(save_root, svs_prefix)
        if not os.path.exists(save_filename):
            img = visWSI(slide, vis_level=vis_level)
            img_vips = pyvips.Image.new_from_array(img)
            img_vips.dzsave(save_filename, tile_size=1024)
            time.sleep(1)
            del img, img_vips

        if os.path.exists(local_svs_filename):
            os.system(f'rm -rf "{local_svs_filename}"')


if __name__ == '__main__':
    main()



















