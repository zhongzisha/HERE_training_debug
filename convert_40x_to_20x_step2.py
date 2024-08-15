import sys,os,glob,shutil
import numpy as np
import pandas as pd
import openslide
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
from natsort import natsorted
import pickle
import idr_torch
import time


def main():
    csv_filename = sys.argv[1]
    save_root = sys.argv[2]
    image_ext = sys.argv[3]
    df = pd.read_csv(csv_filename)

    if idr_torch.rank == 0:
        os.makedirs(os.path.join(save_root, '..', 'svs'), exist_ok=True)

    local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank), str(idr_torch.local_rank))
    os.makedirs(local_temp_dir, exist_ok=True)

    dest_files = glob.glob(os.path.join(save_root, '*.tif'))
    existed_prefixes = set([os.path.basename(f).replace('.tif', '') for f in dest_files])

    drop_ids = []
    for ind, svs_prefix in enumerate(df['prefix'].values):
        if svs_prefix in existed_prefixes:
            drop_ids.append(ind)
    if len(drop_ids) > 0:
        df = df.drop(drop_ids)

    df = df.reset_index(drop=True)

    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    print('index_splits', index_splits)
    print('rank: ', idr_torch.rank)
    print('world_size: ', idr_torch.world_size)
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)

    time.sleep(1)
    for index, row in sub_df.iterrows():
        slide_file_path = row['orig_filepath']

        if not os.path.exists(slide_file_path):
            print('{} not existed'.format(slide_file_path))
            continue

        svs_prefix = row['prefix']  # os.path.basename(slide_file_path).replace(image_ext, '')

        final_filename = os.path.join(save_root, svs_prefix+'.tif')
        if os.path.exists(final_filename):
            print('{} existed'.format(final_filename))
            os.system('ln -sf "{}" "{}"'.format(final_filename, os.path.join(save_root, '..', 'svs', svs_prefix + '.svs')))
            continue

        final_filename_doing = os.path.join(save_root, svs_prefix+'.tif.doing')
        if os.path.exists(final_filename_doing):
            continue

        with open(final_filename_doing, 'w') as fp:
            pass

        local_svs_filename = os.path.join(local_temp_dir, svs_prefix + image_ext)
        os.system(f'cp -RL "{slide_file_path}" "{local_svs_filename}"')
        time.sleep(1)
        local_final_filename = os.path.join(local_temp_dir, svs_prefix+'_final.tif')

        temp_filename = local_svs_filename
        do_pyramid = row['do_pyramid']
        mppx = float(row['mpp'])
        res = 1000 / mppx
        temp_filename2 = None
        if do_pyramid:
            temp_filename = os.path.join(local_temp_dir, svs_prefix + '_temp.tif')
            command1 = 'vips copy "{}" "{}"[pyramid,compression=jpeg,Q=80,tile-width=256,tile-height=256,bigtiff,xres={},yres={}]'.format(
                local_svs_filename, temp_filename, res, res
            )
            command2 = 'tifftools set -y -s ImageDescription  "Aperio Fake |AppMag = 40|MPP = {}" "{}"'.format(
                mppx, temp_filename
            )
            temp_filename2 = temp_filename
        else:
            command1 = None
            command2 = None

        if mppx < 0.15:
            level = 2
        else:
            level = 1

        newmpp = (2 ** level) * mppx
        res = 1000 / newmpp
        command3 = 'vips openslideload "{}" "{}"[pyramid,compression=jpeg,Q=80,tile-width=256,tile-height=256,bigtiff,xres={},yres={}] --level={}'.format(
            temp_filename, local_final_filename, res, res, level
        )

        command4 = 'tifftools set -y -s ImageDescription  "Aperio Fake |AppMag = 20|MPP = {}" "{}"'.format(
            newmpp, local_final_filename
        )

        if command1 is not None:
            os.system(command1)
            time.sleep(1)
            os.system(command2)
            time.sleep(1)

        os.system(command3)
        time.sleep(1)
        os.system(command4)
        time.sleep(1)

        if not os.path.exists(final_filename):
            os.system(f'cp "{local_final_filename}" "{final_filename}"')
            os.system('ln -sf "{}" "{}"'.format(final_filename, os.path.join(save_root, '..', 'svs', svs_prefix + '.svs')))

        os.system(f'rm -rf "{local_svs_filename}" "{local_final_filename}"')
        if temp_filename2 is not None:
            os.system(f'rm -rf "{temp_filename2}"')

        time.sleep(1)
        os.system(f'rm -rf "{final_filename_doing}"')


def check():
    files = glob.glob('*.tif')
    invalid_files = []
    for ind, f in enumerate(files):

        try:
            slide = openslide.open_slide(f)
            factor = slide.level_dimensions[0][0] / slide.level_dimensions[1][0]
            if int(float(factor)) != 2:
                invalid_files.append((f, factor))
        except:
            invalid_files.append((f, -1))

        if ind%1000 == 0:
            print(ind)


if __name__ == '__main__':
    main()











