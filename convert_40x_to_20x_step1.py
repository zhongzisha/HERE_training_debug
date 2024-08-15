import sys,os,glob,shutil
import numpy as np
import pandas as pd
import openslide
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
from natsort import natsorted
import pickle


def main1():
    # Ken
    project = 'KenData_20240814'
    data_dir = '/data/Jiang_Lab/Data/COMPASS_NGS_Cases_20240814'
    save_root = '/data/zhongz2/KenData_20240814/'
    postfix = '.ndpi'
    allfiles = natsorted(glob.glob(os.path.join(data_dir, '*{}'.format(postfix))))

    save_images_dir = os.path.join(save_root, 'images_20x')
    os.makedirs(save_images_dir, exist_ok=True)

    meta_filename = os.path.join(save_root, 'convert_files.pkl')
    if os.path.exists(meta_filename):
        with open(meta_filename, 'rb') as fp:
            tmpdata = pickle.load(fp)
            items_40x_old = tmpdata['items_40x']
            unusual_items_old = tmpdata['unusual_items']
            items_20x_old = tmpdata['items_20x']
            files_old = tmpdata['files']
            del tmpdata
        newfiles = [f for f in allfiles if f not in files_old]        
    else:
        items_40x_old = []
        unusual_items_old = []
        items_20x_old = []
        files_old = []
        newfiles = allfiles
    items_40x = []
    unusual_items = []
    items_20x = []

    for ind, filename in enumerate(newfiles):
        if ind % 100 == 0:
            print(ind)

        slide = openslide.open_slide(filename)
        if 'openslide.mpp-x' not in slide.properties or 'openslide.mpp-y' not in slide.properties:
            unusual_items.append((filename, 0))  # this will remove some images
            continue

        mppx = float(slide.properties['openslide.mpp-x'])
        mppy = float(slide.properties['openslide.mpp-y'])

        prefix = os.path.basename(filename).replace(postfix, '')
        prefix = prefix.replace(' ', '_')
        prefix = prefix.replace(',', '_')
        prefix = prefix.replace('&', '_')
        prefix = prefix.replace('+', '_')

        if 0.4 < mppx < 0.6:  # 20x don't need to convert
            items_20x.append((filename, mppx))
            os.system('ln -sf "{}" "{}"'.format(filename, os.path.join(save_images_dir, prefix+postfix)))
            os.system('ln -sf "{}" "{}"'.format(filename, os.path.join(save_images_dir, '..', 'svs', prefix+'.svs')))
            continue

        if mppx < 0.18 or mppx > 0.6:
            unusual_items.append((filename, mppx))

        if len(slide.level_dimensions) > 1:
            factor = slide.level_dimensions[0][0] / slide.level_dimensions[1][0]
            if int(float(factor)) != 2:
                do_pyramid = True
            else:
                do_pyramid = False
        else:
            do_pyramid = True

        items_40x.append(
            (filename, prefix, mppx, do_pyramid)
        )

    with open(meta_filename, 'wb') as fp:
        pickle.dump({'items_40x': items_40x + items_40x_old, 'unusual_items': unusual_items + unusual_items_old, 'items_20x': items_20x + items_20x_old, 'files': allfiles}, fp)

    df = pd.DataFrame(items_40x, columns=['orig_filepath', 'prefix', 'mpp', 'do_pyramid'])
    existfiles = glob.glob(os.path.join(save_root, '40x_items_to_be_converted_*.csv'))
    df.to_csv(os.path.join(save_root, '40x_items_to_be_converted_{}.csv'.format(len(existfiles)+1)))





















