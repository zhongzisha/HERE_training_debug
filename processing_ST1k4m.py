
import sys,os,glob,shutil,json
import numpy as np
import pandas as pd
import openslide
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1266016250
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import h5py
from utils import save_hdf5

def clear_prefix(prefix):
    prefix = prefix.replace(' ', '_')
    prefix = prefix.replace(',', '_')
    prefix = prefix.replace('&', '_')
    prefix = prefix.replace('+', '_')
    return prefix


def main():
    root = '/data/Jiang_Lab/Data/STimage-1K4M/'
    df = pd.read_csv(f'{root}/meta/meta_all_gene_fixed.csv')

    save_root = '/data/zhongz2/ST_20240903'   # combine previous ST and ST1K4M, removing duplicates
    for d in ['svs', 'patches', 'thumbnails']:
        os.makedirs(os.path.join(save_root, d), exist_ok=True)
    svs_save_dir = os.path.join(save_root, 'svs')
    patch_save_dir = os.path.join(save_root, 'patches')
    thumbnail_save_dir = os.path.join(save_root, 'thumbnails')

    TruePaths = []
    counts_filenames = []
    coord_filenames = []
    spot_sizes = []
    original_widths = []
    original_heights = []
    DX_filenames = []
    slide_ids = []
    for _, row in df.iterrows():
        svs_prefix = 'ST1K4M_' + clear_prefix(row['slide'])
        slide_ids.append(svs_prefix)
        save_path_hdf5 = os.path.join(patch_save_dir, svs_prefix+'.h5')
        DX_filename = os.path.join(svs_save_dir, svs_prefix+'.svs')
        DX_filenames.append(DX_filename)
        svs_filename = '{}/{}/image/{}.png'.format(root, row['tech'], row['slide'])
        counts_filename = '{}/{}/gene_exp/{}_count.csv'.format(root, row['tech'], row['slide'])
        coord_filename = '{}/{}/coord/{}_coord.csv'.format(root, row['tech'], row['slide'])
        exist0 = os.path.exists(svs_filename) and os.path.exists(counts_filename) and os.path.exists(coord_filename)
        if not exist0:
            print(row)
            import pdb
            pdb.set_trace()
        TruePaths.append(svs_filename)
        counts_filenames.append(counts_filename)
        coord_filenames.append(coord_filename)
        # barcode_col_name, X_col_name, Y_col_name

        slide = openslide.open_slide(svs_filename)
        original_width, original_height = slide.level_dimensions[0]
        original_widths.append(original_width)
        original_heights.append(original_height)

        thumbnail = slide.get_thumbnail([512, 512])
        thumbnail.save(os.path.join(thumbnail_save_dir, svs_prefix+'.jpg'))

        with pd.read_csv(coord_filename, chunksize=10) as reader:
            for chunk in reader:
                radius = float(chunk['r'].values[0])  # radius
                spot_sizes.append(2*radius)
                break

        os.system('ln -sf "{}" "{}"'.format(svs_filename, DX_filename))

    barcode_col_names = ['Unnamed: 0' for _ in range(len(TruePaths))]
    X_col_names = ['xaxis' for _ in range(len(TruePaths))]
    Y_col_names = ['yaxis' for _ in range(len(TruePaths))]
    FileSizes = [os.path.getsize(os.path.realpath(f)) for f in TruePaths]
    df['TruePath'] = TruePaths
    df['barcode_col_name'] = barcode_col_names
    df['X_col_name'] = X_col_names
    df['Y_col_name'] = Y_col_names
    df['spot_size'] = spot_sizes
    df['FileSize'] = FileSizes
    df['original_height'] = original_heights
    df['original_width'] = original_widths
    df['slide_id'] = slide_ids
    df['DX_filename'] = DX_filenames


    spatial_dir = '/data/Jiang_Lab/Data/Zisha_Zhong'
    df1 = pd.read_excel('/data/zhongz2/ST_256/all_20231117.xlsx')
    # remove He_2020
    invalid_inds = [rowid for rowid, row in df1.iterrows() if 'He_2020' == row['slide_id'][:7]]
    if len(invalid_inds):
        df1 = df1.drop(index=invalid_inds).reset_index(drop=True)

    spot_sizes = []
    original_widths = []
    original_heights = []
    DX_filenames = []
    for _, row in df1.iterrows():
        svs_filename = row['TruePath']
        svs_prefix = row['slide_id']
        if 'He_2020' == svs_prefix[:7]:
            continue
        save_path_hdf5 = os.path.join(patch_save_dir, svs_prefix+'.h5')
        DX_filename = os.path.join(svs_save_dir, svs_prefix+'.svs')
        DX_filenames.append(DX_filename)
        if '10x_' == svs_prefix[:4]:
            with open('{}/dataset_10x/data/{}/spatial/scalefactors_json.json'.format(spatial_dir, svs_prefix.replace('10x_', '')), 'r') as fp:
                spot_size = float(json.load(fp)['spot_diameter_fullres'])
        elif 'TNBC_' == svs_prefix[:5]:
            with open('{}/hk_TNBC_ST/TNBC_data/{}/spatial/scalefactors_json.json'.format(spatial_dir, svs_prefix), 'r') as fp:
                spot_size = float(json.load(fp)['spot_diameter_fullres'])
        elif 'SCLC_' == svs_prefix[:5]:
            with open('{}/ST_SCLC/{}/spatial/scalefactors_json.json'.format(spatial_dir, svs_prefix), 'r') as fp:
                spot_size = float(json.load(fp)['spot_diameter_fullres'])
        elif 'He_2020' == svs_prefix[:7]:
            spot_size = 72.45312335405033 * 2  # from ST1k4m data, radius there, diameter here
        else:
            print(row)
            raise ValueError("wrong spot_size")
        radius = spot_size/2
        spot_sizes.append(spot_size)

        slide = openslide.open_slide(svs_filename)
        original_width, original_height = slide.level_dimensions[0]
        original_widths.append(original_width)
        original_heights.append(original_height)

        thumbnail = slide.get_thumbnail([512, 512])
        thumbnail.save(os.path.join(thumbnail_save_dir, svs_prefix+'.jpg'))

        os.system('ln -sf "{}" "{}"'.format(svs_filename, DX_filename))

    df1['original_height'] = original_heights
    df1['original_width'] = original_widths
    df1['DX_filename'] = DX_filenames
    df1['spot_size'] = spot_sizes

    # check identical files
    identical_files = {}
    for _, row in df1.iterrows():
        p1, size1, h1, w1 = row['slide_id'], row['FileSize'], row['original_height'], row['original_width']
        for _, row1 in df.iterrows():
            p2, size2, h2, w2 = row1['slide'], row1['FileSize'], row1['original_height'], row1['original_width']
            if size1 == size2 and h1 == h2 and w1 == w2: 
                if p1 in identical_files:
                    identical_files[p1].append(p2)
                else:
                    identical_files[p1] = [p2]




def debug():
    root = '/data/Jiang_Lab/Data/STimage-1K4M/'
    df = pd.read_csv(f'{root}/meta/meta_all_gene_fixed.csv')

    save_root = '/data/zhongz2/ST_20240903'   # combine previous ST and ST1K4M, removing duplicates
    for d in ['svs', 'patches', 'thumbnails']:
        os.makedirs(os.path.join(save_root, d), exist_ok=True)
    svs_save_dir = os.path.join(save_root, 'svs')
    patch_save_dir = os.path.join(save_root, 'patches')
    thumbnail_save_dir = os.path.join(save_root, 'thumbnails')

    TruePaths = []
    counts_filenames = []
    coord_filenames = []
    spot_sizes = []
    original_widths = []
    original_heights = []
    DX_filenames = []
    slide_ids = []
    for _, row in df.iterrows():
        svs_prefix = 'ST1K4M_' + clear_prefix(row['slide'])
        slide_ids.append(svs_prefix)
        save_path_hdf5 = os.path.join(patch_save_dir, svs_prefix+'.h5')
        DX_filename = os.path.join(svs_save_dir, svs_prefix+'.svs')
        DX_filenames.append(DX_filename)
        svs_filename = '{}/{}/image/{}.png'.format(root, row['tech'], row['slide'])
        counts_filename = '{}/{}/gene_exp/{}_count.csv'.format(root, row['tech'], row['slide'])
        coord_filename = '{}/{}/coord/{}_coord.csv'.format(root, row['tech'], row['slide'])
        exist0 = os.path.exists(svs_filename) and os.path.exists(counts_filename) and os.path.exists(coord_filename)
        if not exist0:
            print(row)
            import pdb
            pdb.set_trace()
        TruePaths.append(svs_filename)
        counts_filenames.append(counts_filename)
        coord_filenames.append(coord_filename)
        # barcode_col_name, X_col_name, Y_col_name

        slide = openslide.open_slide(svs_filename)
        original_width, original_height = slide.level_dimensions[0]
        original_widths.append(original_width)
        original_heights.append(original_height)

        thumbnail = slide.get_thumbnail([512, 512])
        thumbnail.save(os.path.join(thumbnail_save_dir, svs_prefix+'.jpg'))

        if True:
            with pd.read_csv(counts_filename, chunksize=20) as reader:
                for chunk in reader:
                    print(chunk)
                    break
        coord_df = pd.read_csv(coord_filename)
        radius = float(coord_df['r'].values[0])  # radius
        spot_sizes.append(2*radius)

        patch_size = np.ceil(2 * 1.1 * radius) # expand some area (10% here)

        barcode_col_name = 'Unnamed: 0'
        X_col_name = 'xaxis'
        Y_col_name = 'yaxis'
        barcodes = coord_df[barcode_col_name].values.tolist()
        Y = coord_df[Y_col_name].values.tolist()
        X = coord_df[X_col_name].values.tolist()

        level_downsamples = []
        dim_0 = slide.level_dimensions[0]

        for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))
        level_dim = slide.level_dimensions

        patch_level = 0
        results = np.array([X,Y]).T.astype(np.int32)
        results[:, 0] -= patch_size//2
        results[:, 1] -= patch_size//2
        results = results.astype(np.int32)
        asset_dict = {'coords': results}

        attr = {'patch_size': patch_size,  # To be considered...
                'patch_level': patch_level,
                'downsample': level_downsamples[patch_level],
                'downsampled_level_dim': tuple(np.array(level_dim[patch_level])),
                'level_dim': level_dim[patch_level],
                'name': svs_prefix,
                'save_path': patch_save_dir}

        attr_dict = {'coords': attr}
        save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')

        os.system('ln -sf "{}" "{}"'.format(svs_filename, DX_filename))

    barcode_col_names = ['Unnamed: 0' for _ in range(len(TruePaths))]
    X_col_names = ['xaxis' for _ in range(len(TruePaths))]
    Y_col_names = ['yaxis' for _ in range(len(TruePaths))]
    FileSizes = [os.path.getsize(os.path.realpath(f)) for f in TruePaths]
    df['TruePath'] = TruePaths
    df['barcode_col_name'] = barcode_col_names
    df['X_col_name'] = X_col_names
    df['Y_col_name'] = Y_col_names
    df['spot_size'] = spot_sizes
    df['FileSize'] = FileSizes
    df['original_height'] = original_heights
    df['original_width'] = original_widths
    df['slide_id'] = slide_ids
    df['DX_filename'] = DX_filenames


    spatial_dir = '/data/Jiang_Lab/Data/Zisha_Zhong'
    df1 = pd.read_excel('/data/zhongz2/ST_256/all_20231117.xlsx')
    spot_sizes = []
    original_widths = []
    original_heights = []
    DX_filenames
    for _, row in df1.iterrows():
        svs_filename = row['TruePath']
        svs_prefix = row['slide_id']
        save_path_hdf5 = os.path.join(patch_save_dir, svs_prefix+'.h5')
        DX_filename = os.path.join(svs_save_dir, svs_prefix+'.svs')
        DX_filenames.append(DX_filename)
        if '10x_' == svs_prefix[:4]:
            with open('{}/dataset_10x/data/{}/spatial/scalefactors_json.json'.format(spatial_dir, svs_prefix.replace('10x_', '')), 'r') as fp:
                spot_size = float(json.load(fp)['spot_diameter_fullres'])
        elif 'TNBC_' == svs_prefix[:5]:
            with open('{}/hk_TNBC_ST/TNBC_data/{}/spatial/scalefactors_json.json'.format(spatial_dir, svs_prefix), 'r') as fp:
                spot_size = float(json.load(fp)['spot_diameter_fullres'])
        elif 'SCLC_' == svs_prefix[:5]:
            with open('{}/ST_SCLC/{}/spatial/scalefactors_json.json'.format(spatial_dir, svs_prefix), 'r') as fp:
                spot_size = float(json.load(fp)['spot_diameter_fullres'])
        elif 'He_2020' == svs_prefix[:7]:
            spot_size = 72.45312335405033 * 2  # from ST1k4m data, radius there, diameter here
        else:
            print(row)
            raise ValueError("wrong spot_size")
        radius = spot_size/2
        spot_sizes.append(spot_size)
        patch_size = np.ceil(2 * 1.1 * radius) # expand some area (10% here)

        coord_df = pd.read_csv(coord_filename)

        slide = openslide.open_slide(svs_filename)
        original_width, original_height = slide.level_dimensions[0]
        original_widths.append(original_width)
        original_heights.append(original_height)

        thumbnail = slide.get_thumbnail([512, 512])
        thumbnail.save(os.path.join(thumbnail_save_dir, svs_prefix+'.jpg'))

        barcode_col_name = row['barcode_col_name']
        Y_col_name = row['Y_col_name']
        X_col_name = row['X_col_name']
        barcodes = coord_df[barcode_col_name].values.tolist()
        Y = coord_df[Y_col_name].values.tolist()
        X = coord_df[X_col_name].values.tolist()

        level_downsamples = []
        dim_0 = slide.level_dimensions[0]

        for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))
        level_dim = slide.level_dimensions

        patch_level = 0
        results = np.array([X,Y]).T.astype(np.int32)
        results[:, 0] -= patch_size//2
        results[:, 1] -= patch_size//2
        results = results.astype(np.int32)
        results1 = np.copy(results)
        results1[:, 0] += patch_size # width
        results1[:, 1] += patch_size # height
        invalid_inds = np.concatenate([
            np.where((results[:,0]<=0)|(results[:,1]<=0))[0],
            np.where((results1[:,0]>=original_width-1)|(results1[:,1]>=original_height-1))[0]
        ])
        results = np.delete(results, np.unique(inds), axis=0)
        del results1
        asset_dict = {'coords': results}

        attr = {'patch_size': patch_size,  # To be considered...
                'patch_level': patch_level,
                'downsample': level_downsamples[patch_level],
                'downsampled_level_dim': tuple(np.array(level_dim[patch_level])),
                'level_dim': level_dim[patch_level],
                'name': svs_prefix,
                'save_path': patch_save_dir}

        attr_dict = {'coords': attr}
        save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')

        os.system('ln -sf "{}" "{}"'.format(svs_filename, DX_filename))

    df1['original_height'] = original_heights
    df1['original_width'] = original_widths
    df1['DX_filename'] = DX_filenames
    df['spot_size'] = spot_sizes

    # check identical files
    identical_files = {}
    for _, row in df1.iterrows():
        p1, size1, h1, w1 = row['slide_id'], row['FileSize'], row['original_height'], row['original_width']
        for _, row1 in df.iterrows():
            p2, size2, h2, w2 = row1['slide'], row1['FileSize'], row1['original_height'], row1['original_width']
            if size1 == size2 and h1 == h2 and w1 == w2: 
                if p1 in identical_files:
                    identical_files[p1].append(p2)
                else:
                    identical_files[p1] = [p2]












