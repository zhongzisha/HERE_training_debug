

import sys,os
import numpy as np
import openslide
import pandas as pd
import json
import glob
import gzip
import scanpy
import time
from utils import save_hdf5
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def generat_st_all():
    df = pd.read_excel('/data/zhongz2/ST_256/all_20231117.xlsx')
    project_name = 'ST_SPOT_256'  # using spot_size for patch extraction
    root_dir = f'/data/zhongz2/{project_name}'
    patches_dir = f'{root_dir}/patches'
    svs_dir = f'{root_dir}/svs'
    os.makedirs(svs_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    save_root4 = patches_dir


    params = {
        '10x': [20, 0.25],
        'SCLC': [40, 0.31],
        'TNBC': [10, 0.65],
        'He_2020': [40, 0.35],
    }

    TNBC_dict = {}
    for p in list(set([os.path.basename(f).replace('_tissue_positions_list.csv.gz', '')
                         for f in glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/*_tissue_positions_list.csv.gz')])):
        a1, a2 = p.split('_')
        TNBC_dict[a2] = a1

    dirs = glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/ST_SCLC/Sample_6233*')
    dirs = [os.path.basename(d) for d in dirs]
    SCLC_dict = {k.split('_')[1]: k for k in dirs}


    for ind, row in df.iterrows():

        f = row['TruePath'].split(' ')[0]
        slide_name = row['slide_id']
        key = None
        for key in list(params.keys()):
            if key in f:
                break
        if key is None:
            continue
        h5filename = os.path.join(save_root4, slide_name+'.h5')
        if os.path.exists(h5filename):
            continue
        
        slide = openslide.open_slide(f)
        
        os.system(f'ln -sf "{f}" "{svs_dir}/{slide_name}.svs"')
        time.sleep(1)

        if 'openslide.mpp-x' in slide.properties:
            mpp = float(slide.properties['openslide.mpp-x'])
        else:
            mpp = params[key][1]
        spot_size = None
        st_patch_size = int(pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))

        if key == 'He_2020':
            prefix = f.split('/')[-2]
            coord_filename = os.path.dirname(f) + '/spot_coordinates.csv'
            coord_df = pd.read_csv(coord_filename)
            counts_filename = os.path.dirname(f) + '/counts.tsv'
            # counts_df = pd.read_csv(os.path.dirname(f) + '/counts.tsv', sep='\t')
            spot_size = None # st_patch_size
            barcode_col_name = 'Unnamed: 0'
            X_col_name = 'X'
            Y_col_name = 'Y'


        elif 'SCLC' in key:
            prefix = os.path.basename(f).replace('.tif', '').split('_')[1]

            coord_filename = '/data/Jiang_Lab/Data/Zisha_Zhong/ST_SCLC/{}/spatial/tissue_positions_list.csv'.format(
                SCLC_dict[prefix]
            )
            coord_df = pd.read_csv(coord_filename, header=None)

            filename1 = '/data/Jiang_Lab/Data/Zisha_Zhong/ST_SCLC/{}/spatial/scalefactors_json.json'.format(
                SCLC_dict[prefix]
            )
            with open(filename1, 'r') as fp:
                json_str = fp.read()
            json_data = json.loads(json_str)
            spot_size = float(json_data['spot_diameter_fullres'])

            filename2 = '/data/Jiang_Lab/Data/Zisha_Zhong/ST_SCLC/{}/filtered_feature_bc_matrix.h5'.format(
                SCLC_dict[prefix]
            )
            counts_filename = filename2
            # counts_df = scanpy.read_10x_h5(filename2).to_df().T

            barcode_col_name = 0
            X_col_name = 5
            Y_col_name = 4

        elif 'TNBC' in key:
            prefix = os.path.basename(f).replace('.tif', '').replace('_','')
            if prefix not in TNBC_dict:
                continue

            coord_filename = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/{}_{}_tissue_positions_list.csv.gz'.format(
                TNBC_dict[prefix], prefix
            )
            coord_df = pd.read_csv(coord_filename, header=None)

            filename1 = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/{}_{}_scalefactors_json.json.gz'.format(
                TNBC_dict[prefix], prefix
            )
            with gzip.open(filename1, 'r') as fp:
                json_bytes = fp.read()
            json_str = json_bytes.decode('utf-8')
            json_data = json.loads(json_str)
            spot_size = float(json_data['spot_diameter_fullres'])

            filename2 = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data/{}_{}_filtered_feature_bc_matrix.h5'.format(
                TNBC_dict[prefix], prefix
            )
            counts_filename = filename2
            # counts_df = scanpy.read_10x_h5(filename2).to_df().T

            barcode_col_name = 0
            X_col_name = 5
            Y_col_name = 4

        elif key == '10x':
            spatial_root = '/data/Jiang_Lab/Data/Zisha_Zhong/dataset_10x/data/'
            dirname = os.path.dirname(f).split('/')[-1]
            prefix = dirname

            d = os.path.join(spatial_root, dirname)
            coord_filename = d + '/spatial/tissue_positions.csv'
            if not os.path.exists(coord_filename):
                coord_filename = d + '/spatial/tissue_positions_list.csv'
                coord_df = pd.read_csv(coord_filename, header=None)
                barcode_col_name = 0
                Y_col_name = 4
                X_col_name = 5
                in_tissue_col_name = 1
            else:
                coord_df = pd.read_csv(coord_filename)
                barcode_col_name = 'barcode'
                Y_col_name = 'pxl_row_in_fullres'
                X_col_name = 'pxl_col_in_fullres'
                in_tissue_col_name = 'in_tissue'

            filename1 = d + '/spatial/scalefactors_json.json'
            with open(filename1, 'r') as fp:
                json_str = fp.read()
            json_data = json.loads(json_str)
            spot_size = float(json_data['spot_diameter_fullres'])

            files2 = glob.glob(d + '/*_filtered_feature_bc_matrix.h5')
            filename2 = files2[0]
            counts_filename = filename2
            # counts_df = scanpy.read_10x_h5(filename2).to_df().T

        else:
            prefix = ''
            coord_df = None
            counts_df = None
            coord_filename = None
            counts_filename = None
            barcode_col_name = None
            Y_col_name = None
            X_col_name = None
            raise ValueError("error")
            # pdb.set_trace()
            print('fatal error')


        barcodes = coord_df[barcode_col_name].values.tolist()
        Y = coord_df[Y_col_name].values.tolist()
        X = coord_df[X_col_name].values.tolist()


        patch_level = 0
        # patch_size = int(pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))
        if spot_size is not None:
            patch_size = int(np.ceil(spot_size/48) * 48)
        else:
            patch_size = st_patch_size

        all_coords = np.array([X, Y]).T
        all_coords[:, 0] -= patch_size // 2
        all_coords[:, 1] -= patch_size // 2
        all_coords = all_coords.astype(np.int32)

        level_downsamples = []
        dim_0 = slide.level_dimensions[0]

        for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))
        level_dim = slide.level_dimensions

        
        asset_dict = {'coords': all_coords}
        attr = {'patch_size': patch_size,  # To be considered...
                'patch_level': patch_level,
                'downsample': level_downsamples[patch_level],
                'downsampled_level_dim': tuple(np.array(level_dim[patch_level])),
                'level_dim': level_dim[patch_level],
                'name': slide_name,
                'save_path': save_root4}

        attr_dict = {'coords': attr}

        save_hdf5(h5filename, asset_dict, attr_dict, mode='w')

        print(slide_name, len(all_coords), len(coord_df), slide.dimensions)
        print(coord_df.head())





def generat_st_all2():
    df = pd.read_excel('/data/zhongz2/ST_256/all_20231117.xlsx')
    project_name = 'ST_SPOT_256'  # using spot_size for patch extraction
    root_dir = f'/data/zhongz2/{project_name}'
    patches_dir = f'{root_dir}/patches'
    svs_dir = f'{root_dir}/svs'
    os.makedirs(svs_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    save_root4 = patches_dir


    params = {
        '10x': [20, 0.25],
        'SCLC': [40, 0.31],
        'TNBC': [10, 0.65],
        'He_2020': [40, 0.35],
    }

    TNBC_dict = {}
    for p in list(set([os.path.basename(f).replace('_tissue_positions_list.csv.gz', '')
                         for f in glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/*_tissue_positions_list.csv.gz')])):
        a1, a2 = p.split('_')
        TNBC_dict[a2] = a1

    dirs = glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/ST_SCLC/Sample_6233*')
    dirs = [os.path.basename(d) for d in dirs]
    SCLC_dict = {k.split('_')[1]: k for k in dirs}


    for ind, row in df.iterrows():

        f = row['TruePath'].split(' ')[0]
        slide_name = row['slide_id']
        key = None
        for key in list(params.keys()):
            if key in f:
                break
        if key is None:
            continue
        h5filename = os.path.join(save_root4, slide_name+'.h5')
        # if os.path.exists(h5filename):
        #     continue
        
        slide = openslide.open_slide(f)
        
        # os.system(f'ln -sf "{f}" "{svs_dir}/{slide_name}.svs"')
        # time.sleep(1)

        if 'openslide.mpp-x' in slide.properties:
            mpp = float(slide.properties['openslide.mpp-x'])
        else:
            mpp = params[key][1]
        spot_size = None
        st_patch_size = int(pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))

        if key == 'He_2020':
            prefix = f.split('/')[-2]
            coord_filename = os.path.dirname(f) + '/spot_coordinates.csv'
            coord_df = pd.read_csv(coord_filename)
            counts_filename = os.path.dirname(f) + '/counts.tsv'
            # counts_df = pd.read_csv(os.path.dirname(f) + '/counts.tsv', sep='\t')
            spot_size = None # st_patch_size
            barcode_col_name = 'Unnamed: 0'
            X_col_name = 'X'
            Y_col_name = 'Y'


        elif 'SCLC' in key:
            prefix = os.path.basename(f).replace('.tif', '').split('_')[1]

            coord_filename = '/data/Jiang_Lab/Data/Zisha_Zhong/ST_SCLC/{}/spatial/tissue_positions_list.csv'.format(
                SCLC_dict[prefix]
            )
            coord_df = pd.read_csv(coord_filename, header=None)

            filename1 = '/data/Jiang_Lab/Data/Zisha_Zhong/ST_SCLC/{}/spatial/scalefactors_json.json'.format(
                SCLC_dict[prefix]
            )
            with open(filename1, 'r') as fp:
                json_str = fp.read()
            json_data = json.loads(json_str)
            spot_size = float(json_data['spot_diameter_fullres'])

            filename2 = '/data/Jiang_Lab/Data/Zisha_Zhong/ST_SCLC/{}/filtered_feature_bc_matrix.h5'.format(
                SCLC_dict[prefix]
            )
            counts_filename = filename2
            # counts_df = scanpy.read_10x_h5(filename2).to_df().T

            barcode_col_name = 0
            X_col_name = 5
            Y_col_name = 4

        elif 'TNBC' in key:
            prefix = os.path.basename(f).replace('.tif', '').replace('_','')
            if prefix not in TNBC_dict:
                continue

            coord_filename = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/{}_{}_tissue_positions_list.csv.gz'.format(
                TNBC_dict[prefix], prefix
            )
            coord_df = pd.read_csv(coord_filename, header=None)

            filename1 = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/{}_{}_scalefactors_json.json.gz'.format(
                TNBC_dict[prefix], prefix
            )
            with gzip.open(filename1, 'r') as fp:
                json_bytes = fp.read()
            json_str = json_bytes.decode('utf-8')
            json_data = json.loads(json_str)
            spot_size = float(json_data['spot_diameter_fullres'])

            filename2 = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data/{}_{}_filtered_feature_bc_matrix.h5'.format(
                TNBC_dict[prefix], prefix
            )
            counts_filename = filename2
            # counts_df = scanpy.read_10x_h5(filename2).to_df().T

            barcode_col_name = 0
            X_col_name = 5
            Y_col_name = 4

        elif key == '10x':
            spatial_root = '/data/Jiang_Lab/Data/Zisha_Zhong/dataset_10x/data/'
            dirname = os.path.dirname(f).split('/')[-1]
            prefix = dirname

            d = os.path.join(spatial_root, dirname)
            coord_filename = d + '/spatial/tissue_positions.csv'
            if not os.path.exists(coord_filename):
                coord_filename = d + '/spatial/tissue_positions_list.csv'
                coord_df = pd.read_csv(coord_filename, header=None)
                barcode_col_name = 0
                Y_col_name = 4
                X_col_name = 5
                in_tissue_col_name = 1
            else:
                coord_df = pd.read_csv(coord_filename)
                barcode_col_name = 'barcode'
                Y_col_name = 'pxl_row_in_fullres'
                X_col_name = 'pxl_col_in_fullres'
                in_tissue_col_name = 'in_tissue'

            filename1 = d + '/spatial/scalefactors_json.json'
            with open(filename1, 'r') as fp:
                json_str = fp.read()
            json_data = json.loads(json_str)
            spot_size = float(json_data['spot_diameter_fullres'])

            files2 = glob.glob(d + '/*_filtered_feature_bc_matrix.h5')
            filename2 = files2[0]
            counts_filename = filename2
            # counts_df = scanpy.read_10x_h5(filename2).to_df().T

        else:
            prefix = ''
            coord_df = None
            counts_df = None
            coord_filename = None
            counts_filename = None
            barcode_col_name = None
            Y_col_name = None
            X_col_name = None
            raise ValueError("error")
            # pdb.set_trace()
            print('fatal error')


        a = barcode_col_name == row['barcode_col_name']
        b = X_col_name == row['X_col_name']
        c = Y_col_name == row['Y_col_name']

        print(slide_name, a, b, c)
        if not np.all([a, b, c]):
            raise ValueError("error")

        # barcodes = coord_df[barcode_col_name].values.tolist()
        # Y = coord_df[Y_col_name].values.tolist()
        # X = coord_df[X_col_name].values.tolist()


        # patch_level = 0
        # # patch_size = int(pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))
        # if spot_size is not None:
        #     patch_size = int(np.ceil(spot_size/48) * 48)
        # else:
        #     patch_size = st_patch_size

        # all_coords = np.array([X, Y]).T
        # all_coords[:, 0] -= patch_size // 2
        # all_coords[:, 1] -= patch_size // 2
        # all_coords = all_coords.astype(np.int32)

        # level_downsamples = []
        # dim_0 = slide.level_dimensions[0]

        # for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
        #     estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        #     level_downsamples.append(estimated_downsample) if estimated_downsample != (
        #         downsample, downsample) else level_downsamples.append((downsample, downsample))
        # level_dim = slide.level_dimensions

        
        # asset_dict = {'coords': all_coords}
        # attr = {'patch_size': patch_size,  # To be considered...
        #         'patch_level': patch_level,
        #         'downsample': level_downsamples[patch_level],
        #         'downsampled_level_dim': tuple(np.array(level_dim[patch_level])),
        #         'level_dim': level_dim[patch_level],
        #         'name': slide_name,
        #         'save_path': save_root4}

        # attr_dict = {'coords': attr}

        # save_hdf5(h5filename, asset_dict, attr_dict, mode='w')

        # print(slide_name, len(all_coords), len(coord_df), slide.dimensions)
        # print(coord_df.head())




def generate_patches():
    df = pd.read_excel('/data/zhongz2/ST_256/all_20231117.xlsx')
    spatial_dir = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data'
    project_name = 'TNBC_256'
    root_dir = f'/data/zhongz2/{project_name}'
    patches_dir = f'{root_dir}/patches'
    svs_dir = f'{root_dir}/svs'
    os.makedirs(svs_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    save_root4 = patches_dir

    TNBC_dict = {}
    for p in list(set([os.path.basename(f).replace('_tissue_positions_list.csv.gz', '')
                         for f in glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/*_tissue_positions_list.csv.gz')])):
        a1, a2 = p.split('_')
        TNBC_dict[a2] = a1

    for ind, row in df.iterrows():
        if 'TNBC' not in row['slide_id']:
            continue
        f = row['TruePath']
        slide_name = row['slide_id']
        prefix = row['slide_id'].replace('TNBC_', '')
        h5filename = os.path.join(save_root4, slide_name+'.h5')
        if os.path.exists(h5filename):
            continue
        
        slide = openslide.open_slide(f)
        
        os.system(f'ln -sf "{f}" "{svs_dir}/{slide_name}.svs"')
        time.sleep(1)

        with open('{}/{}/spatial/scalefactors_json.json'.format(spatial_dir, slide_name), 'r') as fp:
            spot_size = float(json.load(fp)['spot_diameter_fullres'])

        
        coord_filename = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/{}_{}_tissue_positions_list.csv.gz'.format(
            TNBC_dict[prefix], prefix
        )
        coord_df = pd.read_csv(coord_filename, header=None)

        if False:
            filename1 = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/coordinates/{}_{}_scalefactors_json.json.gz'.format(
                TNBC_dict[prefix], prefix
            )
            with gzip.open(filename1, 'r') as fp:
                json_bytes = fp.read()
            json_str = json_bytes.decode('utf-8')
            json_data = json.loads(json_str)

        if False:
            filename2 = '/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data/{}_{}_filtered_feature_bc_matrix.h5'.format(
                TNBC_dict[prefix], prefix
            )
            counts_filename = filename2
            counts_df = scanpy.read_10x_h5(filename2).to_df().T

        barcode_col_name = 0
        X_col_name = 5
        Y_col_name = 4

        barcodes = coord_df[barcode_col_name].values.tolist()
        Y = coord_df[Y_col_name].values.tolist()
        X = coord_df[X_col_name].values.tolist()


        patch_level = 0
        # patch_size = int(pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))
        patch_size = int(np.ceil(spot_size/48) * 48)
        all_coords = np.array([X, Y]).T
        all_coords[:, 0] -= patch_size // 2
        all_coords[:, 1] -= patch_size // 2
        all_coords = all_coords.astype(np.int32)

        level_downsamples = []
        dim_0 = slide.level_dimensions[0]

        for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))
        level_dim = slide.level_dimensions

        
        asset_dict = {'coords': all_coords}
        attr = {'patch_size': patch_size,  # To be considered...
                'patch_level': patch_level,
                'downsample': level_downsamples[patch_level],
                'downsampled_level_dim': tuple(np.array(level_dim[patch_level])),
                'level_dim': level_dim[patch_level],
                'name': slide_name,
                'save_path': save_root4}

        attr_dict = {'coords': attr}

        save_hdf5(h5filename, asset_dict, attr_dict, mode='w')

        print(slide_name, len(all_coords), len(coord_df), slide.dimensions)
        print(coord_df.head())

if __name__ == '__main__':
    # generate_patches()
    generat_st_all2()





















