
import sys,os,glob,shutil,json
import numpy as np
import pandas as pd
import openslide
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import h5py
import scanpy
from utils import save_hdf5
import idr_torch 
from matplotlib import pyplot as plt


def clear_prefix(prefix):
    prefix = prefix.replace(' ', '_')
    prefix = prefix.replace(',', '_')
    prefix = prefix.replace('&', '_')
    prefix = prefix.replace('+', '_')
    return prefix


def clean_data_merge_previous_data():  # result is all_cleaned_20240904.xlsx
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
    df['coord_filename'] = coord_filenames
    df['counts_filename'] = counts_filenames


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


    import sys,os,glob,shutil,json
    import numpy as np
    import openslide
    import pandas as pd

    from matplotlib import pyplot as plt
    import PIL
    PIL.Image.MAX_IMAGE_PIXELS = 1266016250
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    root = '/data/zhongz2/ST_20240903'
    df = pd.read_csv(f'{root}/ST1K4M.csv')
    df1 = pd.read_csv(f'{root}/PreviousST.csv')

    common_cols = set(df.columns.values).intersection(set(df1.columns.values))
    common_cols.remove('Unnamed: 0')
    common_cols = list(common_cols)
    alldf = pd.concat([df[common_cols], df1[common_cols]], axis=0)

    def get_pos(W, H):
        return [
            [W//4, H//4],   [W//2, H//4],   [3*W//4, H//4],
            [W//4, H//2],   [W//2, H//2],   [3*W//4, H//2],
            [W//4, 3*H//4], [W//2, 3*H//4], [3*W//4, 3*H//4],
        ]
    pixel_values = []
    for _, row in alldf.iterrows():
        svs_prefix = row['slide_id']
        slide = openslide.open_slide(row['DX_filename'])
        pixel_value = []
        for loc in get_pos(row['original_width'], row['original_height']):
            patch_sum = np.array(slide.read_region(location=loc, level=0, size=(5, 5))).sum()
            pixel_value.append(patch_sum)
        pixel_values.append(pixel_value)
        slide.close()
    alldf['pixel_sum'] = np.array(pixel_values).sum(axis=1)
    alldf['newcol'] = alldf.apply(lambda x: '{:d}_{:d}_{:d}_{:d}'.format(x['FileSize'], x['original_width'], x['original_height'], x['pixel_sum']), axis=1)

    save_root = '/data/zhongz2/temp29/check_st1k4m_v2'
    if os.path.isdir(save_root):
        shutil.rmtree(save_root, ignore_errors=True)
        import time
        time.sleep(1)
    os.makedirs(save_root, exist_ok=True)

    for k, v in alldf['newcol'].value_counts().to_dict().items():
        if v > 1:
            fig, axes = plt.subplots(nrows=1, ncols=v)
            for i, (_, row) in enumerate(alldf[alldf['newcol']==k].iterrows()):
                im = Image.open(os.path.join(root, 'thumbnails', row['slide_id']+'.jpg'))
                axes[i].imshow(im)
            plt.savefig(os.path.join(save_root, k+'.jpg'))
            plt.close()

    names_dict = {}    
    names_1 = []
    from natsort import natsorted  
    for k, v in alldf['newcol'].value_counts().to_dict().items():
        if v > 1:
            names_dict[k] = natsorted(alldf[alldf['newcol']==k]['slide_id'].values.tolist())
        else:
            names_1.append(k)
    
    need_to_download = [
        "ST1K4M_Human_Breast_10X_06092021_Visium", # done
        "ST1K4M_Human_Colon_10X_10052023_Visium_control_rep1", # done
        "ST1K4M_Human_Colon_10X_10052023_Visium_control_rep2",# done
        "ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep1",# done
        "ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep2",# done
        "ST1K4M_Human_Prostate_10X_06092021_Visium_cancer",# done
        "ST1K4M_Human_Prostate_10X_06092021_Visium_normal",# done
        "ST1K4M_Human_Prostate_10X_07122022_Visium",# done
        "ST1K4M_Mouse_Brain_Lung_10X_02212023_Visium_2mm_edge", # done
        "ST1K4M_Mouse_Brain_10X_08162021_Visium",
        "ST1K4M_Mouse_Brain_10X_10052023_Visium_control_rep1",
        "ST1K4M_Mouse_Brain_10X_10052023_Visium_control_rep2",
        "ST1K4M_Mouse_Brain_10X_10052023_Visium_post_xenium_rep1",
        "ST1K4M_Mouse_Brain_10X_10052023_Visium_post_xenium_rep2",
        "ST1K4M_Mouse_Kidney_10X_08162021_Visium", # done
        "ST1K4M_Human_Colon_10X_03252024_VisiumHD", # done
        "ST1K4M_Human_Lung_10X_03292024_VisiumHD", #done
        "ST1K4M_Mouse_Brain_10X_03292024_VIsiumHD", #done
        "ST1K4M_Mouse_Intenstine_10X_03252024_VisiumHD" #done
    ]


    max_len = max([len(v) for v in names_dict.values()])
    for k, v in names_dict.items():
        if len(v) != max_len:
            names_dict[k] = v+['' for _ in range(max_len - len(v))]
    names_df = pd.DataFrame(names_dict).T

    names_df = pd.read_excel(f'{root}/names.xlsx', index_col=0)
    names_df['first_non_nan'] = names_df.apply(lambda row: row[row.first_valid_index()] if row.first_valid_index() is not None else np.nan, axis=1)

    valid_newcol_values = [k for k, v in alldf['newcol'].value_counts().to_dict().items() if v==1]
    valid_slide_ids = alldf[alldf['newcol'].isin(valid_newcol_values)]['slide_id'].values.tolist() + names_df['first_non_nan'].values.tolist()
    alldf1 = alldf[alldf['slide_id'].isin(valid_slide_ids)].reset_index(drop=True)
    alldf1 = alldf1.drop(columns=['pixel_sum', 'newcol'])

    coord_filenames = []
    counts_filenames = []
    for _, row in alldf1.iterrows():
        svs_prefix = row['slide_id']
        if 'ST1K4M' == svs_prefix[:6]:
            coord_filenames.append(row['TruePath'].replace('/image/', '/coord/').replace('.png', '_coord.csv'))
            counts_filenames.append(row['TruePath'].replace('/image/', '/gene_exp/').replace('.png', '_count.csv'))
        else:
            df2 = df1[df1['slide_id']==svs_prefix]
            coord_filenames.append(df2['coord_filename'].values[0])
            counts_filenames.append(df2['counts_filename'].values[0])
    alldf1['coord_filename'] = coord_filenames
    alldf1['counts_filename'] = counts_filenames

    for _, row in alldf1.iterrows():
        existed = []
        for name in ['TruePath', 'DX_filename', 'coord_filename', 'counts_filename']:
            existed.append(os.path.exists(row[name]))
        if not np.all(existed):
            print(row)
    sorted_columns = [
        'slide_id', 'original_width', 'original_height', 'spot_size', 'barcode_col_name', 'X_col_name', 'Y_col_name', 'DX_filename', 'TruePath',
        'coord_filename', 'counts_filename', 'FileSize'
    ]

    # check identical files
    identical_files = {}
    for _, row in df1.iterrows():
        p1, size1, h1, w1 = row['slide_id'], row['FileSize'], row['original_height'], row['original_width']
        slide1 = openslide.open_slide(os.path.join(root, 'svs', p1+'.svs'))
        for _, row1 in df.iterrows():
            p2, size2, h2, w2 = row1['slide'], row1['FileSize'], row1['original_height'], row1['original_width']
            if size1 == size2 and h1 == h2 and w1 == w2: 
                # randomly read 5 times
                slide2 = openslide.open_slide(os.path.join(root, 'svs', 'ST1K4M_'+clear_prefix(p2)+'.svs'))
                is_the_same = [False for _ in range(5)]
                for i in range(5):
                    x = np.random.randint(20, w1 - 20)
                    y = np.random.randint(20, h1 - 20) 
                    patch1 = np.array(slide1.read_region(location=(x, y), level=0, size=(10, 10)))
                    patch2 = np.array(slide2.read_region(location=(x, y), level=0, size=(10, 10)))
                    if np.all(patch1==patch2):
                        is_the_same[i] = True
                if np.all(is_the_same):
                    if p1 in identical_files:
                        identical_files[p1].append(p2)
                    else:
                        identical_files[p1] = [p2]

    save_root = '/data/zhongz2/temp29/check_st1k4m'
    if os.path.isdir(save_root):
        shutil.rmtree(save_root, ignore_errors=True)
        
    os.makedirs(save_root, exist_ok=True)
    for k, v in identical_files.items():
        num_ims = 1 + len(v)
        plt.close()
        fig, axes = plt.subplots(nrows=1, ncols=num_ims)
        im = Image.open(os.path.join(root, 'thumbnails', k+'.jpg'))
        i = 0
        axes[i].imshow(im)
        for ii, prefix in enumerate(v):
            im = Image.open(os.path.join(root, 'thumbnails', 'ST1K4M_' +clear_prefix(prefix)+'.jpg'))
            axes[ii + 1].imshow(im)
        plt.savefig(os.path.join(save_root, k+'.jpg'))
        plt.close()

    # after check these thumbnail images, manually find out the image that need to remove out
    duplicated_prefixes = [
        '10x_V1_Mouse_Brain_Sagittal_Anterior_Section_2_1.1.0', 'Mouse_Brain_10X_06232020_Visium_Sagittal_Anterior_Section_2',
        '10x_V1_Adult_Mouse_Brain_2.1.0', 'Mouse_Brain_10X_06232020_Visium_Coronal',
        '10x_V1_Adult_Mouse_Brain_1.1.0', 'Mouse_Brain_10X_06232020_Visium_Coronal',
        '10x_V1_Mouse_Brain_Sagittal_Posterior_Section_2_1.0.0', 'Mouse_Brain_10X_06232020_Visium_Sagittal_Posterior_Section_2',
        '10x_Targeted_Visium_Human_Cerebellum_Neuroscience_2.0.0', 'Human_Brain_10X_10272020_Visium_Cerebellum_WholeTranscriptome',
        '10x_Parent_Visium_Human_Cerebellum_1.2.0', 'Human_Brain_10X_10272020_Visium_Cerebellum_WholeTranscriptome',
        '10x_V1_Mouse_Brain_Sagittal_Anterior_Section_2_2.0.0', 'Mouse_Brain_10X_06232020_Visium_Sagittal_Anterior_Section_2',
        '10x_CytAssist_11mm_FFPE_Mouse_Embryo_2.0.0', 'Mouse_Embryo_10X_07142022_Visium', 
        '10x_V1_Adult_Mouse_Brain_1.0.0', 'Mouse_Brain_10X_06232020_Visium_Coronal',
        '10x_V1_Mouse_Brain_Sagittal_Posterior_Section_2_2.0.0', 'Mouse_Brain_10X_06232020_Visium_Sagittal_Posterior_Section_2',
        '10x_Targeted_Visium_Human_Cerebellum_Neuroscience_1.2.0', 'Human_Brain_10X_10272020_Visium_Cerebellum_WholeTranscriptome',
        '10x_V1_Mouse_Brain_Sagittal_Anterior_Section_2_1.0.0', 'Mouse_Brain_10X_06232020_Visium_Sagittal_Anterior_Section_2',
        '10x_V1_Mouse_Brain_Sagittal_Posterior_Section_2_1.1.0', 'Mouse_Brain_10X_06232020_Visium_Sagittal_Posterior_Section_2',
        '10x_CytAssist_11mm_FFPE_Mouse_Embryo_2.1.0', 'Mouse_Embryo_10X_07142022_Visium'
    ]




    import sys,os,glob,shutil,json
    import numpy as np
    import openslide
    import pandas as pd

    from matplotlib import pyplot as plt
    import PIL
    PIL.Image.MAX_IMAGE_PIXELS = 1266016250
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    root = '/data/zhongz2/ST_20240903'
    df = pd.read_excel(f'{root}/cleaned1.xlsx')
    # check identical files
    identical_files = {}
    for i in range(len(df)):
        row = df.iloc[i]
        p1, size1, h1, w1 = row['slide_id'], row['FileSize'], row['original_height'], row['original_width']
        slide1 = openslide.open_slide(os.path.join(root, 'svs', p1+'.svs'))
        for j in range(i+1, len(df)):
            row1 = df.iloc[j]
            p2, size2, h2, w2 = row1['slide_id'], row1['FileSize'], row1['original_height'], row1['original_width']
            if size1 == size2 and h1 == h2 and w1 == w2: 
                # randomly read 5 times
                slide2 = openslide.open_slide(os.path.join(root, 'svs', p2+'.svs'))
                is_the_same = [False for _ in range(5)]
                for i in range(5):
                    x = np.random.randint(20, w1 - 20)
                    y = np.random.randint(20, h1 - 20) 
                    patch1 = np.array(slide1.read_region(location=(x, y), level=0, size=(10, 10)))
                    patch2 = np.array(slide2.read_region(location=(x, y), level=0, size=(10, 10)))
                    if np.all(patch1==patch2):
                        is_the_same[i] = True
                if np.all(is_the_same):
                    if p1 in identical_files:
                        identical_files[p1].append(p2)
                    else:
                        identical_files[p1] = [p2]

    save_root = '/data/zhongz2/temp29/check_st1k4m_v1'
    if os.path.isdir(save_root):
        shutil.rmtree(save_root, ignore_errors=True)
        
    os.makedirs(save_root, exist_ok=True)
    for k, v in identical_files.items():
        num_ims = 1 + len(v)
        plt.close()
        fig, axes = plt.subplots(nrows=1, ncols=num_ims)
        im = Image.open(os.path.join(root, 'thumbnails', k+'.jpg'))
        i = 0
        axes[i].imshow(im)
        for ii, prefix in enumerate(v):
            im = Image.open(os.path.join(root, 'thumbnails', prefix+'.jpg'))
            axes[ii + 1].imshow(im)
        plt.savefig(os.path.join(save_root, k+'.jpg'))
        plt.close()


    duplicated_prefixes = [
        'ST1K4M_GSE245097_GSM7836298', 'ST1K4M_GSE245097_GSM7836299', 'ST1K4M_GSE245097_GSM7836300',
        'ST1K4M_GSE245097_GSM7836299', 'ST1K4M_GSE245097_GSM7836300'
    ]


def gen_coords():

    root = '/data/zhongz2/ST_20240903'
    svs_save_dir = os.path.join(root, 'svs')
    patch_save_dir = os.path.join(root, 'patches')
    thumbnail_save_dir = os.path.join(root, 'thumbnails')
    save_root = f'{root}/spot_figures/'
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(patch_save_dir, exist_ok=True)
    os.makedirs(thumbnail_save_dir, exist_ok=True)
    os.makedirs(svs_save_dir, exist_ok=True)

    # os.makedirs(save_root, exist_ok=True)
    df = pd.read_excel(f'{root}/all_cleaned_20240904.xlsx')

    existed_prefixes = [os.path.basename(f).replace('.jpg', '') for f in glob.glob(os.path.join(save_root, '*.jpg'))]
    df = df[~df['slide_id'].isin(existed_prefixes)].reset_index(drop=True)

    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size) 
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)
    cmap = plt.get_cmap("tab10")
    colors = (np.array(cmap.colors)*255).astype(np.uint8)

    for rowid, row in sub_df.iterrows():
        svs_prefix = row['slide_id']
        if os.path.exists(os.path.join(save_root, f'{svs_prefix}.jpg')):
            continue
        print('begin ', rowid, svs_prefix)
        save_path_hdf5 = os.path.join(patch_save_dir, svs_prefix+'.h5')
        spot_size = row['spot_size']
        patch_size = int(np.ceil(1.1 * spot_size)) # expand some area (10% here)

        slide = openslide.open_slide(row['DX_filename'])

        barcode_col_name = row['barcode_col_name']
        X_col_name = row['X_col_name']
        Y_col_name = row['Y_col_name']
        try:
            barcode_col_name = int(float(barcode_col_name))
            X_col_name = int(float(X_col_name))
            Y_col_name = int(float(Y_col_name))
        except:
            pass
        
        if isinstance(X_col_name, int):
            coord_df = pd.read_csv(row['coord_filename'], header=None)
        else:
            coord_df = pd.read_csv(row['coord_filename'])

        barcodes = coord_df[barcode_col_name].values.tolist()
        stX = coord_df[X_col_name].values.tolist()
        stY = coord_df[Y_col_name].values.tolist()

        level_downsamples = []
        dim_0 = slide.level_dimensions[0]

        for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))
        level_dim = slide.level_dimensions

        patch_level = 0
        results = np.array([stX,stY]).T.astype(np.int32)
        results[:, 0] -= patch_size//2
        results[:, 1] -= patch_size//2
        results = results.astype(np.int32)
        results1 = np.copy(results)
        results1[:, 0] += patch_size # width
        results1[:, 1] += patch_size # height
        invalid_inds = np.concatenate([
            np.where((results[:,0]<=0)|(results[:,1]<=0))[0],
            np.where((results1[:,0]>=row['original_width']-1)|(results1[:,1]>=row['original_height']-1))[0]
        ])

        if len(invalid_inds):
            results = np.delete(results, np.unique(invalid_inds), axis=0)
            # remove invalid_coords and create new coord_filename
            # remove invalid_spots and create new counts_filname
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

        # plot spot figure

        W, H = slide.level_dimensions[0]
        img = slide.read_region((0, 0), 0, (W, H)).convert('RGB')
        draw = ImageDraw.Draw(img)
        img2 = Image.fromarray(255*np.ones((H, W, 3), dtype=np.uint8))
        draw2 = ImageDraw.Draw(img2)
        circle_radius = int(spot_size * 0.5)
        # colors = np.concatenate([colors, 128*np.ones((colors.shape[0], 1), dtype=np.uint8)], axis=1)
        for ind, (x,y) in enumerate(zip(stX, stY)):
            xy = [x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius]
            draw.ellipse(xy, outline=(255, 128, 0), width=8)
            x -= patch_size // 2
            y -= patch_size // 2
            xy = [x, y, x+patch_size, y+patch_size]
            draw2.rectangle(xy, fill=(144, 238, 144))
        img3 = Image.blend(img, img2, alpha=0.4)
        img3.save(os.path.join(save_root, f'{svs_prefix}.jpg'))

        slide.close()
        del img, img2, img3, draw, draw2 


if __name__ == '__main__':
    gen_coords()



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












