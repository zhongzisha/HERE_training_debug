import sys,os,shutil,glob
import pandas as pd


def main1():
    splits_dir = '/data/zhongz2/temp29/debug/splits'


    split = 0
    alldf = []
    for subset in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(splits_dir, '{}-{}.csv'.format(subset, split)), index_col=0, low_memory=False)
        alldf.append(df)
    alldf = pd.concat(alldf)

    svs_prefixes = [os.path.basename(f).replace('.svs', '') for f in alldf['DX_filename']]

    df = pd.read_excel('/data/zhongz2/tcga_ffpe_all/appmag_mpp.xlsx', index_col=0)
    df['svs_prefix'] = [os.path.basename(f).replace('.svs', '') for f in df['image_filename']]
    df1 = df[df['svs_prefix'].isin(svs_prefixes)]

    for _, row in df.iterrows():
        os.symlink(row['image_filename'], "/data/zhongz2/tcga/TCGA-ALL2_256/allsvs/{}.svs".format(row['svs_prefix']))


def TCGA_data():
    # combine TCGA train/val/test

    import sys,os,shutil,glob
    import pandas as pd
    import numpy as np
    from common import CLASSIFICATION_DICT, ALL_CLASSIFICATION_DICT, PAN_CANCER_SITES

    train = pd.read_csv('splits/train-0.csv', low_memory=False)
    val = pd.read_csv('splits/val-0.csv', low_memory=False)
    test = pd.read_csv('splits/test-0.csv', low_memory=False)

    df = pd.concat([train, val, test], axis=0)
    df['cancer_type'] = df['PanCancerSiteID'].map({site_id+1: site_name for site_id, site_name in enumerate(PAN_CANCER_SITES)})


    cols = [col for col in df.columns if 'Unnamed' not in col]
    df = df[cols]

    cols1 = [col for col in df.columns if '_cls' in col]
    cols2 = [col for col in df.columns if 'HALLMARK' in col]

    df = df[['case_id', 'cancer_type', 'slide_id', 'DX_filename']+cols1+cols2]

    mapper_column_names = {'case_id': 'ID', 'DX_filename': 'image_filename'}
    for col in cols1:
        mapper_column_names[col] = col.replace('_cls', '')
    for col in cols2:
        mapper_column_names[col] = col.replace('_sum', '')

    df = df.rename(columns=mapper_column_names)

    df['image_filename'] = [os.path.basename(f) for f in df['image_filename'].values]

    df.to_excel('HERE_TCGA.xlsx', index=None)

    for col in cols1:
        col = col.replace('_cls','')
        print(df[col].value_counts())



def CPTAC():

    import sys,os,shutil,glob
    import pandas as pd
    import numpy as np
    from common import CLASSIFICATION_DICT, ALL_CLASSIFICATION_DICT, REGRESSION_LIST

    clinical2 = pd.read_csv('/Users/zhongz2/down/clinical.project-cptac-2.2024-12-20/clinical.tsv', sep='\t')
    clinical3 = pd.read_csv('/Users/zhongz2/down/clinical.project-cptac-3.2024-12-20/clinical.tsv', sep='\t')
    clinical = pd.concat([clinical2, clinical3], axis=0)
    clinical = clinical.groupby('case_submitter_id').first().reset_index()

    df1 = pd.read_csv('splits/PanCancer_CONCH_labels.csv')
    df2 = pd.read_csv('splits/PanCancer_CONCH_predictions.csv', index_col=0)
    df3 = pd.read_csv('splits/TP53_HERE_CONCH_results.csv', index_col=0)

    df1['svs_prefix'] = df2['svs_prefix']
    df1.drop(columns=['barcode'], inplace=True)

    df = df3.merge(df1, left_on='svs_prefix', right_on='svs_prefix', how='left')

    cols = []
    for k,v in CLASSIFICATION_DICT.items():
        if k+'_label' in df.columns:
            cols.append(k+'_label')
    for k in REGRESSION_LIST:
        cols.append(k.replace('TIDE_',''))

    df = df[['barcode', 'svs_prefix'] + cols]


    with open('splits/allsvs.txt', 'r') as fp:
        filenames = [line.strip() for line in fp.readlines()]
    df2 = pd.DataFrame(filenames, columns=['image_filename'])
    df2['svs_prefix'] = [os.path.splitext(os.path.basename(f))[0] for f in df2['image_filename'].values]
    df2['cancer_type'] = [f.split('/')[-2] for f in df2['image_filename'].values]

    # df = df.merge(df2, left_on='svs_prefix', right_on='svs_prefix', how='inner').reset_index(drop=True)
    df = df2.merge(df, left_on='svs_prefix', right_on='svs_prefix', how='left').reset_index(drop=True)

    cols1 = [col for col in df.columns if '_cls' in col]
    cols2 = [col for col in df.columns if 'HALLMARK_' in col]

    mapper_column_names = {'barcode': 'ID', 'svs_prefix': 'slide_id'}
    gene_mut_cols = []
    for col in cols1:
        gene_mut_cols.append(col.replace('_cls_label', ''))
        mapper_column_names[col] = col.replace('_cls_label', '')
    gene_exp_cols = []
    for col in cols2:
        gene_exp_cols.append(col.replace('_sum', ''))
        mapper_column_names[col] = col.replace('_sum', '')

    if True:
        df1 = df.copy()
        barcodes = []
        for svs_prefix in df['svs_prefix'].values:
            found = []
            for v in clinical['case_submitter_id'].values:
                if v in svs_prefix:
                    found.append(v)
            if len(found) == 1:  # exact one match
                barcodes.append(found[0])
            elif len(found) == 0: # no match
                barcodes.append('FAKE_CASE')
            elif svs_prefix in found: # multi match, has one exact match
                barcodes.append(svs_prefix)
            else: 
                found1 = [v for v in found if v not in ['604', '1488']]
                print(svs_prefix, found, found1)
                barcodes.append(found1[0])
        df1['barcode'] = barcodes
        clinical.loc['FAKE_CASE'] = [pd.NA for _ in range(len(clinical.columns))]
        df2 = df1.merge(clinical, left_on='barcode', right_on='case_submitter_id', how='left').reset_index(drop=True)

        df2['image_filename'] = [f.replace('/data/zhongz2/CPTAC/allsvs/', '') for f in df2['image_filename'].values]

        df2.to_excel('splits/CPTAC_all.xlsx')

    df = df.rename(columns=mapper_column_names)
    df = df[['ID','cancer_type', 'slide_id', 'image_filename']+gene_mut_cols+gene_exp_cols]

    df1 = df[df['ID']!='FAKE_CASE']
    df2 = df[df['ID']=='FAKE_CASE']

    df1 = df1.groupby('ID').first()

    image_filenames = []
    for ID in df1.index.values:
        image_filenames.append(','.join(df[df['ID']==ID]['image_filename'].values.tolist()))
    df1['image_filename'] = image_filenames
    df1.drop(columns=['slide_id'], inplace=True)
    df1 = df1.reset_index()
    df2 = df2.drop(columns=['slide_id'])

    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    for col in gene_mut_cols:
        df.loc[df['ID']=='FAKE_CASE', col] = pd.NA
    df.loc[df['ID']=='FAKE_CASE', 'ID'] = pd.NA

    df['image_filename'] = [f.replace('/data/zhongz2/CPTAC/allsvs/', '') for f in df['image_filename'].values]

    df.to_excel('HERE_CPTAC.xlsx', index=None)

# patches: 323027031