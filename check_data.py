import sys,os,shutil,glob
import pandas as pd


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












