


import sys,os,glob,shutil,json
import numpy as np
import pandas as pd
import openslide
import pyvips
import h5py
import torch
import time
from scipy.stats import percentileofscore
from utils import save_hdf5, visHeatmap, score2percentile, to_percentiles
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
from PIL import Image, ImageFile, ImageDraw, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pyvips
import idr_torch
import gc
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import cv2
from common import CLASSIFICATION_DICT, REGRESSION_LIST, IGNORE_INDEX_DICT, PAN_CANCER_SITES
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import red, green, black, gray
from reportlab.graphics.renderPM import drawToFile


def text_image(text, size, font_path='helvetica.ttf', font_size=32):
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font, font_size=font_size)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_pos = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(text_pos, text, fill=(0, 0, 0), font=font, font_size=font_size)
    return np.array(img)

def add_margin_between_images(images, margin=5, is_vertical=False):
    if len(images) == 0:
        return np.array([])

    h, w, c = images[0].shape
    if is_vertical:
        margin_img = np.full((margin, w, c), 255, dtype=np.uint8)
    else:
        margin_img = np.full((h, margin, c), 255, dtype=np.uint8)

    result = images[0]
    for img in images[1:]:
        result = np.concatenate((result, margin_img, img), axis=0 if is_vertical else 1)
    return result

def combine_images_with_labels(images, rows, cols, image_size, margin=5, font_path='helvetica.ttf', font_size=96):
    row_labels = [text_image('Cluster {}'.format(str(i)), image_size, font_path, font_size=font_size) for i in range(rows)]
    # col_labels = [text_image('Top-{}'.format(str(j)), image_size, font_path, font_size=font_size) for j in range(cols)]

    images = images.reshape(rows, cols, image_size[0], image_size[1], 3)

    labeled_rows = []
    for i in range(rows):
        row_images = [row_labels[i]] + [images[i, j] for j in range(cols)]
        labeled_row = add_margin_between_images(row_images, margin=margin)
        labeled_rows.append(labeled_row)

    top_row_images = [np.full((*image_size, 3), 255, dtype=np.uint8)]#  + col_labels
    top_row = add_margin_between_images(top_row_images, margin=margin)

    # full_grid = add_margin_between_images([top_row] + labeled_rows, margin=margin, is_vertical=True)
    full_grid = add_margin_between_images(labeled_rows, margin=margin, is_vertical=True)
    return full_grid

def test():

    # Example usage
    images = np.random.randint(0, 256, (40, 28, 28, 3), dtype=np.uint8)
    combined = combine_images_with_labels(images, 8, 5, (28, 28), margin=5)

    plt.imshow(combined, cmap='gray')
    plt.axis('off')
    plt.show()




def clustering(X0, n_clusters=8, top_n=5):

   # scaler = MinMaxScaler()
    scaler = StandardScaler()
    X = scaler.fit_transform(X0)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Get distances to centroids
    distances = kmeans.transform(X)  # shape: (n_samples, n_clusters)

    # For each cluster, find top N closest samples
    top_nearest_indices = []
    cluster_ids = []
    all_cluster_indices = []
    for cluster_id in range(kmeans.n_clusters):
        # Get distances to this centroid
        dists_to_cluster = distances[:, cluster_id]
        
        # Get indices of samples assigned to this cluster
        cluster_members = np.where(kmeans.labels_ == cluster_id)[0]
        all_cluster_indices.append(cluster_members)
        
        # Sort those members by distance to centroid
        sorted_members = cluster_members[np.argsort(dists_to_cluster[cluster_members])]
        
        # Take top N
        top_n_ids = sorted_members[:min(top_n, len(sorted_members))]
        top_nearest_indices.append(top_n_ids)
        cluster_ids.append(np.ones_like(top_n_ids)*cluster_id)
    
    return np.concatenate(top_nearest_indices), np.concatenate(cluster_ids), np.concatenate(all_cluster_indices)


def _save_pdf(save_dir, texts, colors, svs_prefix, cancer_type, thres1, thres2):


    left_path = os.path.join(save_dir, f"bottom.png")
    right_path = os.path.join(save_dir, f"top.png")

    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        print(f"Missing image(s), skipping.")
        return False

    left_img = Image.open(left_path)
    right_img = Image.open(right_path)

    output_pdf = os.path.join(save_dir, '..', f'{svs_prefix}.pdf')
    page_width, page_height = landscape(A4)

    c = canvas.Canvas(output_pdf, pagesize=landscape(A4))
    padding = 20
    title_height = 30
    font_size = 16
    font_name = "Helvetica"

    # Calculate max dimensions per side
    max_width = (page_width - 3 * padding) / 2
    max_height = page_height - title_height - 4 * padding

    def get_scaled_dimensions(img):
        ratio = min(max_width / img.width, max_height / img.height, 1.0)
        return int(img.width * ratio), int(img.height * ratio)

    left_w, left_h = get_scaled_dimensions(left_img)
    right_w, right_h = get_scaled_dimensions(right_img)

    left_x = padding
    right_x = padding * 2 + max_width
    y = (page_height - max(left_h, right_h)) / 2 - 10

    # Draw titles
    c.setFont(font_name, font_size)
    c.drawCentredString(page_width/2, page_height-padding, svs_prefix + " ({})".format(cancer_type))
    # c.drawCentredString(page_width/2, page_height-2*padding, os.path.basename(dir_path))

    total_width = sum(c.stringWidth(word + "  ", font_name, font_size) for word in texts)
    start_x = (page_width - total_width) / 2
    xx = start_x
    yy = page_height-2*padding
    for word, color in zip(texts, colors):
        c.setFillColor(color)
        c.drawString(xx, yy, word)
        # Add spacing after the word based on its width
        xx += c.stringWidth(word + "  ", font_name, font_size)
    
    c.setFillColor(black)
    c.drawCentredString(left_x + left_w / 2, page_height - 3*padding,   f"       Top 5 patches (attention_score<{thres1:.2f})")
    c.drawCentredString(right_x + right_w / 2, page_height - 3*padding, f"       Top 5 patches (attention_score>{thres2:.2f})")

    # Draw images without resizing them beforehand
    c.drawImage(left_path, left_x, y, width=left_w, height=left_h, preserveAspectRatio=True, mask='auto')
    c.drawImage(right_path, right_x, y, width=right_w, height=right_h, preserveAspectRatio=True, mask='auto')

    c.showPage()

    c.save()
    # print(f"PDF saved to {output_pdf}")
    # image = convert_from_bytes(c.getpdfdata())[0]
    # image.save(output_pdf.replace('.pdf', '.png'))
    # drawToFile(c, output_pdf.replace('.pdf', '.png'), fmt='PNG')
    
    return True



def main():

    subset = sys.argv[1]
    model_name = sys.argv[2]
    best_split = int(sys.argv[3])
    prefix = 'TCGA_{}{}'.format(subset, best_split) # sys.argv[1] # TCGA_trainval3 or TCGA_test3

    csv_filename = f'/data/zhongz2/temp29/debug/splits/{subset}-{best_split}.csv'
    all_labels = pd.read_csv(csv_filename, low_memory=False)
    all_labels['cancer_type'] = all_labels['PanCancerSiteID'].map({site_id+1: site_name for site_id, site_name in enumerate(PAN_CANCER_SITES)})
    all_labels['svs_prefix'] = [os.path.splitext(os.path.basename(row['DX_filename']))[0] for _, row in all_labels.iterrows()]
    all_labels = all_labels.set_index('svs_prefix')

    all_results = {}
    for cls_name in CLASSIFICATION_DICT.keys():
        print(cls_name)
        all_results[cls_name] = pd.read_csv(f'/data/zhongz2/CPTAC/predictions_v2_TCGA_filterTrue_2_20250409/{subset}/{model_name}_{cls_name}_gt_and_pred.csv')

    svs_dir = "/data/zhongz2/tcga/TCGA-ALL2_256/svs"
    patches_dir = "/data/zhongz2/tcga/TCGA-ALL2_256/patches"
    feats_dir = f'/data/zhongz2/download/{prefix}/{model_name}/pt_files'
    preds_dir = f'/data/zhongz2/download/{prefix}/{model_name}/pred_files'

    save_root = preds_dir.replace('pred_files', 'heatmap_files')
    save_root = preds_dir.replace('pred_files', 'heatmap_files_check2') # for Revision 2 20250411
    os.makedirs(save_root, exist_ok=True)
    save_root_top_patches = preds_dir.replace('pred_files', 'heatmap_top_patches2')
    os.makedirs(save_root_top_patches, exist_ok=True)

    pt_files = glob.glob(os.path.join(preds_dir, '*.pt'))
    # existed_prefixes = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(save_root, '*.tif'))]
    existed_prefixes = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(save_root_top_patches) if '.pdf' in f and f[0]!='.']
    needtodo_files = [f for f in pt_files if os.path.splitext(os.path.basename(f))[0] not in existed_prefixes]

    print('existing files: ', len(existed_prefixes))
    print('needtodo', len(needtodo_files))
    indices = np.arange(len(needtodo_files))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    # print('index_splits', index_splits)
    needtodo_files = [needtodo_files[i] for i in index_splits[idr_torch.rank]]
    print(idr_torch.rank, len(needtodo_files))

    
    for fi, f in enumerate(needtodo_files):
        # if fi == 10:
        #     break

        top_bottom = {
            'top':0.9,
            'bottom':0.1
        }
        num_clusters = 8
        top_n = 5
        margin = 5
        is_ok = False
        while True:

            if num_clusters <= 1 or top_bottom['bottom']>=0.6 or top_bottom['top']<=0.3:
                break

            svs_prefix = os.path.splitext(os.path.basename(f))[0]

            A_raw = torch.load(os.path.join(preds_dir, svs_prefix+'.pt'))['A_raw']

            with h5py.File(os.path.join(patches_dir, svs_prefix+'.h5'), 'r') as file:
                all_coords = file['coords'][:]
                patch_size = file['coords'].attrs['patch_size']
                patch_level = file['coords'].attrs['patch_level']

            slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))

            cancer_type = all_labels.loc[svs_prefix]['cancer_type']

            texts = []
            colors = []
            for cls_name in sorted(CLASSIFICATION_DICT.keys()):
                df = all_results[cls_name]
                if svs_prefix in df['svs_prefix'].values:
                    df1 = df[df['svs_prefix']==svs_prefix]
                    gt = df1['gt'].values[0]
                    pred = df1['pred'].values[0]
                    texts.append('{}({})'.format(cls_name.replace('_cls', ''), gt))

                    if gt == pred:
                        colors.append(green)
                    else:
                        colors.append(red)
                else:
                    texts.append(cls_name.replace('_cls', ''))
                    colors.append(gray)
                        

            try:
                A = np.copy(A_raw)[0]
                attention_scores = to_percentiles(A)
                attention_scores /= 100

                save_dir = os.path.join(save_root_top_patches, svs_prefix)
                os.makedirs(save_dir, exist_ok=True)
                feats = torch.load(os.path.join(feats_dir, svs_prefix+'.pt'))

                for tb,tb_thres in top_bottom.items():

                    if tb=='top':
                        red_inds = np.where(attention_scores>tb_thres)[0]
                    else:
                        red_inds = np.where(attention_scores<tb_thres)[0]
                    
                    if len(red_inds) < num_clusters*top_n:
                        continue

                    red_feats = feats[red_inds].cpu().numpy()
                    red_top_inds, red_top_cluster_ids, red_cluster_ids = clustering(red_feats, n_clusters=num_clusters, top_n=top_n)
                    red_top_inds = red_inds[red_top_inds]

                    if False:
                        whole_img = 255*np.ones((num_clusters*patch_size+(num_clusters-1)*margin, top_n*patch_size+(top_n-1)*margin, 3), dtype=np.uint8)
                        for rid in range(num_clusters):
                            rid_inds = np.where(red_top_cluster_ids==rid)[0]
                            for cid, ind in enumerate(red_top_inds[rid_inds]):
                                x,y = all_coords[ind]
                                patch = np.array(slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert('RGB'))
                                whole_img[rid*(patch_size+margin):(rid*(patch_size+margin)+patch_size), cid*(patch_size+margin):(cid*(patch_size+margin)+patch_size), :] = patch
                        cv2.imwrite(os.path.join(save_dir, f'{tb}.jpg'), whole_img[:,:,::-1])
                    else:
                        images = []
                        for rid in range(num_clusters):
                            rid_inds = np.where(red_top_cluster_ids==rid)[0]
                            for cid, ind in enumerate(red_top_inds[rid_inds]):
                                x,y = all_coords[ind]
                                patch = np.array(slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert('RGB'))
                                images.append(patch)
                        images = np.stack(images)
                        combined = combine_images_with_labels(images, num_clusters, top_n, (patch_size, patch_size), margin=margin,\
                            font_size=48)
                        cv2.imwrite(os.path.join(save_dir, f'{tb}.png'), combined[:,:,::-1])
                    # patch.save(os.path.join(save_dir, f'{tb}_c{cid}_patch_x{x}_y{y}.jpg'))

                time.sleep(1)
                is_ok = _save_pdf(save_dir, texts, colors, svs_prefix, cancer_type, top_bottom['bottom'], top_bottom['top'])


                if False:
                    save_filename = '{}/{}_heatmap.tif'.format(save_root, svs_prefix)
                    img = visHeatmap(slide, scores=A, coords=all_coords,
                                    vis_level=0, patch_size=(patch_size, patch_size),
                                    convert_to_percentiles=True)
                    print(type(img), img.size)
                    # img.save(save_filename)
                    img_vips = pyvips.Image.new_from_array(img)
                    # img_vips.dzsave(save_filename, tile_size=1024)
                    img_vips.tiffsave(save_filename, compression="jpeg",
                        tile=True, tile_width=256, tile_height=256,
                        pyramid=True,  bigtiff=True)
                    del img, img_vips
                    gc.collect()

                # img_vips.write_to_file(save_filename, tile=True, compression="jpeg", bigtiff=True, pyramid=True)
                # time.sleep(1)
                # del img, img_vips

            except:
                print(f'error {f}')
                is_ok = False

            if not is_ok:
                top_bottom['bottom']+=0.05
                top_bottom['top']-=0.05
                num_clusters-=2
            else:
                break


def find_case_with_mutation1():

    import sys,os,shutil,glob
    import pandas as pd
    import numpy as np
    from common import CLASSIFICATION_DICT, REGRESSION_LIST, IGNORE_INDEX_DICT, PAN_CANCER_SITES

    subset = 'test'
    model_name = 'CONCH'
    best_split = 3
    prefix = 'TCGA_{}{}'.format(subset, best_split) # sys.argv[1] # TCGA_trainval3 or TCGA_test3
    result_dir = f'/data/zhongz2/download/TCGA_{subset}{best_split}/{model_name}/heatmap_top_patches2'

    csv_filename = f'/data/zhongz2/temp29/debug/splits/{subset}-{best_split}.csv'
    all_labels = pd.read_csv(csv_filename, low_memory=False)
    all_labels['cancer_type'] = all_labels['PanCancerSiteID'].map({site_id+1: site_name for site_id, site_name in enumerate(PAN_CANCER_SITES)})
    all_labels['svs_prefix'] = [os.path.splitext(os.path.basename(row['DX_filename']))[0] for _, row in all_labels.iterrows()]
    all_labels = all_labels.set_index('svs_prefix')

    all_results = {}
    result_dict = {}
    for cls_name in CLASSIFICATION_DICT.keys():
        print(cls_name)
        df = pd.read_csv(f'/data/zhongz2/CPTAC/predictions_v2_TCGA_filterTrue_2_20250409/{subset}/{model_name}_{cls_name}_gt_and_pred.csv')

        all_results[cls_name] = df

        right = df[(df['gt']==df['pred']) & (df['gt']==1)]
        wrong = df[(df['gt']!=df['pred']) & (df['gt']==1)]

        if len(right) > 0:
            if 'right' in result_dict:
                result_dict['right'].extend(right['svs_prefix'].values.tolist())
            else:
                result_dict['right'] = right['svs_prefix'].values.tolist()

        if len(wrong) > 0:
            if 'wrong' in result_dict:
                result_dict['wrong'].extend(wrong['svs_prefix'].values.tolist())
            else:
                result_dict['wrong'] = wrong['svs_prefix'].values.tolist()

    save_root = '/data/zhongz2/check_attention_patches'
    os.makedirs(save_root, exist_ok=True)


    for k, l in result_dict.items():
        a,b = np.unique(l, return_counts=True)
        inds = np.argsort(b)[::-1]
        a = a[inds]
        b = b[inds]
        print(k, b)
        save_dir = os.path.join(save_root, k)
        os.makedirs(save_dir, exist_ok=True)
        for aa,bb in zip(a, b):
            os.system('cp "{}" "{}"'.format(os.path.join(result_dir, aa+'.pdf'), os.path.join(save_dir, f'{bb}_{aa}.pdf')))


if __name__ == '__main__':
    main()