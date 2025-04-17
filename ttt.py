import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import red, green, black
import glob
import pandas as pd
from common import CLASSIFICATION_DICT, REGRESSION_LIST, IGNORE_INDEX_DICT, PAN_CANCER_SITES

# Update this with your list of directories
directories = ["dir1", "dir2", "dir3"]
subset = 'test'
split = 3
model_name = 'CONCH'
directories = glob.glob(f'/data/zhongz2/download/TCGA_{subset}{split}/{model_name}/heatmap_top_patches2/TCGA*')

csv_filename = f'/data/zhongz2/temp29/debug/splits/{subset}-{split}.csv'
all_labels = pd.read_csv(csv_filename, low_memory=False)
all_labels['cancer_type'] = all_labels['PanCancerSiteID'].map({site_id+1: site_name for site_id, site_name in enumerate(PAN_CANCER_SITES)})
all_labels['svs_prefix'] = [os.path.splitext(os.path.basename(row['DX_filename']))[0] for _, row in all_labels.iterrows()]
all_labels = all_labels.set_index('svs_prefix')

all_results = {}
for cls_name in CLASSIFICATION_DICT.keys():
    print(cls_name)
    all_results[cls_name] = pd.read_csv(f'/data/zhongz2/CPTAC/predictions_v2_TCGA_filterTrue_2_20250409/{subset}/{model_name}_{cls_name}_gt_and_pred.csv')

# List of directories to process
# directories = ["dir1", "dir2", "dir3"]

output_pdf = "combined_images_high_quality.pdf"
page_width, page_height = landscape(A4)

c = canvas.Canvas(output_pdf, pagesize=landscape(A4))
padding = 20
title_height = 30
font_size = 16
font_name = "Helvetica"

for dir_path in directories:
    svs_prefix = os.path.basename(dir_path)

    cancer_type = all_labels.loc[svs_prefix]['cancer_type']

    texts = []
    colors = []
    for cls_name in CLASSIFICATION_DICT.keys():
        df = all_results[cls_name]
        if svs_prefix in df['svs_prefix'].values:
            df1 = df[df['svs_prefix']==svs_prefix]
            gt = df1['gt'].values[0]
            pred = df1['pred'].values[0]
            texts.append(cls_name.replace('_cls', ''))
            if gt == pred:
                colors.append(green)
            else:
                colors.append(red)
                

    left_path = os.path.join(dir_path, "bottom.png")
    right_path = os.path.join(dir_path, "top.png")
    
    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        print(f"Missing image(s) in {dir_path}, skipping.")
        continue

    left_img = Image.open(left_path)
    right_img = Image.open(right_path)

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

    total_width = sum(c.stringWidth(word + " ", font_name, font_size) for word in texts)
    start_x = (page_width - total_width) / 2
    xx = start_x
    yy = page_height-2*padding
    for word, color in zip(texts, colors):
        c.setFillColor(color)
        c.drawString(xx, yy, word)
        # Add spacing after the word based on its width
        xx += c.stringWidth(word + " ", font_name, font_size)
    
    c.setFillColor(black)
    c.drawCentredString(left_x + left_w / 2, page_height - 3*padding,   f"       Top 5 patches (attention_score<0.1)")
    c.drawCentredString(right_x + right_w / 2, page_height - 3*padding, f"       Top 5 patches (attention_score>0.9)")

    # Draw images without resizing them beforehand
    c.drawImage(left_path, left_x, y, width=left_w, height=left_h, preserveAspectRatio=True, mask='auto')
    c.drawImage(right_path, right_x, y, width=right_w, height=right_h, preserveAspectRatio=True, mask='auto')

    c.showPage()

c.save()
print(f"PDF saved to {output_pdf}")
