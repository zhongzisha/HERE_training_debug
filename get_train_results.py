import sys, os, glob, shutil
import pandas as pd
import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from common import PAN_CANCER_SITES, CLASSIFICATION_DICT, REGRESSION_LIST
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.colors import red, green, black, gray
from PIL import Image

morandi_colors = [
    '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', 
    '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', 
    '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690'
    ]

COMMON_COLORS = {
    'mobilenetv3': 'orange',
    'MobileNetV3': 'orange',
    'CLIP': 'Green',
    'PLIP': 'gray',
    'ProvGigaPath': 'purple',
    'CONCH': 'blue',
    'UNI': '#008080',# '#029370', morandi_colors[-1]
}
from plot_results_20250113_UNI import all_colors
COMMON_COLORS = all_colors

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def shorten_prefix2(prefix):
    prefix = prefix.replace('imagenet', '').replace('_adam_None_weighted_ce_', '').replace(
        '_wd1e-4_reguNone1e-4_25632', '')
    prefix = prefix.replace('mobilenetv38', 'MobileNetV3')
    for pp in ['CLIP', 'PLIP', 'ProvGigaPath', 'CONCH', 'MobileNetV3', 'mobilenetv3', 'UNI']: 
        prefix = prefix.replace(f'{pp}_4', pp)
        prefix = prefix.replace(f'{pp}_8', pp)
    return prefix


def plot_box_plot_for_check(scores, prefixes, title, save_filename):
    colors = ['#D7191C', '#2C7BB6']
    data_a = scores.tolist()
    fig = plt.figure()
    bpl = plt.boxplot(data_a, positions=np.arange(len(data_a)), sym='', widths=0.6)
    set_box_color(bpl, colors[0])  # colors are from http://colorbrewer2.org/

    # draw circles
    xs = []
    for x1, y1 in zip(np.arange(len(data_a)), data_a):
        xs.append(x1)
        x1 = np.random.normal(x1, 0.04, size=len(y1))
        plt.scatter(x1, y1, c=colors[0], alpha=0.2)

    plt.xticks(np.arange(len(data_a)), [shorten_prefix2(prefix) for prefix in prefixes], rotation=45, ha="right", rotation_mode="anchor")
    plt.xlabel('Different encoders')
    plt.ylabel('Ranking frequency among {} tasks.'.format(len(data_a[0])))
    plt.savefig(save_filename, bbox_inches='tight', dpi=600)
    plt.savefig(save_filename.replace('.png', '.svg'), bbox_inches='tight')
    plt.title(title)
    plt.close(fig)


def save_figure_to_one_pdf_new2(subset, task_type, task_name, mean_results, std_results, select_columns,
                           epoch_step, prefixes, save_root, pdf_file_handle=None):
    epochs = np.arange(mean_results[0].shape[0])
    indices = np.arange(0, mean_results[0].shape[0], epoch_step)

    ranks = []
    mean_values1 = []
    for ki, key in enumerate(select_columns):
        mean_values = []
        for ii, (mean1, std1) in enumerate(zip(mean_results, std_results)):
            mean_value = mean1[int(0.4 * len(mean1)):, ki]  # last 60% epochs
            sort_index = np.argsort(mean_value.flatten())
            mean_values.append(np.mean(mean_value[sort_index[5:-5]]))
        mean_values = np.array(mean_values)
        mean_values1.append(mean_values)
        if 'auc' in key or 'c_index' in key or 'r2score' in key or 'pearsonr' in key or 'pearmanr' in key:
            sorted_indices = np.argsort(mean_values)[::-1]  # for classification, bigger, better
        else:
            sorted_indices = np.argsort(mean_values)  # for MSE and loss, smaller, better
        ranks.append({ind: rank for rank, ind in enumerate(sorted_indices)})

    best_scores = {}
    img_paths = []
    for ki, key in enumerate(select_columns):
        if 'auc' in key or 'mse' in key or 'c_index' in key \
                or 'r2score' in key or 'pearsonr' in key or 'pearmanr' in key:
            pass
        # else:  # surv, loss
            # key = key.replace('_cls', '').replace('_sum', '')

        font_size = 18
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
        plt.xlabel('Epochs')
        if task_type != 'loss':
            plt.ylabel('AUC scores' if 'cls' in task_type else 'Spearmanr correlation score')
        else:
            if '_cls' == key[-4:]:
                plt.ylabel('Cross-entropy loss')
            else:
                plt.ylabel('Mean-squared loss')

        # plt.title('{} task: {}'.format(task_type, key), fontsize=font_size)
        if task_type == 'loss':
            key1 = key.replace('_cls', '').replace('_sum', '')
        else:
            key1 = key
        title = key1.replace('_cls_auc_weighted', '') if 'auc' in key1 else key1.replace('_sum_spearmanr_corr', '').replace('_spearmanr_corr', '')
        plt.title(title + " ({})".format(subset), fontsize=font_size)
        plt.tick_params(pad = 10)
        print(key)

        best_label = None
        best_score = None
        for ii, (mean1, std1) in enumerate(zip(mean_results, std_results)):
            rank = ranks[ki][ii]
            mean_value = mean_values1[ki][ii]
            # label = '({}:{:.6f}) {}'.format(rank, mean_value, shorten_prefix2(prefixes[ii]))
            label = shorten_prefix2(prefixes[ii])
            if rank == 0:
                # plt.errorbar(epochs[indices], mean1[indices, ki], std1[indices, ki], fmt='-o',
                #              label=label,
                #              linewidth=4, color='red')
                plt.errorbar(epochs[indices], mean1[indices, ki], std1[indices, ki], fmt='-o',
                             label=label, color=COMMON_COLORS[label],
                             linewidth=4)
                best_label = label
                best_score = mean_value
            else:
                plt.errorbar(epochs[indices], mean1[indices, ki], std1[indices, ki], fmt='-o', color=COMMON_COLORS[label],
                             label=label)

        best_scores[key] = best_score

        plt.grid()
        ylim = plt.ylim()
        if subset == 'val' and 'Stage' in task_name and 'PAAD' in save_root:  # 20221119, Peng suggest to remove outliers
            plt.ylim(0, 0.5)
        if subset == 'val' and 'CAF' in task_name and 'PAAD' in save_root:  # 20221119, Peng suggest to remove outliers
            plt.ylim(0, 0.3)
        if subset == 'val' and 'CAF' in task_name and 'BRCA' in save_root:  # 20221119, Peng suggest to remove outliers
            plt.ylim(0, 0.05)
        if subset == 'val' and 'CTL' in task_name and 'PAAD' in save_root:  # 20221119, Peng suggest to remove outliers
            plt.ylim(0, 1.5)
        if subset == 'val' and 'Dys' in task_name:  # 20221119, Peng suggest to remove outliers
            plt.ylim(0, 1)
        if subset == 'val' and 'MDSC' in task_name:  # 20221119, Peng suggest to remove outliers
            plt.ylim(0, 0.3)
        ylim = plt.ylim()
        if 'r2score' in key or 'personr' in key or 'pearmanr' in key:
            plt.ylim(-1.0, 1.1)
        if '_pearmanr_corr' in key:
            key = key.replace('_pearmanr_corr', '_pearsonr_corr')
        
        plt.tight_layout()

        save_filename = os.path.join(save_root, '{}_{}_step{}.png'.format(subset, key.replace('/', '_'), epoch_step))
        plt.savefig(save_filename, bbox_inches='tight', transparent=True, dpi=600)
        plt.savefig(save_filename.replace('.png', '.svg'), bbox_inches='tight', transparent=True, format='svg')

        if pdf_file_handle is not None:
            pdf_file_handle.savefig(fig)

        plt.close()

        img_paths.append(save_filename)
    return ranks, best_scores, img_paths

"""
begin mobilenetv3
TCGA-ALL2, best_epoch: 32, best_split: 3, mean value: [26.96678552], 
[[26.73870231]
 [26.69949782]
 [26.9149441 ]
 [27.80078376]
 [26.67999959]]

begin CLIP
TCGA-ALL2, best_epoch: 48, best_split: 3, mean value: [25.46747714], 
[[25.51961642]
 [25.80916389]
 [24.47571859]
 [26.54644472]
 [24.9864421 ]]

begin PLIP
TCGA-ALL2, best_epoch: 44, best_split: 2, mean value: [28.83604787], 
[[28.62619627]
 [28.94292738]
 [29.03153617]
 [28.7766052 ]
 [28.80297432]]

begin ProvGigaPath
TCGA-ALL2, best_epoch: 39, best_split: 1, mean value: [33.24150728], 
[[33.26106165]
 [33.49437003]
 [33.0977323 ]
 [33.37102611]
 [32.98334629]]

begin CONCH
TCGA-ALL2, best_epoch: 37, best_split: 3, mean value: [33.95193159], 
[[32.8254739 ]
 [33.69786708]
 [33.71757559]
 [35.09961563]
 [34.41912573]]
"""


""" epoch 50
begin mobilenetv3
TCGA-ALL2, best_epoch: 32, best_split: 1, mean value: [26.84497921], 
[[26.47217891]
 [27.22670201]
 [26.97138443]
 [26.95306755]
 [26.60156315]]

begin CLIP
TCGA-ALL2, best_epoch: 48, best_split: 3, mean value: [25.50994445], 
[[25.42180675]
 [25.61419662]
 [24.69003383]
 [26.51350539]
 [25.31017968]]

begin PLIP
TCGA-ALL2, best_epoch: 49, best_split: 3, mean value: [29.28577201], 
[[29.60092453]
 [28.7463423 ]
 [29.06418942]
 [30.18882975]
 [28.82857405]]

begin ProvGigaPath
TCGA-ALL2, best_epoch: 39, best_split: 1, mean value: [33.24150728], 
[[33.26106165]
 [33.49437003]
 [33.0977323 ]
 [33.37102611]
 [32.98334629]]

begin CONCH
TCGA-ALL2, best_epoch: 37, best_split: 3, mean value: [33.95193159], 
[[32.8254739 ]
 [33.69786708]
 [33.71757559]
 [35.09961563]
 [34.41912573]]
"""

""" epoch 100
begin mobilenetv3
TCGA-ALL2, best_epoch: 32, best_split: 3, mean value: [26.96678552], 
[[26.73870231]
 [26.69949782]
 [26.9149441 ]
 [27.80078376]
 [26.67999959]]

begin CLIP
TCGA-ALL2, best_epoch: 97, best_split: 1, mean value: [26.46315352], 
[[26.72797169]
 [27.00353473]
 [26.0617172 ]
 [26.90730903]
 [25.61523493]]

begin PLIP
TCGA-ALL2, best_epoch: 66, best_split: 3, mean value: [29.25514214], 
[[29.86416785]
 [28.85635187]
 [28.72621062]
 [30.07893741]
 [28.75004294]]

begin ProvGigaPath
TCGA-ALL2, best_epoch: 39, best_split: 1, mean value: [33.24150728], 
[[33.26106165]
 [33.49437003]
 [33.0977323 ]
 [33.37102611]
 [32.98334629]]

begin CONCH
TCGA-ALL2, best_epoch: 53, best_split: 3, mean value: [34.16496451], 
[[33.35970299]
 [33.56453657]
 [34.04463461]
 [35.49385124]
 [34.36209713]]

begin UNI
TCGA-ALL2, best_epoch: 58, best_split: 3, mean value: [33.65394501], 
[[33.28786349]
 [33.84315466]
 [33.62571556]
 [33.92088263]
 [33.59210872]]
"""


""" no attention
begin PLIP
TCGA-ALL2, best_epoch: 66, best_split: 1, mean value: [29.11916396], 
[[28.57568554]
 [29.89418162]
 [28.20150555]
 [29.74166515]
 [29.18278191]]

begin ProvGigaPath
TCGA-ALL2, best_epoch: 39, best_split: 0, mean value: [32.74395793], 
[[33.32756371]
 [32.95985017]
 [32.40131981]
 [32.75990765]
 [32.27114829]]

begin CONCH
TCGA-ALL2, best_epoch: 61, best_split: 1, mean value: [33.41015442], 
[[32.7633769 ]
 [34.12984309]
 [32.72588764]
 [33.79433179]
 [33.63733268]]

begin UNI
TCGA-ALL2, best_epoch: 44, best_split: 1, mean value: [33.00725942], 
[[33.0650199 ]
 [33.75186445]
 [32.61928498]
 [33.1197163 ]
 [32.48041149]]
"""

def compare_attention_with_noattention():

    atten_values = {
        'PLIP': [[29.86416785],
                [28.85635187],
                [28.72621062],
                [30.07893741],
                [28.75004294]],
        'ProvGigaPath': [[33.26106165],
                [33.49437003],
                [33.0977323 ],
                [33.37102611],
                [32.98334629]],
        'CONCH':
                [[33.35970299],
                [33.56453657],
                [34.04463461],
                [35.49385124],
                [34.36209713]],
        'UNI':
                [[33.28786349],
                [33.84315466],
                [33.62571556],
                [33.92088263],
                [33.59210872]]
    }

    noatten_values = {
        'PLIP': [[28.57568554],
                [29.89418162],
                [28.20150555],
                [29.74166515],
                [29.18278191]],
        'ProvGigaPath': [[33.32756371],
            [32.95985017],
            [32.40131981],
            [32.75990765],
            [32.27114829]],
                    'CONCH':
            [[32.7633769 ],
            [34.12984309],
            [32.72588764],
            [33.79433179],
            [33.63733268]],
                    'UNI':
            [[33.0650199 ],
            [33.75186445],
            [32.61928498],
            [33.1197163 ],
            [32.48041149]]
    }

    keys = ['PLIP', 'ProvGigaPath', 'CONCH', 'UNI']

    plt.close('all')
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i, key in enumerate(keys):
        ii,jj=i%2, i//2
        ax = axes[ii,jj]

        val0 = np.array(atten_values[key])
        val1 = np.array(noatten_values[key])

        ax.plot(val0, marker='.', color='r', ls='-', label='w attention')
        ax.plot(val1, marker='.', color='g', ls='-', label='w/o attention')
        ax.title.set_text(key)
        ax.set_xticks([0,1,2,3,4],[f'run{i}' for i in range(5)])
    fig.suptitle('Sum of scores on 5 independent runs', fontsize=16)
    plt.tight_layout()
    plt.savefig('/Users/zhongz2/down/a.png')

    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    means = [[np.array(atten_values[k]).mean() for k in keys],
        [np.array(noatten_values[k]).mean() for k in keys]]
    ax.plot(means[0], marker='.', color='r', ls='-', label='w attention')
    ax.plot(means[1], marker='.', color='g', ls='-', label='w/o attention')
    plt.xticks([i for i in range(len(keys))], keys)
    plt.legend()
    plt.title('Mean scores with 5 independent runs on TCGA test set')
    plt.tight_layout()
    plt.savefig('/Users/zhongz2/down/b.png')

def check_best_split_v2():

    results_dir = 'results_20240724_e100'
    results_dir = 'results_20241128_e100_noattention'
    # for backbone in ['mobilenetv3', 'CLIP', 'PLIP', 'ProvGigaPath', 'CONCH', 'UNI']: 
    for backbone in ['PLIP', 'ProvGigaPath', 'CONCH', 'UNI']: 
        print(f'\nbegin {backbone}')
        for proj_name in ['TCGA-ALL2']:
            task_types = ['cls', 'reg']
            splits_dir = f'/Volumes/data-1/temp29/debug/{results_dir}/ngpus2_accum4_backbone{backbone}_dropout0.25'
            subset = 'test'

            all_results = []
            all_tasks = []
            for task_type in task_types:  # no 'cls'
                if task_type == 'cls':
                    if proj_name == 'TCGA-ALL2':
                        task_names = list(CLASSIFICATION_DICT.keys())
                    else:
                        raise ValueError('error project name')
                elif task_type == 'reg':
                    task_names = REGRESSION_LIST
                else:
                    task_names = None

                for task_name in task_names:
                    if task_type == 'cls':
                        accu_col = [task_name + '_auc_weighted']
                    elif task_type == 'reg':
                        accu_col = [task_name + '_mse', task_name + '_r2score', task_name + '_pearsonr_corr',
                                    task_name + '_spearmanr_corr']  # pearmanr_corr should be pearsonr
                        accu_col = [task_name + '_spearmanr_corr']  # pearmanr_corr should be pearsonr
                    elif task_type == 'surv':
                        accu_col = ['c_index']
                    else:
                        accu_col = None
                    select_columns = accu_col

                    results = []
                    for split in range(5):
                        filename1 = '{}/split_{}/{}_log.csv'.format(splits_dir, split, subset)
                        df = pd.read_csv(filename1, low_memory=False)
                        if select_columns[0] not in df:
                            print('no columns')
                            continue
                        df11 = df[select_columns]
                        if df11.isnull().values.any():
                            print('nan existed')
                            df11 = df11.fillna(df11.max(axis=0))
                        results.append(df11.values)
                    if len(results) == 0:
                        print('no results')
                        continue
                    results = np.array(results)
                    all_results.append(results)
                    all_tasks.append(f'{task_type}_{task_name}_{select_columns[0]}')

            all_results1 = np.array(all_results)  # num_tasks x num_splits(5) x num_epochs(100)
            best_epoch = np.argmax(np.sum(all_results1,axis=0).mean(axis=0)) # BRCA: 54, PanCancer: 22
            best_split = np.argmax(all_results1.sum(axis=0)[:, best_epoch])
            print('{}, best_epoch: {}, best_split: {}, mean value: {}, \n{}'.format(
                proj_name, best_epoch, best_split,
                all_results1.sum(axis=0).mean(axis=0)[best_epoch],
                all_results1.sum(axis=0)[:, best_epoch]))  # the sum values for 5 splits in the best epoch



def create_pdf_preserve_quality_300dpi(list0, list1, output_path="output.pdf"):
    dpi = 600
    pt_per_inch = 72  # 1 inch = 72 points

    c = canvas.Canvas(output_path)

    for img0_path, img1_path in zip(list0, list1):
        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)

        # Use original pixel size scaled to 300 DPI
        img0_width_pt = img0.width * pt_per_inch / dpi
        img0_height_pt = img0.height * pt_per_inch / dpi
        img1_width_pt = img1.width * pt_per_inch / dpi
        img1_height_pt = img1.height * pt_per_inch / dpi

        page_width = img0_width_pt + img1_width_pt
        page_height = max(img0_height_pt, img1_height_pt)

        c.setPageSize((page_width, page_height))

        # Align images to the bottom
        c.drawImage(ImageReader(img0), 0, 0, width=img0_width_pt, height=img0_height_pt)
        c.drawImage(ImageReader(img1), img0_width_pt, 0, width=img1_width_pt, height=img1_height_pt)

        c.showPage()

    c.save()


def _save_pdf(list0, list1, output_path="output.pdf"):


    # left_path = os.path.join(save_dir, f"bottom.png")
    # right_path = os.path.join(save_dir, f"top.png")

    # if not (os.path.exists(left_path) and os.path.exists(right_path)):
    #     print(f"Missing image(s), skipping.")
    #     return False

    # left_img = Image.open(left_path)
    # right_img = Image.open(right_path)

    output_pdf = output_path # os.path.join(save_dir, '..', f'{svs_prefix}.pdf')
    page_width, page_height = landscape(A4)

    c = canvas.Canvas(output_pdf, pagesize=landscape(A4))
    padding = 1 # 20
    title_height = 0 # 30
    font_size = 16
    font_name = "Helvetica"

    for left_path, right_path in zip(list0, list1):
        left_img = Image.open(left_path)
        right_img = Image.open(right_path)

        # Calculate max dimensions per side
        max_width = (page_width - padding)/2 # (page_width - 3 * padding) / 2
        max_height = page_height - padding # page_height - title_height - 4 * padding

        def get_scaled_dimensions(img):
            ratio = min(max_width / img.width, max_height / img.height, 1.0)
            return int(img.width * ratio), int(img.height * ratio)

        left_w, left_h = get_scaled_dimensions(left_img)
        right_w, right_h = get_scaled_dimensions(right_img)

        left_x = padding
        right_x = padding * 2 + max_width
        y = (page_height - max(left_h, right_h)) / 2 #  - 10

        # Draw titles
        # c.setFont(font_name, font_size)
        # c.drawCentredString(page_width/2, page_height-padding, svs_prefix + " ({})".format(cancer_type))
        # c.drawCentredString(page_width/2, page_height-2*padding, os.path.basename(dir_path))

        # total_width = sum(c.stringWidth(word + "  ", font_name, font_size) for word in texts)
        # start_x = (page_width - total_width) / 2
        # xx = start_x
        # yy = page_height-2*padding
        # for word, color in zip(texts, colors):
        #     c.setFillColor(color)
        #     c.drawString(xx, yy, word)
        #     # Add spacing after the word based on its width
        #     xx += c.stringWidth(word + "  ", font_name, font_size)
        
        # c.setFillColor(black)
        # # c.drawCentredString(left_x + left_w / 2, page_height - 3*padding,   f"             Top patches (attention_score<{thres1:.2f})")
        # # c.drawCentredString(right_x + right_w / 2, page_height - 3*padding, f"             Top patches (attention_score>{thres2:.2f})")
        # c.drawCentredString(left_x + left_w*1/6+left_w*5/6 / 2, page_height - 3*padding,   f"Top patches (attention_score<{thres1:.2f})")
        # c.drawCentredString(right_x + right_w*1/6+right_w*5/6 / 2, page_height - 3*padding, f"Top patches (attention_score>{thres2:.2f})")

        # Draw images without resizing them beforehand
        c.drawImage(left_path, left_x, y, width=left_w, height=left_h, preserveAspectRatio=True, mask='auto')
        c.drawImage(right_path, right_x, y, width=right_w, height=right_h, preserveAspectRatio=True, mask='auto')

        c.showPage()

    c.save()
    # print(f"PDF saved to {output_pdf}")
    # image = convert_from_bytes(c.getpdfdata())[0]
    # image.save(output_pdf.replace('.pdf', '.png'))
    # drawToFile(c, output_pdf.replace('.pdf', '.png'), fmt='PNG')
    

def main(results_dir, task_types=['cls', 'reg']):

    proj_name = 'TCGA-ALL2'
    per_cancer = 0
    num_gpus = 2

    epoch_step = 2

    label_names = {
        'train': 'TCGA (train)',
        'test': 'TCGA (test)',
        'outside_test0': 'TransNEO (external test)',
        'outside_test1': 'METABRIC (external test)'
    }
    if per_cancer and proj_name == 'TCGA-ALL2':
        all_sites = PAN_CANCER_SITES
        subsets = ['train', 'test']
    else:
        all_sites = [proj_name]
        # subsets = ['train', 'test1', 'test2']  # here, train is the combination of train and val
        subsets = ['train', 'test', 'outside_test0', 'outside_test1']

    # task_types = ['cls', 'reg']
    # task_types = ['loss']
    accum_iters = [4]
    network_dims = {   # only for TCGA-ALL2
        # 'mobilenetv3': 1280,
        # 'CLIP': 512,
        'PLIP': 512,
        'ProvGigaPath': 1536,
        'CONCH': 512,
        'UNI': 1024
    }
    save_postfix = '_'.join(task_types)

    save_root = '/Users/zhongz2/down/figures_20240801_e50_top3'
    save_root = '/Users/zhongz2/down/figures_20240830_e100_top3'
    save_root = '/Users/zhongz2/down/figures_20240902_e100_top4' # Add UNI 
    save_root = '/Users/zhongz2/down/figures_20241129_e100_top4' # Add UNI 

    sub_epochs = [1]
    # results_dir = 'results_20241128_e100_noattention'
    # results_dir = 'results_20240724_e100'
    save_root = f'/Users/zhongz2/down/temp_20250419/train_figures_{results_dir}_{save_postfix}' # Add UNI 
    save_root = f'/Users/zhongz2/down/temp_20250813/train_figures_{results_dir}_{save_postfix}' # Add UNI 
    if os.path.isdir(save_root):
        shutil.rmtree(save_root, ignore_errors=True)
    os.makedirs(save_root, exist_ok=True)

    for site_id, site_name in enumerate(all_sites):
        print('begin {}'.format(site_name))

        all_img_paths = {}

        for subset in subsets:

            if per_cancer:
                pdfsavefilename = '{}/{}_{}_all_results.pdf'.format(save_root, subset, site_name)
            else:
                pdfsavefilename = '{}/{}_all_results.pdf'.format(save_root, subset)
            if os.path.exists(pdfsavefilename):
                shutil.rmtree(pdfsavefilename, ignore_errors=True)
            pdf_file_handle = PdfPages(pdfsavefilename)

            all_mean_results = {}
            all_ranks = {}
            all_best_scores = {}
            prefixes = None
            all_results = {}

            temp_img_paths = []
            for task_type in task_types:  # no 'cls'
                if task_type == 'cls':
                    if proj_name == 'TCGA-ALL2':
                        task_names = list(CLASSIFICATION_DICT.keys())
                    else:
                        raise ValueError('error project name')
                elif task_type == 'reg':
                    task_names = REGRESSION_LIST
                elif task_type == 'loss':
                    task_names = list(CLASSIFICATION_DICT.keys()) + REGRESSION_LIST
                else:
                    task_names = None

                all_ranks[task_type] = {}
                all_best_scores[task_type] = {}
                all_results[task_type] = []
                for task_name in task_names:
                    if task_type == 'cls':
                        accu_col = [task_name + '_auc_weighted']
                    elif task_type == 'reg':
                        accu_col = [task_name + '_mse', task_name + '_r2score', task_name + '_pearsonr_corr',
                                    task_name + '_spearmanr_corr']  # pearmanr_corr should be pearsonr
                        accu_col = [task_name + '_spearmanr_corr']  # pearmanr_corr should be pearsonr
                    elif task_type == 'surv':
                        accu_col = ['c_index']
                    elif task_type == 'loss':
                        accu_col = [task_name]
                    else:
                        accu_col = None
                    select_columns = accu_col

                    if per_cancer:
                        save_dir = os.path.join(save_root, subset, 'PerCancer', site_name, task_type)
                    else:
                        save_dir = os.path.join(save_root, subset, task_type)
                    os.makedirs(save_dir, exist_ok=True)

                    mean_results = []
                    std_results = []

                    filenames = []
                    prefixes = []
                    valid_iiis = []
                    all_results_ = []
                    for accum_iter in accum_iters:
                        for backbone in network_dims.keys():
                            filenames.append(
                                # '/Volumes/data-1/temp29/debug/results_20240724/ngpus{}_accum{}_backbone{}_dropout0.25'.format(
                                #     num_gpus, accum_iter, backbone
                                # )
                                '/Volumes/data-1/temp29/debug/{}/ngpus{}_accum{}_backbone{}_dropout0.25'.format(
                                   results_dir, num_gpus, accum_iter, backbone
                                )
                            )
                            prefixes.append(
                                '{}_{}'.format(backbone, accum_iter)
                            )

                    print('filenames', filenames)

                    for iii, filename in enumerate(filenames):
                        results = []

                        for split in range(5):
                            if per_cancer:
                                filename1 = filename + '/split_{}/{}/{}/{}_log.csv'.format(split, subset,
                                                                                           site_name,
                                                                                           subset)
                            else:
                                filename1 = filename + '/split_{}/{}_log.csv'.format(split, subset)
                            if not os.path.exists(filename1):
                                print(filename1)
                                raise ValueError('check it')
                            else:
                                df = pd.read_csv(filename1)
                                if select_columns[0] not in df:
                                    continue
                                df11 = df[select_columns]
                                if df11.isnull().values.any():
                                    df11 = df11.fillna(df11.max(axis=0))
                                results.append(df11.values)

                        if len(results) == 0:
                            continue
                        if len(np.where(np.isnan(results))[0]) > 0:
                            continue

                        #  the following is for the early-stopping
                        validlen = min([results[i].shape[0] for i in range(len(results))])
                        results = [results[i][:validlen] for i in range(len(results))]

                        results = np.stack(results)
                        mean_results.append(np.mean(results, axis=0))  # num_epochs x 2
                        std_results.append(np.std(results, axis=0))
                        valid_iiis.append(iii)
                        all_results_.append(results)

                    if len(mean_results) == 0:
                        continue

                    for iii in valid_iiis:
                        if prefixes[iii] in all_mean_results:
                            all_mean_results[prefixes[iii]] += [mean_results[iii]]
                        else:
                            all_mean_results[prefixes[iii]] = [mean_results[iii]]

                    all_results_ = np.array(all_results_)
                    all_results[task_type].append(all_results_)

                    ranks, best_scores, img_paths = save_figure_to_one_pdf_new2(subset, task_type, task_name, mean_results, std_results,
                                                   select_columns, epoch_step, prefixes, save_dir,
                                                   pdf_file_handle=pdf_file_handle)
                    all_ranks[task_type][task_name] = ranks
                    all_best_scores[task_type][task_name] = best_scores
                    if len(img_paths) > 0:
                        temp_img_paths.extend(img_paths)

            pdf_file_handle.close()

            if not per_cancer:
                from collections import OrderedDict
                savefilename = '{}/{}_best_epoch.png'.format(save_root, subset)
                font_size = 30
                font_size_label = 20
                figure_width = 7
                plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                plt.tick_params(pad = 10)
                fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
                with open(savefilename.replace('.png', '_data.pkl'), 'wb') as fp:
                    pickle.dump(all_mean_results, fp)
                # sortresults={}
                for prefix, all_mean_results_ in all_mean_results.items():
                    all_mean_results_ = all_mean_results[prefix]
                    x = np.array(all_mean_results_).sum(axis=0).flatten()
                    best_epoch = np.argsort(x)[-1] if task_type != 'loss' else np.argsort(x)[0]
                    label = shorten_prefix2(prefix)
                    plt.plot(x, 'o-', label=label, color=COMMON_COLORS[label])

                for prefix, all_mean_results_ in all_mean_results.items():
                    x = np.array(all_mean_results_).sum(axis=0).flatten()
                    best_epoch = np.argsort(x)[-1] if task_type != 'loss' else np.argsort(x)[0]
                    if True: #task_type != 'loss':
                        plt.scatter(best_epoch, x[best_epoch], s=144, c='red')
                    # sortresults[prefix] = x[best_epoch]

                plt.xlabel('Epochs')
                if task_type != 'loss':
                    plt.ylabel('Overall scores')
                else:
                    plt.ylabel('Overall losses')
                plt.title(label_names[subset], fontsize=font_size)
                plt.grid()
                if subset == 'train':
                    leg=plt.legend(loc='lower right' if task_type!='loss' else 'upper right',handlelength=0, handletextpad=0, fancybox=True)
                    for item, text in zip(leg.legend_handles, leg.get_texts()):
                        print(item.get_color(), text)
                        text.set_color(item.get_color())
                        item.set_visible(False)
                # plt.tight_layout()
                plt.savefig(savefilename, bbox_inches='tight', transparent=True, dpi=600)
                plt.savefig(savefilename.replace('.png', '.svg'), bbox_inches='tight', transparent=True, format='svg')
                plt.close()

                temp_img_paths.insert(0, savefilename)

            if not per_cancer and task_type == 'loss':  # for plot loss

                # for loss 
                from collections import OrderedDict
                from matplotlib import pyplot as plt
                from matplotlib.lines import Line2D
                import pickle, os
                import numpy as np

                with open(os.path.join(save_root, subset+'_best_epoch_data.pkl'), 'rb') as fp:
                    all_mean_results = pickle.load(fp)

                savefilename = '{}/{}_best_epoch_loss.png'.format(save_root, subset)
                font_size = 30
                font_size_label = 20
                figure_width = 7
                plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                plt.tick_params(pad = 10)
                fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
                # with open(savefilename.replace('.png', '_data.pkl'), 'wb') as fp:
                #     pickle.dump(all_mean_results, fp)
                # sortresults={}
                for prefix, all_mean_results_ in all_mean_results.items():
                    all_mean_results_ = all_mean_results[prefix]
                    x = np.array(all_mean_results_).sum(axis=0).flatten()
                    cls_loss = np.array(all_mean_results_)[:11].sum(axis=0).flatten()
                    reg_loss = np.array(all_mean_results_)[11:].sum(axis=0).flatten()
                    # best_epoch = np.argsort(x)[-1] if task_type != 'loss' else np.argsort(x)[0]
                    label = shorten_prefix2(prefix)
                    # plt.plot(x, 'o-', label=label, color=COMMON_COLORS[label])
                    plt.plot(cls_loss, 'o-', label=label, color=COMMON_COLORS[label])
                    # plt.plot(reg_loss, '*-', label=label, color=COMMON_COLORS[label])
                    plt.plot(reg_loss, '*-', color=COMMON_COLORS[label])

                if False:
                    for prefix, all_mean_results_ in all_mean_results.items():
                        x = np.array(all_mean_results_).sum(axis=0).flatten()
                        best_epoch = np.argsort(x)[-1] if task_type != 'loss' else np.argsort(x)[0]
                        if True: #task_type != 'loss':
                            plt.scatter(best_epoch, x[best_epoch], s=144, c='red')
                        # sortresults[prefix] = x[best_epoch]

                plt.xlabel('Epochs')
                # if task_type != 'loss':
                #     plt.ylabel('Overall scores')
                # else:
                #     plt.ylabel('Overall losses')
                plt.ylabel('Losses')
                plt.title(label_names[subset], fontsize=font_size)
                plt.ylim([0, 13])
                plt.grid()
                if subset == 'train':
                    # leg=plt.legend(loc='lower right' if task_type!='loss' else 'upper center',fontsize=14,handlelength=0, handletextpad=0, fancybox=True)
                    # for item, text in zip(leg.legend_handles, leg.get_texts()):
                    #     print(item.get_color(), text)
                    #     text.set_color(item.get_color())
                    #     item.set_visible(False)
                    # plt.gca().add_artist(leg)  # keep the first legend

                    # Second legend: markers → loss types
                    style_handles = [
                        Line2D([0], [0], marker='o', color='black', linestyle='-', label="Classification loss"),
                        Line2D([0], [0], marker='*', color='black', linestyle='-', label="Regression loss")
                    ]
                    # plt.legend(handles=style_handles, fontsize=18, loc="best")
                    plt.legend(handles=style_handles, fontsize=24, loc="best")

                # plt.tight_layout()
                plt.savefig(savefilename, bbox_inches='tight', transparent=True, dpi=600)
                plt.savefig(savefilename.replace('.png', '.svg'), bbox_inches='tight', transparent=True, format='svg')
                plt.close()

            all_img_paths[subset] = temp_img_paths

            if False:
                # processing all_ranks
                num_tasks = sum([len(all_ranks[task_type]) for task_type in task_types])  # len(all_ranks['cls']) + len(all_ranks['reg'])
                num_prefixes = len(prefixes)
                scores = np.zeros((num_prefixes, num_tasks), dtype=np.uint16)
                ind = 0
                for task_index, task_type in enumerate(task_types):
                    for ki, key in enumerate(all_ranks[task_type].keys()):
                        sort_inds = list(all_ranks[task_type][key][0].keys())
                        ranks = list(all_ranks[task_type][key][0].values())
                        scores[sort_inds, ind] = ranks
                        ind += 1

                plot_box_plot_for_check(scores, prefixes, title='{}'.format(subset),
                                        save_filename=os.path.join('{}/{}_boxplot.png'.format(save_root, subset)))

                for task_index, task_type in enumerate(task_types):
                    savefilename111 = os.path.join('{}/{}_{}_best_scores.png'.format(save_root, subset, task_type))
                    # fig = plt.figure(figsize=(36, 36) if task_type=='reg' else (16, 16))
                    # ax = fig.add_subplot(111)
                    font_size = 30
                    font_size_label = 20
                    figure_width = 7
                    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                    plt.tick_params(pad = 10)
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figure_width, figure_width), frameon=False)

                    xs = []
                    ys = []
                    color = ['cyan', 'blue', 'green', 'red']
                    for kkks, vvvs in all_best_scores[task_type].items():
                        for kkkk, vvvv in vvvs.items():
                            xs.append(kkkk.replace('_sum_spearmanr_corr', '').replace('_cls_auc_weighted', '').replace('_auc_weighted', '').replace('CLS_', '').replace('_spearmanr_corr', ''))
                            ys.append(vvvv)
                            break

                    xs = np.array(xs)
                    ys = np.array(ys)
                    ind1 = np.where(ys<0.6)[0]
                    ind2 = np.where((ys>=0.6)&(ys<0.7))[0]
                    ind3 = np.where((ys>=0.7)&(ys<0.8))[0]
                    ind4 = np.where(ys>=0.8)[0]
                    y_pos = np.arange(len(xs))
                    for iii, indd in enumerate([ind1, ind2, ind3, ind4]):
                        ax.barh(y_pos[indd], ys[indd], align='center', color=color[iii])
                    ax.set_yticks(y_pos, labels=xs)
                    ax.invert_yaxis()  # labels read top-to-bottom
                    ax.set_xlabel('Tasks')
                    ax.set_title('Spearman Correlation' if task_type == 'reg' else "AUC")

                    plt.grid()
                    plt.legend(loc='lower right')
                    if task_type == 'reg':
                        plt.yticks(fontsize=font_size)
                        plt.xticks(fontsize=font_size)
                    plt.savefig(savefilename111, bbox_inches='tight', transparent=True, dpi=600)
                    plt.savefig(savefilename111.replace('.png', '.svg'), bbox_inches='tight', transparent=True, format='svg')
                    plt.close()

        # import pdb
        # pdb.set_trace()
        savefilename = os.path.join('{}/all_in_one_{}.pkl'.format(save_root, site_name))
        with open(savefilename, 'wb') as fp:
            pickle.dump(all_img_paths, fp) 
        create_pdf_preserve_quality_300dpi(all_img_paths['train'], all_img_paths['test'], 
            output_path=savefilename.replace('.pkl', '.pdf'))
        _save_pdf(all_img_paths['train'], all_img_paths['test'], savefilename.replace('.pkl', '_v2.pdf'))

def plot_loss_curves():

    import sys, os, glob, shutil
    import pandas as pd
    import numpy as np
    import pdb
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_pdf import PdfPages
    from common import PAN_CANCER_SITES, CLASSIFICATION_DICT, REGRESSION_LIST

    root = '/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backboneCONCH_dropout0.25/split_3'
    save_dir = '/data/zhongz2/HERE_loss_figures'
    os.makedirs(save_dir, exist_ok=True)

    losses = {}
    for subset in ['train', 'test']:
        df = pd.read_csv(f'{root}/{subset}_e99_log.csv', index_col=0, low_memory=False)
        cls_cols = [k for k,v in CLASSIFICATION_DICT.items() if k in df]
        reg_cols = [k for k in REGRESSION_LIST if k in df]
        df1 = df[cls_cols+reg_cols]
        losses[subset] = df1

    fig, ax = plt.subplots()
    for subset in ['train', 'test']:
        plt.plot(losses[subset].sum(axis=1).values)
    plt.savefig(f'{save_dir}/losses.png')
    plt.close('all')

if __name__ == '__main__':

    if False:
        for results_dir in ['results_20240724_e100']:
            for task_types in [['loss']]:
                main(results_dir, task_types=task_types)
    else:
        for results_dir in ['results_20240724_e100', 'results_20241128_e100_noattention']:
            for task_types in [['cls', 'reg'], ['loss']]:
                main(results_dir, task_types=task_types)

