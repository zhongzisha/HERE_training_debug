
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from gurobipy import Model, GRB
from PIL import Image

import sys,os,shutil,glob,json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

MORANDI_colors = [
    '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', 
    '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', 
    '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690'
    ]

# def hex_to_rgb(value):
#     """Convert a hex color to an RGB tuple."""
#     value = value.lstrip("#")
#     return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


# def rgb_to_hex(rgb):
#     return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


# def euclidean_distance(color1, color2):
#     """Calculate the Euclidean distance between two RGB colors."""
#     return int(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))


# def find_next_color(palette, remaining_colors):
#     """Find the color from remaining_colors that maximizes the average distance to the current palette."""
#     max_avg_distance = 0
#     next_color = None
#     for color in remaining_colors:
#         print(color, palette)
#         avg_distance = sum(
#             euclidean_distance(hex_to_rgb(color), hex_to_rgb(p)) for p in palette
#         ) / len(palette)
#         if avg_distance > max_avg_distance:
#             max_avg_distance = avg_distance
#             next_color = color
#     return next_color


# def sort_colors(colors):
#     """This is only to demonstrate the algorithm to get the ordered colors."""
#     current_palette = colors[:6]

#     remaining_colors = [color for color in colors if color not in current_palette]
#     for _ in range(len(remaining_colors)):
#         next_color = find_next_color(current_palette, remaining_colors)
#         current_palette.append(next_color)
#         remaining_colors.remove(next_color)
#     return current_palette


# def get_colors(dir: str, num: int = 18, delta: int = 1500):
#     """Get a list of colors from the Morandi palette."""
#     if dir is None:
#         return colors[:num]

#     color_counts = defaultdict(int)
#     for filename in os.listdir(dir):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             filepath = os.path.join(dir, filename)
#             print(f"Processing {filepath}")
#             with Image.open(filepath) as img:
#                 img = img.resize((100, 100))
#                 img = img.convert("RGB")
#                 color_sig = img.getcolors(img.size[0] * img.size[1])
#                 for count, color in color_sig:
#                     matched = False
#                     for existing_color in color_counts:
#                         distance = euclidean_distance(existing_color, color)
#                         if distance < delta:
#                             color_counts[existing_color] += count
#                             matched = True
#                             break
#                     if not matched:
#                         color_counts[color] = count

#     top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:num]
#     top_colors_hex = [(rgb_to_hex(color), count) for color, count in top_colors]
#     return [color for color, _ in top_colors_hex]


# def group_colors(colors, num_per_group):
#     """Group colors into groups of fixed size such that the sum of the distances between each pair of colors in the same group is maximized."""

#     # Create a new model
#     m = Model("color_grouping")

#     # Create variables
#     x = [
#         [m.addVar(vtype=GRB.BINARY, name=f"{i}-{j}") for j in range(len(num_per_group))]
#         for i in range(len(colors))
#     ]

#     # Set objective
#     objective = 0
#     for i in range(len(colors)):
#         for j in range(i + 1, len(colors)):
#             for k in range(len(num_per_group)):
#                 objective += (
#                     x[i][k]
#                     * x[j][k]
#                     * euclidean_distance(hex_to_rgb(colors[i]), hex_to_rgb(colors[j]))
#                 )
#     m.setObjective(objective, GRB.MAXIMIZE)
#     m.setParam("TimeLimit", 5 * 60)

#     # Add constraints
#     for i in range(len(colors)):
#         m.addConstr(sum(x[i]) <= 1)  # each color can only be at most in one group

#     for j in range(len(num_per_group)):
#         m.addConstr(
#             sum(x[i][j] for i in range(len(colors))) == num_per_group[j]
#         )  # each group has a fixed number of colors

#     # Optimize model
#     m.optimize()

#     # Check if a solution exists
#     if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
#         (
#             print("Warning: suboptimal solution found")
#             if m.status == GRB.SUBOPTIMAL
#             else None
#         )
#         result = [[] for _ in range(len(num_per_group))]
#         for i in range(len(colors)):
#             for k in range(len(num_per_group)):
#                 if x[i][k].x == 1:
#                     result[k].append(colors[i])
#         return result
#     else:
#         raise Exception("No solution found")



# colors = sort_colors(get_colors(None, 18))
# # sns.palplot(colors)
# # plt.title("Giorgio Morandi's palette")
# # plt.show()

# ## Example 2: getting grouped K-colors
# groups = group_colors(colors, [2, 2, 2, 1, 2, 2, 3, 2, 1])

# 'RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'UNI'
# ['HERE_PLIP', 'PLIP', 'ProvGigaPath', 'CONCH', 'HERE_ProvGigaPath', 'HERE_CONCH', 'UNI', 'HERE_UNI']
# palette = [
#     '#008080', '#029370', 'purple', 'mediumpurple', 'blue', 'royalblue', 'gray', 'lightgray',
# ]
all_colors = {
    'HERE': '#686789', 'Original': '#686789', 'CONCH':'#686789', 
    'PLIP': '#7D7465', 
    # 'PLIP': '#E5E2B9', 
    'Yottixel': '#BEB1A8', 'KimiaNet': '#BEB1A8', 
    # 'SISH': '#A79AEE', # '#A79A89', 
    'UNI': '#8A95A9', 
    # 'CLIP': '#ECCED0', 
    'MobileNetV3': '#B7AF70', 'mobilenetv3': '#B7AF70', #'#B77F70', 
    # 'ProvGigaPath': '#E8D3C0', 
    'SISH': '#7B8B81', 'DenseNet': '#7B8B81', #'#7A8A71',  #'#7A8A71', 
    # 'UNI': '#789798', 
    'RetCCL': '#B57C82', 
    'CLIP': '#9F885E', # '#9FABB9', 
    # 'ProvGigaPath': '#B0B1B6', 
    'ProvGigaPath': '#99857E', 
    # 'UNI': '#88878D', 
    # 'UNI': '#91A0A5', 
    'HIPT': '#9AA655',  #'#9AA690'
}
SAVE_ROOT='/Users/zhongz2/down/temp_20250207'
# for f in glob.glob(os.path.join(SAVE_ROOT, '*.xlsx')):
#     shutil.rmtree(f, ignore_errors=True)


# 20250113
def plot_jinlin_evaluation_boxplots():

    # 20240831 update different colors for R0 vs R2|R4, HERE_CONCH
    import os
    import numpy as np
    import pandas as pd
    # from matplotlib import pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import ranksums, wilcoxon
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.colors
    import matplotlib.lines
    from matplotlib.transforms import Bbox, TransformedBbox
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.image import BboxImage
    from matplotlib.patches import Circle
    from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                    AnnotationBbox)
    from matplotlib.cbook import get_sample_data
    from statannotations.Annotator import Annotator

    def cohend(d1, d2) -> pd.Series:
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = np.mean(d1, axis=0), np.mean(d2, axis=0)
        # return the effect size
        return (u1 - u2) / s


    # excel_filename = '/Users/zhongz2/down/HERE/Tables/refined.xlsx'
    # df = pd.read_excel(excel_filename, index_col=0)
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0426.xlsx'
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0507.xlsx'
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0508.xlsx'
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0522.xlsx'
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0831.xlsx'
    df = pd.read_excel(excel_filename)

    # 20241213 Jinin evaluation 6 cases
    # df1 = pd.read_excel('/Users/zhongz2/down/hidare_result  6 METHODS 5-6 CASES.xlsx', sheet_name='CombinedZZS')

    # 20241215 Jinin evaluation 6 cases
    df1 = pd.read_excel('/Users/zhongz2/down/hidare_result  6 METHODS 5-6 CASES_20241215.xlsx', sheet_name='CombinedZZS')
    df1 = df1.groupby('query').agg({'RetCCL': 'mean', 'Yottixel': 'mean', 'SISH': 'mean'})
    # 20250115 Jinlin evaluation 101 cases for 3 methods
    df11 = pd.read_excel('/Users/zhongz2/down/refined Ver20250111.xlsx', sheet_name='CombinedZZS')
    for method in ['RetCCL', 'SISH', 'Yottixel']:
        vals = []
        for v in df11[method].values:
            if isinstance(v, str):
                m = np.array([float(vv) for vv in v.split(',')]).mean()
            else:
                m = v
            vals.append(m)
        print(vals)
        df11[method] = vals
    df1 = df11.set_index('query')
    df2 = df1.merge(df, left_on='query', right_on='query', how='inner').reset_index()

    df = df2

    if 'AdaptiveHERE' not in df.columns:
        # get Adaptive HERE score
        df['AdaptiveHERE'] = df[['r0', 'r2', 'r4']].fillna(-1).max(axis=1)
        df.to_excel(excel_filename)

    df['r2_r4'] = df[['r2', 'r4']].fillna(-1).max(axis=1)

    sites = df['site'].value_counts().index.values
    site_mappers = {kk:kk for kk in sites[:10]}
    site_mappers.update({kk:'others' for ii,kk in enumerate(sites) if ii>=10})
    df['tissue site'] = df['site'].values
    df['tissue site'] = df['tissue site'].map(site_mappers)
    sites = ['lung', 'liver', 'ovary', 'stomach', 'breast', 'lymph node', 'soft tissue', 'testis', 'kidney', 'colon', 'others']


    compared_method = 'AdapHERECONCH'
    data0 = df[[compared_method]]
    data1 = df[['WebPLIP']]
    res = ranksums(data1, data0)
    zscores = res.statistic 
    reject, pvals_corrected, alphacSidakfloat, alphacBonffloat = multipletests(res.pvalue, method='fdr_bh')
    HERE_wins = 100*len(np.where(df[compared_method]>df['WebPLIP'])[0])/len(df)


    def hex_to_rgb(value):
        """Convert a hex color to an RGB tuple."""
        value = value.lstrip('#')
        return tuple(int(value[i:i+2], 16)/255. for i in (0, 2, 4))
    # groups = ['structure']
    COLOR_PALETTES={
        'structure': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'cell type':  [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'cell shape': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'cytoplasm': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'label': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ]
    }

    for k,v in COLOR_PALETTES.items():
        newv = []
        for vv in v:
            if isinstance(vv, str) and '#' in vv:
                newv.append(hex_to_rgb(vv))
            else:
                newv.append(vv)
        # newv = [(int(vv[0]*255), int(vv[1]*255), int(vv[2]*255)) for vv in newv]
        COLOR_PALETTES[k] = newv


    save_root = f'{SAVE_ROOT}/jinlin_evaluation'
    save_root = f'{SAVE_ROOT}/Fig4 evaluation results'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)

    writer = pd.ExcelWriter(os.path.join(SAVE_ROOT, 'Fig3&4.xlsx'))
    df11=df.copy()
    group_names = {
        'structure': 'tissue structure',
        'cell type': 'cell type',
        'cell shape': 'cellular shape',
        'cytoplasm': 'cytoplasm',
        'label': 'tissue composition',
        'tissue site': 'tissue site'
    }
    df11 = df11[['query'] +list(group_names.keys())+['WebPLIP', 'RetCCL', 'Yottixel', 'SISH', compared_method]]
    cols_dict = group_names.copy()
    cols_dict.update({compared_method: 'HERE', 'WebPLIP': 'PLIP'})
    df11=df11.rename(columns=cols_dict)
    df11.to_excel(writer, sheet_name='Fig 3, Fig 4b,4c HERE101')

    if True: # box plot with p-values
        df2 = df[[compared_method, 'WebPLIP', 'RetCCL', 'Yottixel', 'SISH']].rename(columns={compared_method: 'HERE', 'WebPLIP': 'PLIP'})
        df3 = pd.melt(df2, value_vars=['RetCCL', 'Yottixel', 'SISH', 'PLIP', 'HERE'], var_name='method', value_name='score')
        df3['court'] = [i for i in range(len(df2))]*5

        order = ['PLIP', 'Yottixel', 'RetCCL', 'SISH', 'HERE']

        if True: 
            pvalues = {}
            for method in ['RetCCL', 'Yottixel', 'SISH', 'PLIP']:
                U, pvalue = wilcoxon(df2[method].values, df2['HERE'].values, alternative='two-sided')
                pvalues[method] = pvalue

            font_size = 30
            figure_height = 8
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=True)
            ax = plt.gca()

            num_group = 1
            palette = [(0, 0, 0), (0, 0, 0)]
            g=sns.boxplot(data=df3, x="method", y="score", order=order, showfliers=False, palette=all_colors, ax=ax, linewidth=2) 

            g.set(ylabel=None)
            g.set(xlabel=None)

            for i,box in enumerate([p for p in g.patches if not p.get_label()]): 
                color = box.get_facecolor()
                box.set_edgecolor(color)
                box.set_facecolor((0, 0, 0, 0))

                # iterate over whiskers and median lines
                # for j in range(5*i,5*(i+1)):
                #     g.lines[j].set_color(color) 

            plt.ylim([0.5, 5.5])  
            plt.yticks(ticks=[1, 2, 3, 4, 5], labels=['(dissimilar) 1', '(slightly dissimilar) 2', '(similar) 3', '(highly similar) 4', '(indistinguishable) 5'])
            # sns.despine(top=False, right=False, bottom=False, left=False, ax=g)
            sns.despine(top=True, right=True, bottom=False, left=False, ax=g)
            # g=sns.stripplot(data=df3, x="method", y="score", legend=False, marker="$\circ$", ec="face", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
            g=sns.stripplot(data=df3, x="method", y="score", order=order, legend=False, marker="$\circ$", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
            g.set(ylabel='Expert score')
            g.set(xlabel=None)
            g.set_xticklabels(order, rotation=25, ha="right")#, va='center', rotation_mode='anchor')

            # pairs = [(), (), (), ()]
            # # Annotate the plot
            # annotator = Annotator(ax, pairs, data=df3, x="method", y="score")
            # annotator.configure(test='t-test_ind', text_format='simple', loc='inside')
            # annotator.apply_and_annotate()

            # Annotate p-value
            for ii, method in enumerate(order[:-1]):
                x1, x2 = ii, 4  # x-coordinates of the groups
                y, h, col = df3['score'].max() + 0.1, 0.05, 'k'  # y-coordinate, height, color
                # y += (ii+1)*h  # (4 - ii) * h + 0.5
                # h *= (4-ii)
                y += (4-ii-1)*0.45
                # y += (ii+1)*0.1
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col, clip_on=False)
                # ax.text((x1 + x2) * .5, y + h, "p = {:.3e}".format(pvalues[method]), ha='center', va='bottom', color=col)
                ax.text((x1 + x2) * .5, y + h, "{:.3E}".format(pvalues[method]), ha='center', va='bottom', color=col, fontsize=28, clip_on=False)

            # plt.tight_layout()

            # for i in range(1):
            #     for p1, p2 in zip(g.collections[i].get_offsets().data, g.collections[i+num_group].get_offsets().data):
            #         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.5)

            # plt.text(0, 5.15, 'P={:.2e}'.format(pvalue), fontsize=30, color=(0, 0, 0))
            plt.savefig(f'{save_root}/overall_boxplot2.png', bbox_inches='tight', transparent=True, format='png')
            plt.savefig(f'{save_root}/overall_boxplot2.svg', bbox_inches='tight', transparent=True, format='svg')
            df3.to_excel(f'{save_root}/overall_boxplot2.xlsx')
            plt.close()

            df4 = df2[['RetCCL', 'Yottixel', 'SISH', 'PLIP', 'HERE']]
            df4.insert(0, column='query', value=df['query'].values)

    # for indexed pixels
    if False:
        results_dirs = {
            'Yottixel': '/Volumes/data-1/PSC_Yottixel/FEATURES/DATABASE/BOBS/NCI/HERE101_1024_v5_debug',
            'SISH': '/Volumes/data-1/PSC_SISH/FEATURES/DATABASE/MOSAICS/NCI/KenData/20x/search_results_1024_v5_debug/',
            'RetCCL': '/Volumes/data-1/PSC_Yottixel/FEATURES/DATABASE/MOSAIC/NCI/HERE101/RetCCL_feats_yottixel_mosaic_v3_1024_v5_cosine_debug/',
        }

        nums = {
            'HERE': 122104273
        }
        pixel_counts = {
            'HERE': 8002225635328,  # 122104273 * 256 * 256
        }


        # Yottixel
        with open(os.path.join(results_dirs['Yottixel'], 'all_feats.pkl'), 'rb') as fp:
            data = pickle.load(fp)
        nums['Yottixel'] = len(data['all_coords'])
        pixel_counts['Yottixel'] = nums['Yottixel'] * 1000 * 1000


        # SISH
        with open(os.path.join(results_dirs['SISH'], 'database_meta.pkl'), 'rb') as fp:
            data = pickle.load(fp)
        nums['SISH'] = sum([len(v) for k, v in data.items()])
        pixel_counts['SISH'] = nums['SISH'] * 1024 * 1024

        # RetCCL
        with open(os.path.join(results_dirs['RetCCL'], 'all_feats.pkl'), 'rb') as fp:
            data = pickle.load(fp)
        nums['RetCCL'] = len(data['all_coords'])
        pixel_counts['RetCCL'] = nums['RetCCL'] * 1024 * 1024

    if True:
        # indexed pixels
        nums = {'HERE': 122104273, 'Yottixel': 413634, 'SISH': 649848, 'RetCCL': 414944, 'PLIP': 2337}
        pixel_counts = {'HERE': 8002225635328,
        'Yottixel': 413634000000,
        'SISH': 681415016448,
        'RetCCL': 435100319744,
        'PLIP': 2337*256*256}

        font_size = 30
        figure_height = 8
        figure_width = 10
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
        ax = plt.gca()

        keys = ['PLIP', 'Yottixel', 'RetCCL', 'SISH', 'HERE']
        values = [pixel_counts[k]/1e12 for k in keys]
        df = pd.DataFrame({'method': keys, 'pixels':values})
        # g=sns.barplot(all_df1, x="method", y="score", hue="method", palette=palette, legend=False)
        g=sns.barplot(df, x="method", y="pixels", hue="method", palette=all_colors, legend=False)
        # g.set_yscale("log")
        g.tick_params(pad=10)
        g.set_xlabel("")
        # g.set_ylabel("Overall performance")
        g.set_ylabel("Number of indexed pixels ($\\times 10^{12}$)")
        # g.set_ylim([0, 1])
        # g.legend.set_title("")
        # g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=10, ha="right")
        # g.legend.remove()
        # g.set_xticklabels(g.get_xticklabels(), fontsize=9)
        # print(name1, g.get_yticklabels())
        # g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
        g.set_xticklabels(g.get_xticklabels(), rotation=15, ha="right")#, va='center', rotation_mode='anchor')
        # for ci, tick_label in enumerate(g.get_xticklabels()):
        #     tick_label.set_color(palette[ci])
        # plt.tight_layout()

        # plt.bar(keys, values)
        # plt.xlabel('Methods')
        # plt.ylabel("Number of pixels ($\\times 10^{12}$)")
        plt.title('Comparison on search space')

        plt.savefig(f'{save_root}/number_of_pixels_comparison.png', bbox_inches='tight', transparent=True, format='png')
        plt.savefig(f'{save_root}/number_of_pixels_comparison.svg', bbox_inches='tight', transparent=True, format='svg')
        plt.savefig(f'{save_root}/number_of_pixels_comparison.pdf', bbox_inches='tight', transparent=True, format='pdf')
        df.to_csv(f'{save_root}/number_of_pixels_comparison.csv')
        df1 = df.copy()
        df1['pixels'] = df1['pixels']*1e12
        df1.to_excel(writer, sheet_name='Fig 4d indexed pixels')
        plt.close('all')

    writer.close()


# 20240708  added CONCH
def main_20240708_encoder_comparision():

    import numpy as np
    import pandas as pd
    import pickle
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cbook import get_sample_data
    from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage, TextArea)
    from matplotlib.patches import Circle
    # sns.set_theme(style="whitegrid")

    root = '/Volumes/data-1/temp_20240801'
    root = '/Volumes/Jiang_Lab/Data/Zisha_Zhong/temp_20240801'
    save_root = '/Users/zhongz2/down/temp_20241218/encoder_comparison'
    save_root = f'{SAVE_ROOT}/Extended Data Fig 1 encoder_comparison'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)
    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
    dataset_names = ['bcss_512_0.8', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names1 = ['BCSS', 'NuCLS', 'PanNuke', 'Kather100K']

    writer = pd.ExcelWriter(os.path.join(SAVE_ROOT, 'Extended Data Fig 1.xlsx'))

    all_dfs = {}
    for di, dataset_name in enumerate(dataset_names): #, 'NuCLS', 'kather100k']:
        methods = ['CLIP', 'YottixelKimiaNet', 'YottixelKimiaNetNoFinetuned', 'YottixelDenseNet', 'YottixelDenseNet224', 'SISH', 'RetCCL', 'MobileNetV3', 'DenseNet121', 'PLIP', 'HiDARE', 'HiDARE1000', 'HIPT']
        label_names = ['CLIP(224)', 'YottixelKimiaNet(1000)', 'YottixelKimiaNetNoFinetuned(1000)', 'YottixelDenseNet(1000)', 'YottixelDenseNet(224)', 'SISH(1024)', 'RetCCL(224)', 'MobileNetV3(224)', 'DenseNet121(224)', 'PLIP(224)', 'HiDARE(224)', 'HiDARE(1000)', 'HIPT(224)']
        methods = ['YottixelKimiaNet', 'YottixelKimiaNetNoFinetuned','SISH', 'RetCCL', 'DenseNet121', 'HIPT','MobileNetV3', 'CLIP', 'PLIP', 'HiDARE_mobilenetv3', 'HiDARE_CLIP', 'HiDARE_PLIP']
        label_names = ['YottixelKimiaNet(1000)', 'YottixelKimiaNetNoFinetuned(1000)', 'SISH(1024)', 'RetCCL(224)', 'DenseNet121(224)', 'HIPT(224)','MobileNetV3(224)', 'CLIP(224)', 'PLIP(224)', 'HiDARE_MobileNetV3(224)', 'HiDARE_CLIP(224)', 'HiDARE_PLIP(224)']
        methods = ['YottixelKimiaNet','SISH', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP']
        label_names = ['Yottixel', 'SISH', 'RetCCL', 'DenseNet', 'HIPT', 'CLIP', 'PLIP', 'HiDARE']
        methods = ['YottixelKimiaNet', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP', 'MobileNetV3', 'ProvGigaPath']
        label_names = ['Yottixel', 'RetCCL', 'SISH', 'HIPT', 'CLIP', 'PLIP', 'HERE', 'MobileNetV3', 'ProvGigaPath']
        methods = ['YottixelKimiaNet', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HiDARE_ProvGigaPath']
        label_names = ['Yottixel', 'RetCCL', 'SISH', 'HIPT', 'CLIP', 'PLIP', 'HERE', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov']
        methods = ['YottixelKimiaNet', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HiDARE_ProvGigaPath', 'CONCH', 'HiDARE_CONCH', 'UNI', 'HiDARE_UNI']
        label_names = ['Yottixel', 'RetCCL', 'SISH', 'HIPT', 'CLIP', 'PLIP', 'HERE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HERE_ProvGigaPath', 'CONCH', 'HERE_CONCH', 'UNI', 'HERE_UNI']
        methods = ['YottixelKimiaNet', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HiDARE_ProvGigaPath', 'CONCH', 'HiDARE_CONCH', 'UNI', 'HiDARE_UNI']
        label_names = ['KimiaNet', 'RetCCL', 'DenseNet', 'HIPT', 'CLIP', 'PLIP', 'HERE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HERE_ProvGigaPath', 'CONCH', 'HERE_CONCH', 'UNI', 'HERE_UNI']
        if True:
            data = []
            dfs = []
            for method, label_name in zip(methods, label_names):
                method1 = 'Yottixel' if method == 'YottixelKimiaNet' else method
                filename = f'{root}/{dataset_name}_{method1}_feats_results1.csv'
                # if not os.path.exists(filename) and 'kather100k' in dataset_name and ('Yottixel' in method or 'SISH' in method):
                #     filename = filename.replace('.csv', '_random100_random100.csv')
                if True:# 'HiDARE_' in method1 or 'ProvGigaPath' in method1 or 'CONCH' in method1:
                    filename = filename.replace(method1, f'{method1}_0')
                if not os.path.exists(filename):
                    print(filename, ' not existed')
                    break
                print(filename)
                df = pd.read_csv(filename, index_col=0)
                if dataset_name == 'NuCLS':
                    df = df.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                data.append(df.values[:2, :])

                df1 = pd.read_csv(filename, index_col=0)
                if dataset_name == 'NuCLS':
                    df1 = df1.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                df1 = df1.T
                df1['method'] = label_name
                df1.index.name = 'label'
                dfs.append(df1)
            df2 = pd.concat(dfs).reset_index()
            df3 = df2.copy()
            df3['dataset_name'] = dataset_names1[di]
            all_dfs[dataset_name] = df3 


            if len(data) != len(methods):
                print('wrong data')
                break

            for jj, name in enumerate(['Acc', 'Percision']):
                name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'
                species = df.columns.values
                penguin_means1 = {
                    method: [float('{:.4f}'.format(v)) for v in data[mi][jj].tolist()]
                    for mi, method in enumerate(methods)
                }
                penguin_means = {
                    method: [v for v in data[mi][jj].tolist()]
                    for mi, method in enumerate(methods)
                }

                # plt.tight_layout()
                df1 =pd.DataFrame(penguin_means)
                df1.columns = label_names
                df2 = df1.T
                df2.columns = species
                df2.to_csv(f'{save_root}/{dataset_name}_{name1}.csv')

    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k'] 
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
    dataset_names = ['bcss_512_0.8', 'NuCLS', 'PanNuke', 'kather100k'] 
    dataset_names1 = ['BCSS',  'NuCLS', 'PanNuke', 'Kather100K']
    xticklabels = {
        'BCSS': ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Fat', 'Plasma', 'Other infil', 'Vessel'],
        'NuCLS': ['Lymphocyte', 'Macrophage', 'Stroma', 'Plasma', 'Tumor'],
        'PanNuke': ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial'],
        'Kather100K': ['Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus', 'Muscle', 'Mucosa', 'Stroma', 'Adeno epithelium']
    }
    palette = sns.color_palette('colorblind')
    palette = [
        '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', 
        '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', 
        '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690'
    ]

    datadata = pd.concat([v for k,v in all_dfs.items()],axis=0).reset_index(drop=True)
    datadata.to_csv(os.path.join(save_root, 'all.csv'))

    datadata11 = datadata.copy()
    datadata11 = datadata11.rename(columns={'Percision':'Precision', 'Acc':'Majority Vote Accuracy'})
    datadata11.to_excel(writer, sheet_name='Ext Fig. 1a,1b encoder')

    #get the ranking
    for name in ['Percision', 'Acc']: 

        name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'
        all_df = None
        all_df_median = None
        for dataset_name in dataset_names:
            df = pd.read_csv(f'{save_root}/{dataset_name}_{name1}.csv', index_col=0)
            dff = df.copy()
            df = df.sum(axis=1)
            if all_df is None:
                all_df = df.copy()
            else:
                all_df += df
            if all_df_median is None:
                all_df_median = dff.copy()
            else:
                all_df_median = pd.concat([all_df_median, dff], axis=1)
        all_df=pd.DataFrame(all_df, columns=['score'])
        all_df = pd.DataFrame(all_df_median.median(axis=1), columns=['score'])
        all_df = all_df.sort_values('score', ascending=False)
        all_df.to_csv(f'{save_root}/ranking_{name1}.csv')

        all_df.index.name = 'method'
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE_PLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov', 'HERE_CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'UNI']
        selected_methods = ['RetCCL', 'HIPT', 'DenseNet', 'CLIP', 'KimiaNet', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'UNI']
        all_df = all_df[all_df.index.isin(selected_methods)].reset_index()
        all_df1 = all_df.copy()
        all_df1['score1'] = np.log(all_df1['score'] - 30)
        all_df1['score2'] = all_df1['score']

        datadata1 = []
        for m in all_df['method'].values:
            datadata1.append(datadata[datadata['method']==m])
        datadata1 = pd.concat(datadata1,axis=0).reset_index(drop=True)

        # boxplot with strip dots
        for dolegend in ['auto', False]:
            plt.close('all')
            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            ax = plt.gca()
            g=sns.boxplot(data=datadata1, x="method",  palette=all_colors, y=name, hue="method", legend=False, ax=ax)  #showfliers=False, 

            if dolegend=='auto':
                # plt.legend(title="(n=43)", loc="upper left", bbox_to_anchor=(1, 1))  # Adjust legend position

                ax2 = g.secondary_yaxis('right')
                ax2.set_yticks(g.get_yticks())
                ax2.set_yticklabels(['' for _ in g.get_yticklabels()], rotation=90, ha="left", va='center', rotation_mode='anchor')
                ax2.tick_params(axis='y', length=0)
                ax2.set_ylabel('(n = {})'.format(len(datadata1['label'].unique())))

            # if dolegend=='auto':
            #     sns.move_legend(
            #         ax, "outside right",
            #         title=None
            #     )
            # g.set(ylabel=None)
            # g.set(xlabel=None)
            # g=sns.stripplot(data=datadata1, palette=[(0,0,0),(0,0,0)],x="method", y=name, legend=False, marker="$\circ$", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
            # g.set(ylabel=name1)
            g.set(ylabel=None)#='Accuracy' if name =='Acc' else 'Average Precision')
            g.set(xlabel=None)
            g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
            g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')

            g=sns.stripplot(data=datadata1, x="method", y=name, jitter=True, color='black', alpha=0.4, ax=ax)
            g.set(ylabel='Mean Majority Vote Accuracy' if name =='Acc' else 'Average Precision')
            g.set(xlabel=None)
            g.set(ylim=[-0.05, 1.05])  
            g.set(yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticklabels=['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            plt.ylim([-0.05, 1.05])
            # g.map_dataframe(sns.stripplot, x="method", y=name, legend=False, dodge=True, 
            #     marker="$\circ$", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)

            plt.savefig(os.path.join(save_root, f'ranking_meanstd_{name1}_strip_legend{dolegend}.png'), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, f'ranking_meanstd_{name1}_strip_legend{dolegend}.svg'), bbox_inches='tight', transparent=True, format='svg')
            plt.savefig(os.path.join(save_root, f'ranking_meanstd_{name1}_strip_legend{dolegend}.pdf'), bbox_inches='tight', transparent=True, format='pdf')
            datadata1.to_csv(os.path.join(save_root, f'ranking_meanstd_{name1}_strip_legend{dolegend}.csv'))
            plt.close()

        # from compute_flops.py
        total_params_and_flops = {'PLIP': (151277313, 4413615360), 'CONCH': (395232769, 17738386944), 'ProvGigaPath': (1134953984, 228217640448), 'UNI': (303350784, 61603111936), 'Yottixel': (7978856, 2865546752), 'SISH': (7978856, 2865546752), 'MobileNetV3': (5483032, 225436416), 'HIPT': (21665664, 4607954304), 'CLIP': (151277313, 4413615360), 'RetCCL': (23508032, 4109464576)}
        # Yottixel --> KimiaNet; SISH --> DenseNet
        total_params_and_flops = {'PLIP': (151277313, 4413615360), 'CONCH': (395232769, 17738386944), 'ProvGigaPath': (1134953984, 228217640448), 'UNI': (303350784, 61603111936), 'KimiaNet': (7978856, 2865546752), 'DenseNet': (7978856, 2865546752), 'MobileNetV3': (5483032, 225436416), 'HIPT': (21665664, 4607954304), 'CLIP': (151277313, 4413615360), 'RetCCL': (23508032, 4109464576)}

        all_df2 = pd.DataFrame(total_params_and_flops).T
        all_df2.columns = ['NumParams', 'FLOPs']
        all_df2 = all_df2.loc[all_df['method'].values]
        all_df2.index.name = 'method'
        all_df2 = all_df2[all_df2.index.isin(selected_methods)].reset_index()
        all_df2 = all_df.merge(all_df2, left_on='method', right_on='method')

        # all_df2['score'] = all_df2['score'] / len(datadata1['label'].unique())

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
        COMMON_COLORS = all_colors
        specific_colors = []
        color_ind = 0
        for color_ind, method_name in enumerate(all_df2['method'].values.tolist()):
            if method_name in COMMON_COLORS:
                specific_colors.append(COMMON_COLORS[method_name])
            else:
                specific_colors.append(palette[color_ind])
        all_df2['color'] = specific_colors
        # marker size
        all_df2['marker_size'] = 10*all_df2['NumParams']/all_df2['NumParams'].min()
        # all_df2['marker_size'] = [7, 4.5, 2, 0.5, 0.5, 0.5, 1, 2, 1]
        # all_df2['marker_size'] = all_df2['marker_size'] * 30

        plt.close()
        font_size = 30
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        # plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
        # g=sns.barplot(all_df2, x="method", y="FLOPs", hue="method", palette=palette, legend=False) 
        g=sns.barplot(all_df2, x="method", y="FLOPs", hue="method", palette=all_colors, legend=False) 
        g.tick_params(pad=10)
        g.set_xlabel("")
        g.set_ylabel(r"Total FLOPs ($\times 10^{11}$)")
        g.ticklabel_format(style='sci', axis='y')
        # g.set_ylim([0, 1])
        # g.legend.set_title("")
        # g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=10, ha="right")
        # g.legend.remove()
        # g.set_xticklabels(g.get_xticklabels(), fontsize=9)
        print(name1, g.get_yticklabels())
        g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
        g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')
        
        # for ci, tick_label in enumerate(g.get_xticklabels()):
        #     tick_label.set_color(palette[ci])
        # plt.tight_layout()
        plt.savefig(os.path.join(save_root, f'flops_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'flops_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
        plt.savefig(os.path.join(save_root, f'flops_{name1}.pdf'), bbox_inches='tight', transparent=True, format='pdf')
        all_df2.to_csv(os.path.join(save_root, f'flops_{name1}.csv'))
        all_df2.to_excel(writer, sheet_name='Ext Fig 1c FLOPs')
        plt.close()

        plt.close()
        font_size = 30
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        # plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
        # g=sns.scatterplot(all_df2, x="FLOPs", y="score", hue="method", palette=palette, legend=False)
        # g=sns.scatterplot(all_df2, x="FLOPs", y="score", hue="method", palette=all_df2['color'].values.tolist(), legend=False)
        # g=sns.scatterplot(all_df2, x="FLOPs", y="score", hue="method", palette=[palette[0] for _ in range(len(all_df2))], legend=False)
        g=sns.scatterplot(all_df2, x="FLOPs", y="score", hue="method", palette=['#91A0A5' for _ in range(len(all_df2))], legend=False)
        g.tick_params(pad=10)
        g.set_ylabel("Median scores")
        g.set_xlabel(r"Total FLOPs")
        g.ticklabel_format(style='sci', axis='x')
        # g.set_ylim([30, 38])
        # g.set_ylim([15, 20])
        g.set_ylim([0.7, 1.0])
        # g.set_yticklabels([30, 31, 32, 33, 34, 35, 36, 37, 38])
        # g.legend.set_title("")
        # g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=10, ha="right")
        # g.legend.remove()
        # g.set_xticklabels(g.get_xticklabels(), fontsize=9)
        print(name1, g.get_yticklabels())
        # g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
        # g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')
        
        # for ci, tick_label in enumerate(g.get_xticklabels()):
        #     tick_label.set_color(palette[ci])

        if False:
            with open('/Users/zhongz2/down/figures_20240902_e100_top4/test_best_epoch_data.pkl', 'rb') as fp:
                all_mean_results = pickle.load(fp)
            for prefix, all_mean_results_ in all_mean_results.items():
                x = np.array(all_mean_results_).sum(axis=0).flatten()
                best_epoch = np.argsort(x)[-1]
                print(prefix, best_epoch, x[best_epoch])
        # mobilenetv3_4 32 26.966785516013267
        # CLIP_4 97 26.463153515114186
        # PLIP_4 66 29.2551421379361
        # ProvGigaPath_4 39 33.241507276672884
        # CONCH_4 53 34.164964506392046
        # UNI_4: 58 33.653945013803146
        all_df2['marker_size_by_tcga_scores'] = [v*10 for v in [
            33.653945013803146, 33.241507276672884, 34.164964506392046, 29.2551421379361, # UNI, ProvGigaPath, CONCH, PLIP
            15, 15, 10, 10, 10, 10
        ]]
        for scatters in g.collections:
            # scatters.set_sizes(all_df2['marker_size'].values.tolist()) 
            scatters.set_sizes(all_df2['marker_size_by_tcga_scores'].values.tolist()) 

        # ['ProvGigaPath',
        # 'CONCH',
        # 'PLIP',
        # 'Yottixel',
        # 'SISH',
        # 'MobileNetV3',
        # 'HIPT',
        # 'CLIP',
        # 'RetCCL']

        # UNI	37.17489849
        # ProvGigaPath	37.09885172
        # CONCH	35.82475628
        # PLIP	35.52147874
        # Yottixel	35.33588824
        # SISH	32.98901163
        # MobileNetV3	32.40335112
        # HIPT	32.24158473
        # CLIP	31.47365478
        # RetCCL	31.40132146
        # all_df2['y'] = [35.521479+1.2, 37.098852, 35.521479+0.6, 35.521479, 35.521479-0.6, 32.989012, 32.989012-0.6, 32.989012-1.2, 32.989012-1.8, 32.989012-2.4]
        
        # UNI	18.97677166
        # ProvGigaPath	18.96241341
        # CONCH	18.45292947
        # PLIP	18.13056616
        # Yottixel	17.56864238
        # RetCCL	16.74457692
        # SISH	16.71358908
        # MobileNetV3	16.38733908
        # HIPT	16.26037564
        # CLIP	16.01003912
        all_df2['y'] = [18.13056616+1.2, 18.96241341, 18.13056616+0.4, 18.13056616, 18.13056616-0.6, 16.74457692, 16.74457692-0.4, 16.74457692-0.8, 16.74457692-1.2, 16.74457692-1.6]
        all_df2['x'] = [2.5*17738386944, 0.55*228217640448, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944]


        # for median scores
        # UNI	0.965098
        # ProvGigaPath	0.947465
        # CONCH	0.951773
        # PLIP	0.948800
        # Yottixel	0.907067
        # RetCCL	0.748995
        # SISH	0.851233
        # MobileNetV3	0.851467
        # HIPT	0.836171
        # CLIP	0.843533
        # 
        xx=0.95177305-0.005 # from CONCH
        xxx=0.851233 # from DenseNet/SISH
        all_df2['y'] = [xx+0.025, xx, xx-0.02    , 0.94746454, 0.90706667, xxx+0.025, xxx, xxx-0.025, xxx-0.05 , 0.74899496]
        all_df2['x'] = [5*17738386944, 2.5*17738386944, 2.5*17738386944, 0.6*228217640448,2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944]
        all_df2['TCGA(test)'] = pd.NA
        
        for row_ind, row in all_df2.iterrows():
            # offsetbox = TextArea(row['method'], textprops={'fontsize': 14})
            # ab = AnnotationBbox(offsetbox, (row['FLOPs'], row['score']),
            #                     xybox=(20, -20),
            #                     xycoords='data',
            #                     boxcoords="offset points",
            #                     arrowprops=dict(arrowstyle="->"))
            # g.add_artist(ab)
            # g.text(x, row['score'], row['method'], size=24)
            cc = 'black'
            tt = row['method']
            if tt == 'UNI':
                cc = COMMON_COLORS[tt]
                tt = '{} ({:.3f})'.format(tt, 33.653945013803146)
                all_df2.loc[row_ind, 'TCGA(test)'] = 33.653945013803146
            if tt == 'ProvGigaPath':
                cc = COMMON_COLORS[tt]
                tt = '{}\n    ({:.3f})'.format(tt, 33.241507276672884)
                all_df2.loc[row_ind, 'TCGA(test)'] = 33.241507276672884
            if tt == 'CONCH':
                cc = COMMON_COLORS[tt]
                tt = '{} ({:.3f})'.format(tt, 34.164964506392046)
                all_df2.loc[row_ind, 'TCGA(test)'] = 34.164964506392046
            if tt == 'PLIP':
                cc = COMMON_COLORS[tt]
                tt = '{} ({:.3f})'.format(tt, 29.2551421379361)
                all_df2.loc[row_ind, 'TCGA(test)'] = 29.2551421379361
            g.annotate(tt, color=cc, size=18, xy=(row['FLOPs'], row['score']), ha='left', va='center', xytext=(row['x'], row['y']), \
                 arrowprops=dict(facecolor='black', width=1, headwidth=4, shrink=0.15))

        # plt.tight_layout()
        plt.savefig(os.path.join(save_root, f'ranking_vs_flops_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'ranking_vs_flops_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
        plt.savefig(os.path.join(save_root, f'ranking_vs_flops_{name1}.pdf'), bbox_inches='tight', transparent=True, format='pdf')
        all_df2.to_csv(os.path.join(save_root, f'ranking_vs_flops_{name1}.csv'))
        if name=='Percision':
            all_df2.to_excel(os.path.join(SAVE_ROOT, f'Extended Data Fig 2e.xlsx'))
        plt.close()

        # 
        hue_order = all_df['method'].values
        ylims = {
            'BCSS': [0, 1],
            'Kather100K': [0, 1],
            'PanNuke': [0, 1],
            'NuCLS': [0, 1]
        }
        for di, dataset_name in enumerate(dataset_names):
            if dataset_name not in all_dfs:
                continue
            df2 = all_dfs[dataset_name]
            # Draw a nested barplot by species and sex
            plt.close()

            num_labels = len(df2['label'].value_counts())
            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            # g = sns.catplot(
            #     data=df2, kind="bar",
            #     x="label", y=name, hue="method", hue_order=hue_order,
            #     errorbar="sd", palette=palette, height=6,legend=False,aspect=1.5
            # )
            g = sns.catplot(
                data=df2, kind="bar",
                x="label", y=name, hue="method", hue_order=hue_order,
                errorbar="sd", palette=all_colors, height=6,legend=False,aspect=1.5
            )
            sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
            g.ax.yaxis.tick_right()
            g.ax.set_ylim(ylims[dataset_names1[di]])
            g.ax.yaxis.set_label_position("right")
            g.set_axis_labels("", 'Mean Majority Voting Accuracy' if name=='Acc' else "Average precision")
            print(name1, g.ax.get_yticklabels())
            g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode='anchor')
            g.ax.set_xticklabels([xticklabels[dataset_names1[di]][iii] for iii in range(len(g.ax.get_xticklabels()))], rotation=90, ha="right", va='center', rotation_mode='anchor')
            plt.title(dataset_names1[di], fontsize=font_size)
            plt.savefig(os.path.join(save_root, '{}_{}_result1.png'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, '{}_{}_result1.svg'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='svg')
            plt.savefig(os.path.join(save_root, '{}_{}_result1.pdf'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='pdf')
            df2.to_csv(os.path.join(save_root, '{}_{}_result1.csv'.format(dataset_names[di], name1)))
            plt.close()

    #get the ranking (for HERE methods, HERE_PLIP, PLIP, HERE_)
    # ['HERE_PLIP', 'PLIP', 'ProvGigaPath', 'CONCH', 'HERE_ProvGigaPath', 'HERE_CONCH', 'UNI', 'HERE_UNI']
    palette1 = [
        '#008080', '#029370', 'purple', 'mediumpurple', 'blue', 'royalblue', 'gray', 'lightgray',
    ]
    for name in ['Percision', 'Acc']:
        name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'
        all_df = None
        for dataset_name in dataset_names:
            df = pd.read_csv(f'{save_root}/{dataset_name}_{name1}.csv', index_col=0)
            
            df = df.sum(axis=1)
            if all_df is None:
                all_df = df.copy()
            else:
                all_df += df
        all_df=pd.DataFrame(all_df, columns=['score'])
        all_df = all_df.sort_values('score', ascending=False)
        all_df.to_csv(f'{save_root}/HERE_ranking_{name1}.csv')

        all_df.index.name = 'method'
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE_PLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov', 'HERE_CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH']
        selected_methods = ['HERE_PLIP', 'PLIP', 'ProvGigaPath', 'CONCH', 'HERE_ProvGigaPath', 'HERE_CONCH', 'UNI', 'HERE_UNI']
        all_df = all_df[all_df.index.isin(selected_methods)].reset_index()

        plt.close()
        font_size = 30
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        # plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
        g=sns.barplot(all_df, x="method", y="score", hue="method", palette=palette1, legend=False)
        g.tick_params(pad=10)
        g.set_xlabel("")
        # g.set_ylabel("Overall performance")
        g.set_ylabel('Mean Majority Voting Accuracy' if name=='Acc' else "Average precision")
        # g.set_ylim([0, 1])
        # g.legend.set_title("")
        # g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=10, ha="right")
        # g.legend.remove()
        # g.set_xticklabels(g.get_xticklabels(), fontsize=9)
        print(name1, g.get_yticklabels())
        g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
        g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')
        for ci, tick_label in enumerate(g.get_xticklabels()):
            tick_label.set_color(palette1[ci])
        # plt.tight_layout()
        plt.savefig(os.path.join(save_root, f'HERE_ranking_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'HERE_ranking_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
        plt.savefig(os.path.join(save_root, f'HERE_ranking_{name1}.pdf'), bbox_inches='tight', transparent=True, format='pdf')
        all_df.to_csv(os.path.join(save_root, f'HERE_ranking_{name1}.csv'))
        plt.close()

        # 
        hue_order = all_df['method'].values
        ylims = {
            'BCSS': [0, 1],
            'Kather100K': [0, 1],
            'PanNuke': [0, 1],
            'NuCLS': [0, 1]
        }
        for di, dataset_name in enumerate(dataset_names):
            if dataset_name not in all_dfs:
                continue
            df2 = all_dfs[dataset_name]
            # Draw a nested barplot by species and sex
            plt.close()

            num_labels = len(df2['label'].value_counts())
            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            g = sns.catplot(
                data=df2, kind="bar",
                x="label", y=name, hue="method", hue_order=hue_order,
                errorbar="sd", palette=palette1, height=6,legend=False,aspect=1.5
            )
            sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
            g.ax.yaxis.tick_right()
            g.ax.set_ylim(ylims[dataset_names1[di]])
            g.ax.yaxis.set_label_position("right")
            g.set_axis_labels("", 'Mean Majority Voting Accuracy' if name=='Acc' else "Average precision")
            print(name1, g.ax.get_yticklabels())
            g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode='anchor')
            g.ax.set_xticklabels([xticklabels[dataset_names1[di]][iii] for iii in range(len(g.ax.get_xticklabels()))], rotation=90, ha="right", va='center', rotation_mode='anchor')
            plt.title(dataset_names1[di], fontsize=font_size)
            plt.savefig(os.path.join(save_root, 'HERE_{}_{}_result1.png'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, 'HERE_{}_{}_result1.svg'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='svg')
            plt.savefig(os.path.join(save_root, 'HERE_{}_{}_result1.pdf'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='pdf')
            df2.to_csv(os.path.join(save_root, 'HERE_{}_{}_result1.csv'.format(dataset_names[di], name1)))
            plt.close()

    writer.close()

def plot_search_time_tcga_ncidata():

    import sys,os,glob,shutil
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    import pickle
    import matplotlib.colors as mcolors
    import seaborn as sns
    # sns.set_theme(style="whitegrid")
    sns.despine(top=False, right=False)

    root = '/Volumes/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/'
    save_root = f'{SAVE_ROOT}/hashing_comparison2'
    save_root = f'{SAVE_ROOT}/Fig2 hashing_comparison'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)
    # root = '/Users/zhongz2/down'
    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']

    dataset_names = ['bcss_512_0.8', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names1 = ['BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
    
    ITQ_Dims = [32, 64, 128]
    Ms = [8, 16, 32]
    nlists = [128, 256]
    ITQ_Dims = [32, 64]
    Ms = [16, 32]
    nlists = [128, 256]
    faiss_params = [('IndexFlatIP', None)]
    faiss_params.extend(
        [(f'IndexBinaryFlat_ITQ{dd}_LSH', dd) for dd in ITQ_Dims])
    for m in Ms:
        for nlist in nlists:
            faiss_params.append(
                (f'IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8', m, nlist))


    ylims = {
        'BCSS': [0.0, 1.0],
        'Kather100K': [0.0, 1.0],
        'PanNuke': [0.0, 1.0],
        'NuCLS': [0.0, 1.0]
    }
    # fig, axes = plt.subplots(ncols=1, nrows=len(dataset_names), layout='constrained', figsize=(11,18))
    all_dfs = {}
    for di, dataset_name in enumerate(dataset_names): #, 'NuCLS', 'kather100k']:
        # ax = axes[di]
        methods = ['CLIP', 'YottixelKimiaNet', 'YottixelKimiaNetNoFinetuned', 'YottixelDenseNet', 'YottixelDenseNet224', 'SISH', 'RetCCL', 'MobileNetV3', 'DenseNet121', 'PLIP', 'HiDARE', 'HiDARE1000', 'HIPT']
        label_names = ['CLIP(224)', 'YottixelKimiaNet(1000)', 'YottixelKimiaNetNoFinetuned(1000)', 'YottixelDenseNet(1000)', 'YottixelDenseNet(224)', 'SISH(1024)', 'RetCCL(224)', 'MobileNetV3(224)', 'DenseNet121(224)', 'PLIP(224)', 'HiDARE(224)', 'HiDARE(1000)', 'HIPT(224)']
        # methods = ['YottixelKimiaNet', 'YottixelKimiaNetNoFinetuned', 'YottixelDenseNet', 'YottixelDenseNet224', 'RetCCL', 'MobileNetV3', 'DenseNet121', 'PLIP', 'HiDARE', 'HiDARE1000', 'HIPT']
        # label_names = ['YottixelKimiaNet(1000)', 'YottixelKimiaNetNoFinetuned(1000)', 'YottixelDenseNet(1000)', 'YottixelDenseNet(224)', 'RetCCL(224)', 'MobileNetV3(224)', 'DenseNet121(224)', 'PLIP(224)', 'HiDARE(224)', 'HiDARE(1000)', 'HIPT(224)']
        methods = ['YottixelKimiaNet', 'YottixelKimiaNetNoFinetuned','SISH', 'RetCCL', 'DenseNet121', 'HIPT','MobileNetV3', 'CLIP', 'PLIP', 'HiDARE_mobilenetv3', 'HiDARE_CLIP', 'HiDARE_PLIP']
        label_names = ['YottixelKimiaNet(1000)', 'YottixelKimiaNetNoFinetuned(1000)', 'SISH(1024)', 'RetCCL(224)', 'DenseNet121(224)', 'HIPT(224)','MobileNetV3(224)', 'CLIP(224)', 'PLIP(224)', 'HiDARE_MobileNetV3(224)', 'HiDARE_CLIP(224)', 'HiDARE_PLIP(224)']
        
        methods = ['YottixelKimiaNet','SISH', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP']
        label_names = ['Yottixel', 'SISH', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP']
        
        methods = ['YottixelKimiaNet','SISH', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP']
        label_names = ['KimiaNet', 'DenseNet', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP']
        
        fe_methods = ['HiDARE_mobilenetv3', 'HiDARE_CLIP', 'HiDARE_PLIP', 'HiDARE_CONCH', 'HiDARE_ProvGigaPath']
        fe_label_names = ['HERE_MobileNetV3', 'HERE_CLIP', 'HERE_PLIP', 'HERE_CONCH', 'HERE_Prov']
        # fe_methods = ['HiDARE_PLIP']
        # fe_label_names = ['HiDARE']
        methods = [param[0] for param in faiss_params]
        label_names = [param[0] for param in faiss_params]

        all_dfs[dataset_name] = {}
        for fe_method in fe_methods:
            data = []
            dfs = []
            for method, label_name in zip(methods,label_names):
                fe_method1 = 'Yottixel' if fe_method == 'YottixelKimiaNet' else fe_method
                filename = f'{root}/{dataset_name}_{fe_method1}_feats_binary_{method}_results1.csv'
                # if not os.path.exists(filename) and 'kather100k' in dataset_name and ('Yottixel' in fe_method or 'SISH' in fe_method):
                #     filename = filename.replace('.csv', '_random100_random100.csv')
                if True: # 'HiDARE_' in fe_method1:
                    filename = filename.replace(fe_method1, f'{fe_method1}_0')
                if not os.path.exists(filename):
                    print(filename, ' not existed')
                    break
                df = pd.read_csv(filename, index_col=0)
                if dataset_name == 'NuCLS':
                    df = df.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                data.append(df.values[:2, :])

                df1 = pd.read_csv(filename, index_col=0)
                if dataset_name == 'NuCLS':
                    df1 = df1.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                df1 = df1.T
                df1['method'] = label_name
                df1.index.name = 'label'
                dfs.append(df1)
            df2 = pd.concat(dfs).reset_index()
            df3 = df2.copy()
            df3['dataset_name'] = dataset_names1[di] 
            all_dfs[dataset_name][fe_method] = df3


            if len(data) != len(methods):
                print('wrong data')
                break

            # fig, axes = plt.subplots(ncols=1, nrows=2, layout='constrained', figsize=(16,9))
            for jj, name in enumerate(['Acc', 'Percision']):
                # ax = axes[0]
                name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'
                species = df.columns.values
                penguin_means1 = {
                    method: [float('{:.4f}'.format(v)) for v in data[mi][jj].tolist()]
                    for mi, method in enumerate(methods)
                }
                penguin_means = {
                    method: [v for v in data[mi][jj].tolist()]
                    for mi, method in enumerate(methods)
                }

                # plt.tight_layout()
                df1 =pd.DataFrame(penguin_means)
                df1.columns = label_names
                df2 = df1.T
                df2.columns = species
                fe_method11=fe_method.replace('HiDARE', 'HERE')
                df2.to_csv(f'{save_root}/{dataset_name}_{fe_method11}_{name1}_binary_comparision.csv')


    color_keys = list(mcolors.CSS4_COLORS.keys())
    np.random.shuffle(color_keys)

    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names = ['bcss_512_0.8', 'NuCLS', 'PanNuke', 'kather100k']

    mapper_dict = {
        'IndexFlatIP': 'Original',
        'IndexBinaryFlat_ITQ32_LSH': 'ITQ+LSH(32)',
        'IndexBinaryFlat_ITQ64_LSH': 'ITQ+LSH(64)',
        'IndexBinaryFlat_ITQ128_LSH': 'ITQ+LSH(128)',
        'IndexHNSWFlat_m8_IVFPQ_nlist128_m8': 'HNSW+IVFPQ(8,128)',
        'IndexHNSWFlat_m8_IVFPQ_nlist256_m8': 'HNSW+IVFPQ(8,256)',
        'IndexHNSWFlat_m16_IVFPQ_nlist128_m8': 'HNSW+IVFPQ(16,128)',
        'IndexHNSWFlat_m16_IVFPQ_nlist256_m8': 'HNSW+IVFPQ(16,256)',
        'IndexHNSWFlat_m32_IVFPQ_nlist128_m8': 'HNSW+IVFPQ(32,128)',
        'IndexHNSWFlat_m32_IVFPQ_nlist256_m8': 'HNSW+IVFPQ(32,256)'
    }
    mapper_dict_reverse = {v: k for k,v in mapper_dict.items()}
    xticklabels = {
        'BCSS': ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Fat', 'Plasma', 'Other infil', 'Vessel'],
        'NuCLS': ['Lymphocyte', 'Macrophage', 'Stroma', 'Plasma', 'Tumor'],
        'PanNuke': ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial'],
        'Kather100K': ['Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus', 'Muscle', 'Mucosa', 'Stroma', 'Adeno epithelium']
    }
    fe_methods = ['HiDARE_CONCH']
    fe_label_names = ['HERE_CONCH']
    palette = sns.color_palette('colorblind')



    for name in ['Percision', 'Acc']:
        name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'
        hue_orders = {}
        
        writer = pd.ExcelWriter(os.path.join(SAVE_ROOT, f'Fig2-{name1}.xlsx'))

        for fe_method in fe_methods:
            
            datadata = []
            for kk,vv in all_dfs.items():
                datadata.append(vv[fe_method])
            datadata = pd.concat(datadata, axis=0).reset_index(drop=True)
            # datadata.to_csv(os.path.join(save_root, 'all_{}.csv'.format(name1)))

            #get the ranking 
            all_df = None
            for dataset_name in dataset_names:
                fe_method11=fe_method.replace('HiDARE', 'HERE')
                df = pd.read_csv(f'{save_root}/{dataset_name}_{fe_method11}_{name1}_binary_comparision.csv', index_col=0)
                # if dataset_name == 'NuCLS':
                #     df = df.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                # df1 = pd.DataFrame(np.argsort(df.values, axis=0),columns=df.columns, index=df.index)
                df = df.sum(axis=1)
                if all_df is None:
                    all_df = df.copy()
                else:
                    all_df += df
            all_df=pd.DataFrame(all_df, columns=['score'])
            all_df = all_df.sort_values('score', ascending=False)
            all_df.index.name = 'method'
            # all_df.to_csv(f'{save_root}/{fe_method}_{name1}_binary_ranking.csv')

            indexes = np.array(['IndexBinaryFlat_ITQ32_LSH', 
                'IndexBinaryFlat_ITQ64_LSH',
                'IndexHNSWFlat_m16_IVFPQ_nlist128_m8',
                'IndexHNSWFlat_m8_IVFPQ_nlist256_m8',
                'IndexHNSWFlat_m16_IVFPQ_nlist256_m8',
                'IndexHNSWFlat_m32_IVFPQ_nlist128_m8',
                'IndexHNSWFlat_m8_IVFPQ_nlist128_m8',
                'IndexHNSWFlat_m32_IVFPQ_nlist256_m8',
                'IndexBinaryFlat_ITQ128_LSH', 
                'IndexFlatIP'])
            label_names = ['ITQ+LSH(32)', 'ITQ+LSH(64)', 
                'HNSW+IVFPQ(16,128)', 'HNSW+IVFPQ(8,256)', 'HNSW+IVFPQ(16,256)', 'HNSW+IVFPQ(32,128)', 'HNSW+IVFPQ(8,128)', 'HNSW+IVFPQ(32,256)',
                'ITQ+LSH(128)', 'Original']
            indexes = np.array(['IndexBinaryFlat_ITQ32_LSH', 
                'IndexBinaryFlat_ITQ64_LSH',
                'IndexHNSWFlat_m16_IVFPQ_nlist128_m8',
                # 'IndexHNSWFlat_m8_IVFPQ_nlist256_m8',
                'IndexHNSWFlat_m16_IVFPQ_nlist256_m8',
                'IndexHNSWFlat_m32_IVFPQ_nlist128_m8',
                # 'IndexHNSWFlat_m8_IVFPQ_nlist128_m8',
                'IndexHNSWFlat_m32_IVFPQ_nlist256_m8',
                # 'IndexBinaryFlat_ITQ128_LSH', 
                'IndexFlatIP'])
            label_names = ['ITQ+LSH(32)', 'ITQ+LSH(64)', 
                'HNSW+IVFPQ(16,128)', 
                # 'HNSW+IVFPQ(8,256)', 
                'HNSW+IVFPQ(16,256)', 
                'HNSW+IVFPQ(32,128)', 
                # 'HNSW+IVFPQ(8,128)', 
                'HNSW+IVFPQ(32,256)',
                # 'ITQ+LSH(128)', 
                'Original']
            mappers = {kk:vv for kk,vv in zip(indexes, label_names)}
            label_names.reverse()
            all_df.index = label_names
            all_df.index.name = 'method'
            selected_methods = ['ITQ+LSH(32)', 'ITQ+LSH(64)', 'HNSW+IVFPQ(16,128)', 'HNSW+IVFPQ(16,256)', 'HNSW+IVFPQ(32,128)', 'HNSW+IVFPQ(32,256)', 'Original']
            all_df = all_df[all_df.index.isin(selected_methods)].reset_index()
            hue_order = all_df['method'].values
            hue_orders[fe_method] = hue_order

            datadata11 = datadata.copy()
            datadata11['method'] = datadata11['method'].map(mappers)
            datadata1 = []
            for m in all_df['method'].values:
                datadata1.append(datadata11[datadata11['method']==m])
            datadata1 = pd.concat(datadata1,axis=0).reset_index(drop=True)
            datadata11 = datadata1.copy()
            datadata11 = datadata11.rename(columns={'Percision':'Precision', 'Acc':'Majority Vote Accuracy'})
            datadata11.to_excel(writer, sheet_name='Fig 2b,2c search performance')

            for method1 in ['Original', 'HNSW+IVFPQ(32,128)']: 
                print(datadata11.loc[datadata11['method']==method1, ['Majority Vote Accuracy','Precision']].median())

            # boxplot with strip dots
            for dolegend in ['auto', False]:
                plt.close('all')
                font_size = 30
                figure_height = 7
                figure_width = 7
                plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                plt.tick_params(pad = 10)
                fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
                ax = plt.gca()
                g=sns.boxplot(data=datadata1, x="method",  palette=palette, y=name, hue="method", legend=False, ax=ax)  #showfliers=False, 

                if dolegend=='auto':
                    # plt.legend(title="(n=43)", loc="upper left", bbox_to_anchor=(1, 1))  # Adjust legend position

                    ax2 = g.secondary_yaxis('right')
                    ax2.set_yticks(g.get_yticks())
                    ax2.set_yticklabels(['' for _ in g.get_yticklabels()], rotation=90, ha="left", va='center', rotation_mode='anchor')
                    ax2.tick_params(axis='y', length=0)
                    ax2.set_ylabel('(n = {})'.format(len(datadata1['label'].unique())))

                # if dolegend=='auto':
                #     sns.move_legend(
                #         ax, "outside right",
                #         title=None
                #     )
                # g.set(ylabel=None)
                # g.set(xlabel=None)
                # g=sns.stripplot(data=datadata1, palette=[(0,0,0),(0,0,0)],x="method", y=name, legend=False, marker="$\circ$", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
                # g.set(ylabel=name1)
                g.set(ylabel=None)#='Accuracy' if name =='Acc' else 'Average Precision')
                g.set(xlabel=None)
                g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
                g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')

                g=sns.stripplot(data=datadata1, x="method", y=name, jitter=True, color='black', alpha=0.4, ax=ax)
                g.set(ylabel='Mean Majority Vote Accuracy' if name =='Acc' else 'Average Precision')
                g.set(xlabel=None)
                g.set(ylim=[-0.05, 1.05])  
                g.set(yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticklabels=['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
                plt.ylim([-0.05, 1.05])
                # g.map_dataframe(sns.stripplot, x="method", y=name, legend=False, dodge=True, 
                #     marker="$\circ$", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)


                fe_method11=fe_method.replace('HiDARE', 'HERE')
                plt.savefig(os.path.join(save_root, f'{fe_method11}_ranking_meanstd_{name1}_strip_legend{dolegend}.png'), bbox_inches='tight', transparent=True, format='png')
                plt.savefig(os.path.join(save_root, f'{fe_method11}_ranking_meanstd_{name1}_strip_legend{dolegend}.svg'), bbox_inches='tight', transparent=True, format='svg')
                plt.savefig(os.path.join(save_root, f'{fe_method11}_ranking_meanstd_{name1}_strip_legend{dolegend}.pdf'), bbox_inches='tight', transparent=True, format='pdf')
                # datadata1.to_csv(os.path.join(save_root, f'{fe_method}_ranking_meanstd_{name1}_strip_legend{dolegend}.csv'))
                plt.close()

            for di, dataset_name in enumerate(dataset_names):
                plt.close()
                df2 = all_dfs[dataset_name][fe_method]
                font_size = 30
                figure_height = 7
                figure_width = 7
                plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                plt.tick_params(pad = 10)
                fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
                # Draw a nested barplot by species and sex
                g = sns.catplot(
                    data=df2, kind="bar",
                    x="label", y=name, hue="method",
                    errorbar="sd", palette="colorblind", 
                    legend=False, hue_order=[mapper_dict_reverse[k] for k in hue_order],
                    aspect=1.5
                )
                g.ax.yaxis.tick_right()
                # g.despine(top=False, right=False, left=False, bottom=False)
                sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
                g.ax.set_ylim(ylims[dataset_names1[di]])

                g.ax.yaxis.set_label_position("right")
                g.set_axis_labels("", name1 if 'mAP' not in name1 else 'Average precision')
                # g.legend.set_title("")
                # g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="right", va="center", rotation_mode='anchor')
                g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode='anchor')
                g.ax.set_xticklabels([xticklabels[dataset_names1[di]][iii] for iii in range(len(g.ax.get_xticklabels()))], rotation=90, ha="right", va='center', rotation_mode='anchor')
                # g.legend.remove()
                # g.set_titles(dataset_names1[di])
                # g.ax.set_title(dataset_names1[di], fontsize=font_size) 
                # ax2 = g.ax.twinx()
                # ax2.set_ylabel(dataset_names1[di])
                # ax2.set_yticklabels([])
                # ax2.yaxis.set_label_position('left')
                plt.title(dataset_names1[di], fontsize=font_size)

                plt.savefig(os.path.join(save_root, '{}_{}_binary_result2.png'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='png')
                plt.savefig(os.path.join(save_root, '{}_{}_binary_result2.svg'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='svg')
                df2.to_csv(os.path.join(save_root, '{}_{}_binary_result2.csv'.format(dataset_names[di], name1)))
                plt.close()


            if True:
                with open(f'{root}/bcss_512_0.8_HiDARE_CONCH_0_feats_TCGA_binary_search_times.pkl','rb') as fp:
                    data1 = pickle.load(fp)
                with open(f'{root}/bcss_512_0.8_HiDARE_CONCH_0_feats_NCIData_binary_search_times.pkl','rb') as fp:
                    data2 = pickle.load(fp)
                with open(f'{root}/kather100k_HiDARE_CONCH_0_feats_binary_search_times.pkl', 'rb') as fp:
                    data3 = pickle.load(fp)
                with open(f'{root}/bcss_256_0.5_HiDARE_CONCH_0_feats_binary_search_times.pkl', 'rb') as fp:
                    data4 = pickle.load(fp)
                with open(f'{root}/NuCLS_HiDARE_CONCH_0_feats_binary_search_times.pkl', 'rb') as fp:
                    data5 = pickle.load(fp)
                with open(f'{root}/PanNuke_HiDARE_CONCH_0_feats_binary_search_times.pkl', 'rb') as fp:
                    data6 = pickle.load(fp)
                all_search_times = {
                    'TCGA': {k.replace('faiss_', ''): v for k,v in data1['search_times'].items()},
                    'NCIData': {k.replace('faiss_', ''): v for k,v in data2['search_times'].items()},
                    'Kather100K': data3['search_times'],
                    'BCSS': data4['search_times'],
                    'NuCLS': data5['search_times'],
                    'PanNuke': data6['search_times'],
                }
                with open(f'{root}/faiss_bins_count_and_size.pkl','rb') as fp:
                    data7 = pickle.load(fp)
                df1 = pd.DataFrame({k: {kk.replace('faiss_', ''): vv for kk, vv in v.items()} for k, v in data7['all_sizes'].items()})/1024/1024/1024
                df1.index.name = 'method'
                df2 = pd.DataFrame({k: {kk.replace('faiss_', ''): vv for kk, vv in v.items()} for k, v in data7['all_total_rows'].items()})
                df2.index.name = 'method'
                df3 = pd.DataFrame(all_search_times)
                df3.index.name = 'method'
                df1, df2, df3 = df1.reset_index(), df2.reset_index(), df3.reset_index()
                df1['method'] = df1['method'].map(mapper_dict)
                df2['method'] = df2['method'].map(mapper_dict)
                df3['method'] = df3['method'].map(mapper_dict)
                df1 = df1.set_index('method')
                df2 = df2.set_index('method')
                df3 = df3.set_index('method')
                selected_methods = hue_order # ['ITQ+LSH(32)','ITQ+LSH(64)',  'HNSW+IVFPQ(16,128)', 'HNSW+IVFPQ(16,256)', 'HNSW+IVFPQ(32,128)', 'HNSW+IVFPQ(32,256)', 'Original']
                df1 = df1.loc[selected_methods] # storage size (Gb)
                df2 = df2.loc[selected_methods]  # number of patches
                df3 = df3.loc[selected_methods]  # search time (s)


                # broken axis for search time
                for ppi, proj_name in enumerate(['TCGA', 'NCIData', 'Kather100K']):

                    plt.close()
                    font_size = 30
                    figure_height = 7
                    figure_width = 7
                    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                    plt.tick_params(pad = 10)
                    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(figure_width, figure_height))
                    fig.subplots_adjust(hspace=0.1)  # adjust space between Axes
                    pts = df3[proj_name].values[::-1]
                    x = np.arange(len(pts))
                    palette1=sns.color_palette('colorblind')
                    palette2 = [palette1[i] for i in range(len(pts))][::-1]
                    ax1.barh(x, pts, color=palette2)
                    ax2.barh(x, pts, color=palette2)
                    # zoom-in / limit the view to different portions of the data
                    if proj_name == 'TCGA':
                        ax1.set_xlim(0, 0.8)  # most of the data 
                        ax2.set_xlim(20, 45)  # outliers only
                        ax2.set_xticks([20, 30, 40], ['20', '30', '40'])
                    elif proj_name == 'NCIData':
                        ax1.set_xlim(0, 0.4)  # most of the data
                        ax1.set_xticks([0, 0.3], ['0', '0.3'])
                        ax2.set_xlim(20, 31)  # outliers only
                        ax2.set_xticks([20, 25, 30], ['20', '25', '30'])
                    elif proj_name == 'Kather100K':
                        ax1.set_xlim(0, 0.0005)  # most of the data
                        ax1.set_xticks([0, 0.0003], ['0', '3e-3'])
                        ax2.set_xlim(0.005, 0.015)  # outliers only
                        ax2.set_xticks([0.005, 0.01], ['5e-3', '0.01'])
                    # hide the spines between ax and ax2
                    ax1.spines.right.set_visible(False)
                    ax2.spines.left.set_visible(False)
                    ax1.yaxis.tick_left()
                    ax1.tick_params(labelleft=False)  # don't put tick labels at the top
                    ax2.yaxis.tick_right()

                    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
                    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
                    ax2.plot([0, 0], [1, 0], transform=ax2.transAxes, **kwargs)

                    # plt.title(proj_name, fontsize=font_size)
                    # plt.ylabel(None)
                    # plt.xlabel('Search time (s)')
                    fig.text(0.5, -0.0005, 'Search time (s)', ha='center')
                    fig.text(0.5, 0.9, proj_name, ha='center')

                    fe_method11=fe_method.replace('HiDARE', 'HERE')
                    plt.savefig(os.path.join(save_root, f'{fe_method11}_search_time_comparison_subplot_{proj_name}_v3.png'), bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(os.path.join(save_root, f'{fe_method11}_search_time_comparison_subplot_{proj_name}_v3.svg'), bbox_inches='tight', transparent=True, format='svg')
                    plt.close()
                # df3.to_csv(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_v3.csv'))
                df3.to_excel(writer, sheet_name='Fig 2d, search time (s)')

                # broken plots for storage comparison
                for ppi, proj_name in enumerate(['TCGA', 'NCIData', 'Kather100K']):
                    font_size = 30
                    figure_height = 7
                    figure_width = 7
                    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                    plt.tick_params(pad = 10)

                    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(figure_width, figure_height))
                    fig.subplots_adjust(hspace=0.1)  # adjust space between Axes
                    pts = df1[proj_name].values[::-1]
                    x = np.arange(len(pts))
                    palette1=sns.color_palette('colorblind')
                    palette2 = [palette1[i] for i in range(len(pts))][::-1]
                    ax1.barh(x, pts, color=palette2)
                    ax2.barh(x, pts, color=palette2)
                    # zoom-in / limit the view to different portions of the data
                    if proj_name == 'TCGA':
                        ax1.set_xlim(0, 6)  # most of the data 
                        ax1.set_xticks([0, 2, 4], ['0', '2', '4'])
                        ax2.set_xlim(100, 155)  # outliers only
                        ax2.set_xticks([100, 125, 150], ['100', '125', '150'])
                    elif proj_name == 'NCIData':
                        ax1.set_xlim(0, 5)  # most of the data
                        ax1.set_xticks([0, 2, 4], ['0', '2', '4'])
                        ax2.set_xlim(100, 120)  # outliers only
                        ax2.set_xticks([100, 110, 120], ['100', '110', '120'])
                    elif proj_name == 'Kather100K':
                        ax1.set_xlim(0, 0.005)  # most of the data
                        ax1.set_xticks([0, 0.003], ['0', '3e-3'])
                        ax2.set_xlim(0.05, 0.1)  # outliers only
                        ax2.set_xticks([0.05, 0.075], ['0.05', '0.075'])
                    # hide the spines between ax and ax2
                    ax1.spines.right.set_visible(False)
                    ax2.spines.left.set_visible(False)
                    ax1.yaxis.tick_left()
                    ax1.tick_params(labelleft=False)  # don't put tick labels at the top
                    ax2.yaxis.tick_right()

                    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
                    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
                    ax2.plot([0, 0], [1, 0], transform=ax2.transAxes, **kwargs)

                    # plt.title(proj_name, fontsize=font_size)
                    # plt.ylabel(None)
                    # plt.xlabel('Search time (s)')
                    fig.text(0.5, -0.0005, 'Storage size (Gb)', ha='center')
                    fig.text(0.5, 0.9, proj_name, ha='center')

                    fe_method11=fe_method.replace('HiDARE', 'HERE')
                    plt.savefig(os.path.join(save_root, f'{fe_method11}_storage_comparison_subplot_{proj_name}_v2.png'), bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(os.path.join(save_root, f'{fe_method11}_storage_comparison_subplot_{proj_name}_v2.svg'), bbox_inches='tight', transparent=True, format='svg')
                    plt.close()
                # df1.to_csv(os.path.join(save_root, f'{name1}_{fe_method}_storage_comparison_subplot_v2.csv'))
                df1.to_excel(writer, sheet_name='Fig 2e, storage size (Gb)')


        writer.close()



def Fig3_4():

    # 20240831 update different colors for R0 vs R2|R4, HERE_CONCH
    import os
    import numpy as np
    import pandas as pd
    # from matplotlib import pyplot as plt
    import seaborn as sns
    sns.axes_style()
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import ranksums, wilcoxon
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.colors
    import matplotlib.lines
    from matplotlib.transforms import Bbox, TransformedBbox
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.image import BboxImage
    from matplotlib.patches import Circle
    from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                    AnnotationBbox)
    from matplotlib.cbook import get_sample_data

    def cohend(d1, d2) -> pd.Series:
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = np.mean(d1, axis=0), np.mean(d2, axis=0)
        # return the effect size
        return (u1 - u2) / s


    # excel_filename = '/Users/zhongz2/down/HERE/Tables/refined.xlsx'
    # df = pd.read_excel(excel_filename, index_col=0)
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0426.xlsx'
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0507.xlsx'
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0508.xlsx'
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0522.xlsx'
    excel_filename = '/Users/zhongz2/down/HERE/Tables/refined Ver0831.xlsx'
    df = pd.read_excel(excel_filename)
    # 20241215 Jinin evaluation 6 cases
    df1 = pd.read_excel('/Users/zhongz2/down/hidare_result  6 METHODS 5-6 CASES_20241215.xlsx', sheet_name='CombinedZZS')
    df1 = df1.groupby('query').agg({'RetCCL': 'mean', 'Yottixel': 'mean', 'SISH': 'mean'})
    # 20250115 Jinlin evaluation 101 cases for 3 methods
    df11 = pd.read_excel('/Users/zhongz2/down/refined Ver20250111.xlsx', sheet_name='CombinedZZS')
    for method in ['RetCCL', 'SISH', 'Yottixel']:
        vals = []
        for v in df11[method].values:
            if isinstance(v, str):
                m = np.array([float(vv) for vv in v.split(',')]).mean()
            else:
                m = v
            vals.append(m)
        print(vals)
        df11[method] = vals
    df1 = df11.set_index('query')
    df2 = df1.merge(df, left_on='query', right_on='query', how='inner').reset_index()

    df = df2


    if 'AdaptiveHERE' not in df.columns:
        # get Adaptive HERE score
        df['AdaptiveHERE'] = df[['r0', 'r2', 'r4']].fillna(-1).max(axis=1)
        df.to_excel(excel_filename)

    df['r2_r4'] = df[['r2', 'r4']].fillna(-1).max(axis=1)

    compared_method = 'AdapHERECONCH'
    
    sites = df['site'].value_counts().index.values
    site_mappers = {kk:kk for kk in sites[:10]}
    site_mappers.update({kk:'others' for ii,kk in enumerate(sites) if ii>=10})
    df['tissue site'] = df['site'].values
    df['tissue site'] = df['tissue site'].map(site_mappers)
    sites = ['lung', 'liver', 'ovary', 'stomach', 'breast', 'lymph node', 'soft tissue', 'testis', 'kidney', 'colon', 'others']

    def hex_to_rgb(value):
        """Convert a hex color to an RGB tuple."""
        value = value.lstrip('#')
        return tuple(int(value[i:i+2], 16)/255. for i in (0, 2, 4))
    # groups = ['structure']
    COLOR_PALETTES={
        'structure': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'tissue site': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'cell type':  [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'cell shape': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'cytoplasm': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ],
        'label': [
            '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',  '#ECCED0', 
            '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#8A95A9',  '#ECCED0', 
            # '#99857E', '#88878D', '#91A0A5', '#9AA690' 
        ]
    }

    for k,v in COLOR_PALETTES.items():
        newv = []
        for vv in v:
            if isinstance(vv, str) and '#' in vv:
                newv.append(hex_to_rgb(vv))
            else:
                newv.append(vv)
        # newv = [(int(vv[0]*255), int(vv[1]*255), int(vv[2]*255)) for vv in newv]
        COLOR_PALETTES[k] = newv


    save_root = f'{SAVE_ROOT}/Fig3_4'
    save_root = f'{SAVE_ROOT}/Fig3_4 HERE101'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)

    groups = ['structure', 'cell type', 'cell shape', 'cytoplasm', 'label', 'tissue site']
    # groups = ['structure','tissue site']
    group_names = {
        'structure': 'tissue structure',
        'cell type': 'cell type',
        'cell shape': 'cellular shape',
        'cytoplasm': 'cytoplasm',
        'label': 'tissue composition',
        'tissue site': 'tissue site'
    }
    # groups = ['cell type']
    # group_names = {
    #     'cell type': 'cell type',
    # }
    # df = df[df['label'].notna()].reset_index(drop=True)
    # hue_orders = {}
    order = ['PLIP', 'Yottixel', 'RetCCL', 'SISH', 'HERE']

    for expname in ['overall']: #, 'R0_vs_R2R4']:
        if expname == 'overall':
            df1 = df[['WebPLIP','RetCCL', 'SISH', 'Yottixel', compared_method] + groups].rename(columns={'WebPLIP': 'PLIP','RetCCL':'RetCCL','SISH':'SISH','Yottixel':'Yottixel', compared_method: 'HERE'})
            df2 = pd.melt(df1, value_vars=['PLIP','RetCCL', 'SISH', 'Yottixel','HERE'], id_vars=groups, var_name='method', value_name='score')
        else:
            df1 = df[df['r2_r4']>0][['r0', 'r2_r4'] + groups].rename(columns={'r0': 'R0', 'r2_r4':'R2(R4)'})
            df2 = pd.melt(df1, value_vars=['R0','R2(R4)'], id_vars=groups, var_name='method', value_name='score')
        df2['court'] = [i for i in range(len(df1))] + [i for i in range(len(df1))] + [i for i in range(len(df1))] + [i for i in range(len(df1))] + [i for i in range(len(df1))]


        hue_orders = []
        new_ticks = []
        df3_long = []
        num_groups = {}
        for group in groups:
            df3 = df2.copy()
            
            if 'HERE' in df1.columns:
                sorted_HERE = df1[[group, 'HERE']].groupby(group).mean().sort_values('HERE')
                hue_order = sorted_HERE.index.values
            else:
                hue_order = sorted(df3[group].value_counts().index.values)
            hue_orders.extend(hue_order)
            
            df4 = (df3[group].value_counts()//5).loc[hue_order].reset_index()
            num_groups[group] = len(df3[group].value_counts())

            # boxplot-v2 (20240522) using twin axis 
            mapper_dict = {row[group]: '{}'.format(row['count']) for _, row in df4.iterrows()}
            # df31 = df3.replace({group: mapper_dict})
            new_ticks_ = []
            for v in hue_order:
                new_ticks_.append(mapper_dict[v])
            new_ticks.extend(new_ticks_)

            df3_long.append(df3[[group, 'method', 'score']].rename(columns={group:'group'}))
        df3_long = pd.concat(df3_long, axis=0)
        max_len_l = max([len(v) for v in df3_long['group'].values])
        max_len_r = max([len(v) for v in new_ticks])
        max_num_group = max(list(num_groups.values()))
        # import pdb
        # pdb.set_trace()

        hue_orders = {}
        for group in groups:

            df3 = df2.copy()
            
            # if group == 'pattern2':
            #     df3 = df2[df2['pattern2'].isin(['nest', 'glandular', 'sheets', 'spindle', 'papillary'])].reset_index(drop=True)
            
            if False:
                valid_labels = []
                for group_label in df3[group].value_counts().index.values:
                    if min(df3[df3[group]==group_label]['method'].value_counts()) > 2:
                        valid_labels.append(group_label)
                df3 = df3[df3[group].isin(valid_labels)].reset_index(drop=True)

            num_group = len(df3[group].value_counts())
            if expname == 'overall':
                palette = [COLOR_PALETTES[group][i] for i in range(num_group)]
            else:
                palette = [COLOR_PALETTES[group][-i-1] for i in range(num_group)]

            alpha = 0.7
            palette = [np.array(pp)*alpha+(1-alpha) for pp in palette]

            if 'HERE' in df1.columns:
                sorted_HERE = df1[[group, 'HERE']].groupby(group).mean().sort_values('HERE')
                hue_order = sorted_HERE.index.values
            else:
                hue_order = sorted(df3[group].value_counts().index.values)
            # hue_orders[expname][group] = hue_order
            # if group == 'label' and expname != 'overall':
            #     hue_order = hue_orders['overall'][group]
            if group == 'tissue site':
                hue_order = sites
            print('hue_order', hue_order)
            if expname == 'overall':
                # df4 = df3[group].value_counts().loc[hue_order].reset_index()
                # print('df4', df4)
                # df4 = pd.concat([df4.iloc[range(0, len(df4), 2), :], df4.iloc[range(1, len(df4), 2), :]], axis=0).reset_index(drop=True)
                # df4 = df4.set_index(group)
                df4 = (df3[group].value_counts()//5).loc[hue_order].reset_index()
                # pie chart
                # plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体
                total = df4['count'].sum()

                def fmt(x):
                    # return '{:.1f}%\n(n={:.0f})'.format(x, total*x/100)
                    return '{:.0f}'.format(total*x/100)

                if True:
                    font_size = 30
                    figure_height = 9
                    figure_width = 9
                    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                    plt.tick_params(pad = 10)
                    fig, ax = plt.subplots(figsize=(figure_width, figure_height), frameon=False)
                    labels=df4[group].values
                    print('df4', df4)
                    palette1 = [(1, 1, 1) for _ in range(len(palette))]
                    # pie_patches, pie_texts, pie_autotexts = ax.pie(df4['count'].values, autopct=fmt, colors=palette)
                    pie_patches, pie_texts, pie_autotexts = ax.pie(df4['count'].values, autopct=fmt, colors=palette1, radius=1, textprops={'fontsize': 50})
                    # pie_hander_map = {v: HandlerLineImage("/Users/zhongz2/down/hidare_evaluation_from_Jinlin/png_for_shown/1035409-1.png") for v in pie_patches}
                    # plt.legend(handles=pie_patches, labels=['' for _ in range(len(pie_patches))], handler_map=pie_hander_map)
                    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
                    kw = dict(arrowprops=dict(arrowstyle="->"), bbox=bbox_props, zorder=0, va="center")
                    kw1 = dict(bbox=bbox_props, zorder=0, va="center")
                    for i, (p, label) in enumerate(zip(pie_patches, labels)):
                        ang = (p.theta2 - p.theta1)/2. + p.theta1
                        print(label)
                        y = np.sin(np.deg2rad(ang))
                        x = np.cos(np.deg2rad(ang))
                        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                        connectionstyle = f"angle,angleA=0,angleB={ang}"
                        kw["arrowprops"].update({"connectionstyle": connectionstyle})
                        # Annotate the 2nd position with another image (a Grace Hopper portrait)

                        arr_img = plt.imread("/Users/zhongz2/down/hidare_evaluation_from_Jinlin/allpng_for_shown/p256/{}.png".format(df[df[group]==label]['query'].values[0]), format='png')
                        imagebox = OffsetImage(arr_img, zoom=0.6)
                        imagebox.image.axes = ax
                        ab = AnnotationBbox(imagebox, xy=(x,y),
                                            xybox=(1.3*np.sign(x), 1.35*y),
                                            # xycoords='data',
                                            # boxcoords="offset points",
                                            pad=0.01,
                                            arrowprops=dict(
                                                arrowstyle="->",
                                                connectionstyle=connectionstyle)
                                            )
                        ax.add_artist(ab)
                        # text_anno = ax.annotate(labels[i], xy=(x, y), 
                        #                         xytext=(1.55*np.sign(x), 1.35*y), 
                        #                         horizontalalignment=horizontalalignment, **kw1)
                        text_anno = ax.annotate(labels[i], xy=(x, y), 
                                                xytext=(1.55*np.sign(x), 1.35*y), 
                                                horizontalalignment=horizontalalignment,
                                                fontsize=50)
                        extent = text_anno.get_window_extent() # [[xmin, ymin], [xmax, ymax]]

                        p.set_linewidth(2)
                        p.set_edgecolor('black')


                    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
                    # plt.title(group_names[group]) 
                    plt.savefig(f'{save_root}/pie_{group}_{expname}.png', bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(f'{save_root}/pie_{group}_{expname}.svg', bbox_inches='tight', transparent=True, format='svg')
                    
                    plt.close()

                # df4 = df4.reset_index()
                # df4.columns=['', group]
                df4['color'] = [(int(vv[0]*255), int(vv[1]*255), int(vv[2]*255)) for vv in palette]
                df4.to_excel(f'{save_root}/count_forPieChart_{group}_{expname}.xlsx')

            # boxplot-v2 (20240522) using twin axis 
            if True:
                mapper_dict = {row[group]: '{}'.format(row['count']) for _, row in df4.iterrows()}
                # df31 = df3.replace({group: mapper_dict})
                hue_order = hue_order.tolist() if not isinstance(hue_order, list) else hue_order
                new_ticks = []
                for v in hue_order:
                    new_ticks.append(mapper_dict[v])
                df33 = df3[[group, 'score', 'method']]
                remain_count = max_num_group - len(df4)
                new_cat = hue_order[-1]
                new_tick = new_ticks[-1]
                df333 = [df33]
                for ii in range(remain_count):
                    hue_order.append(f'{new_cat}_{ii}')
                    temp= df33[df33[group]==new_cat].reset_index(drop=True)
                    temp[group]=temp[group].map({new_cat:f'{new_cat}_{ii}'})
                    df333.append(temp)
                    new_ticks.append(new_tick)
                df333 = pd.concat(df333, axis=0)

                for do_legend in [0, 1]:

                    postfix = '_v2'
                    if do_legend==1:
                        postfix='_v2_legend'

                    for style1 in ['style2']:
                        font_size = 30
                        figure_height = 7
                        figure_width = 12
                        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                        plt.rcParams['axes.edgecolor'] = 'black'
                        plt.tick_params(pad = 10)
                        # fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
                        # ax = plt.gca()
                        g=sns.catplot(
                            data=df333, x=group, y="score", hue="method", kind="box", palette=all_colors, 
                            #ax=ax, 
                            hue_order=order,
                            order=hue_order, legend=do_legend==1,
                            height=figure_height,
                            aspect=(len(new_ticks)*(figure_height/3))/figure_height, 
                            # dodge=True,  # Ensure consistent spacing for hue categories
                            # width=0.8,  # Set a fixed width for the boxes
                        )

                        x0y0s,x1y1s = [],[]
                        for p in g.ax.patches:
                            box = p.get_extents().get_points() # [x0,y0],[x1,y1]
                            x0,y0 = box[0]
                            x1,y1 = box[1]
                            x0, y0 = g.ax.transData.inverted().transform((x0, y0))
                            x1, y1 = g.ax.transData.inverted().transform((x1, y1))
                            x0y0s.append([x0,y0])
                            x1y1s.append([x1,y1])
                        x0y0s = np.array(x0y0s)
                        x1y1s = np.array(x1y1s)
                        sort_inds = np.argsort(x0y0s[:,0])
                        x0y0s = x0y0s[sort_inds]
                        sort_inds = np.argsort(x1y1s[:,0])
                        x1y1s = x1y1s[sort_inds]

                        # if group=='structure':
                        #     g.fig.legend(labels=hue_order, ncol=2, loc='outside right')
                        # plt.setp(ax.get_legend().get_texts(), fontsize='24') # for legend text
                        # plt.setp(ax.get_legend().get_title(), fontsize='24') # for legend title
                        xticklabels = []
                        for v in g.ax.get_xticklabels():
                            xticklabels.append(v.get_text().rjust(max_len_l))
                        if style1=='style1':
                            g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="right", va="center")
                            g.ax.set_xticklabels(xticklabels, rotation=90, ha="right", va='center', rotation_mode='anchor')
                        else:        
                            g.ax.set_yticklabels(g.ax.get_yticklabels())#, rotation=90, ha="right", va="center")                
                            # g.ax.set_xticklabels(xticklabels, rotation=45, ha="right", va='center', rotation_mode='anchor')                            g.ax.set_xticklabels(xticklabels, rotation=45, ha="right", va='center', rotation_mode='anchor')
                            g.ax.set_xticklabels(xticklabels, rotation=15, ha="right")#, va='center', rotation_mode='anchor')

                        g.set(ylabel=None)
                        g.set(xlabel=None)
                        g.ax.set(ylim=[0.5, 5.5])  
                        g.ax.set(yticks=[1, 2, 3, 4, 5])
                        plt.ylim([0.5, 5.5])
                        plt.yticks([1, 2,3,4,5], labels=[1,2,3,4,5])
                        # plt.yticks(np.arange(0.5, 5.5, 0.5))
                        # sns.despine(top=True, right=True, bottom=False, left=False, ax=g.ax)

                        # g.map_dataframe(sns.stripplot, x=group, y="score", hue="method", legend=False, dodge=True, 
                        #     marker="$\circ$", ec="face", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3,
                        #     order=hue_order)
                        g.map_dataframe(sns.stripplot, x=group, y="score", hue="method", legend=False, dodge=True, 
                            marker="$\circ$", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3,
                            hue_order=order,
                            order=hue_order)
                        g.set(ylabel='Expert score')
                        # g.set(xlabel=group_names[group])
                        g.set(xlabel=None)
                        # g.map_dataframe(sns.catplot, data=df3, x="method", y="score", hue=group, kind="strip", palette='dark:.25', legend=False, dodge=True)
                        # sns.swarmplot(data=df3, x="method", y="score", hue=group, palette='dark:.25', legend=False, dodge=True)
                        # sns.lineplot(data=df3, x="method", y="score", hue=group, units="court", palette='dark:.7', estimator=None, legend=False)
                        # g.map_dataframe(sns.lineplot, x="method", y="score", hue=group, units="court", estimator=None)
                        # connect line
                        if False:
                            for i in range(num_group):
                                for p1, p2 in zip(g.ax.collections[i].get_offsets().data, g.ax.collections[i+num_group].get_offsets().data):
                                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=palette[i], alpha=0.2)

                        ax2 = g.ax.secondary_xaxis('top')
                        ax2.set_xticks(g.ax.get_xticks())
                        new_ticks = [v.ljust(max_len_r) for v in new_ticks]
                        # print('new_ticks', new_ticks)
                        # import pdb
                        # pdb.set_trace()
                        if style1=='style1':
                            ax2.set_xticklabels(new_ticks, rotation=90, ha="left", va='center', rotation_mode='anchor')
                        else:
                            ax2.set_xticklabels(new_ticks)#,rotation=90, ha="left", va='center', rotation_mode='anchor')
                        ax2.tick_params(axis='x', length=0)
                        ax2.spines['top'].set_visible(False)

                        if num_groups[group] != max_num_group:
                            sns.despine(top=True, right=True, bottom=True, left=False, ax=g.ax)
                            # g.ax.spines['top'].set_visible(False)
                            xmin,xmax=g.ax.get_xlim()
                            ymin,ymax=g.ax.get_ylim()
                            margin=np.abs(xmin-x0y0s[0,0])
                            xx=margin+x1y1s[num_groups[group]*5-1,0]
                            # g.ax.axvline(x=xx,color='black',linewidth=1)
                            # g.ax.vlines(x=xx,ymin=ymin, ymax=ymax,color='black',linewidth=1, linestyle='solid')
                            # g.ax.axhline(y=ymax, xmin=xmin, xmax=margin+x1y1s[num_groups[group]*5-1,0],color='blue',linewidth=1)
                            # g.ax.hlines(y=ymax, xmin=xmin, xmax=xx,color='black',linewidth=1.5, linestyle='solid')
                            g.ax.hlines(y=ymin, xmin=xmin, xmax=xx,color='black',linewidth=1.5, linestyle='solid')

                            g.ax.set_xlim([xmin,xmax])
                            g.ax.set_ylim([ymin,ymax])
                        else:
                            sns.despine(top=True, right=True, bottom=False, left=False, ax=g.ax)

                        g.fig.subplots_adjust(top=0.9, bottom=0.1)  # Adjust margins if needed
                        plt.savefig(f'{save_root}/boxplot_{group}_{expname}{postfix}_{style1}.png', bbox_inches='tight', transparent=True, format='png')
                        plt.savefig(f'{save_root}/boxplot_{group}_{expname}{postfix}_{style1}.svg', bbox_inches='tight', transparent=True, format='svg')
                        df3.to_csv(f'{save_root}/boxplot_{group}_{expname}{postfix}_{style1}.csv')
                        plt.close()

                    if do_legend==0 and group!='structure': # for the legend
                        break



def test_pie_annotation():

    # Image file paths corresponding to each label
    image_dir = '/Users/zhongz2/down/hidare_evaluation_from_Jinlin/allpng_for_shown/p256/'
    image_paths = [
        f'{image_dir}/1035211-1.png',
        f'{image_dir}/1035526-1.png',
        f'{image_dir}/1039659-1.png',
        f'{image_dir}/1040121-1.png',
        f'{image_dir}/1034344-2.png',
    ]

    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
    import numpy as np

    plt.close()

    # Data for the pie chart
    data = [20, 30, 15, 25, 10]
    labels = ['A', 'B', 'C', 'D', 'E']

    fig, ax = plt.subplots()

    # Create pie chart
    wedges, texts, autotexts = ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=140)

    # Customize autotexts (the numbers inside the pie chart)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')

    # Define the radius at which images will be placed
    image_radius = 1.3

    for i, (p, image_path) in enumerate(zip(wedges, image_paths)):
        # Calculate the angle and position for each image
        angle = (p.theta2 - p.theta1) / 2. + p.theta1
        x = np.cos(np.deg2rad(angle)) * image_radius
        y = np.sin(np.deg2rad(angle)) * image_radius

        # Load the image
        img = Image.open(image_path)
        img.thumbnail((50, 50))

        # Place the image using AnnotationBbox
        imagebox = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.set_aspect('equal') 

    plt.savefig('/Users/zhongz2/down/debug.png')
    plt.close()





def test_wave_broken_axis():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    # Sample data
    x = np.arange(10)
    y = np.random.randint(1, 20, size=10)

    plt.close()
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

    # Plot on the first axis
    ax.bar(x, y)
    ax.set_ylim(10, 20)  # Limit for the upper part

    # Plot on the second axis
    ax2.bar(x, y)
    ax2.set_ylim(0, 5)  # Limit for the lower part

    # Hide the spines between the two plots
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Add custom sine wave break
    d = .015  # How big to make the break line
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

    # Add sine wave patch on the top plot
    ax.plot((-d, +d), (-d, +d), **kwargs)  # Top-left diagonal line
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal line

    # Custom sine wave
    x_sine = np.linspace(-d, d, 100)
    y_sine = np.sin(5 * np.pi * x_sine) * d  # 5π for a higher frequency
    ax.add_patch(FancyBboxPatch((1 - d, 0), d * 2, d, boxstyle="round,pad=0", lw=1, color='white'))
    ax.plot(x_sine + 1 - d, y_sine, **kwargs)

    # Repeat for the bottom plot
    kwargs.update(transform=ax2.transAxes)  # Switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal line
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal line

    # Custom sine wave
    ax2.add_patch(FancyBboxPatch((1 - d, 1 - d), d * 2, d, boxstyle="round,pad=0", lw=1, color='white'))
    ax2.plot(x_sine + 1 - d, y_sine + 1, **kwargs)

    # Labels and title
    ax.set_ylabel('Values')
    ax2.set_xlabel('Categories')

    plt.savefig(os.path.join(save_root, f'test.png'))
    plt.close()





def plot_violin_Fig6a(): # run on biowulf


    import pandas as pd
    import numpy as np
    import pickle
    import os
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import ranksums
    import matplotlib.pyplot as plt
    import seaborn as sns


    def cohend(d1, d2) -> pd.Series:
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = np.mean(d1, axis=0), np.mean(d2, axis=0)
        # return the effect size
        return (u1 - u2) / s

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    def set_box_color_ax(ax, bp, color):
        ax.setp(bp['boxes'], color=color)
        ax.setp(bp['whiskers'], color=color)
        ax.setp(bp['caps'], color=color)
        ax.setp(bp['medians'], color=color)

    root = '/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backboneCONCH_dropout0.25/analysis/ST/CONCH/feat_before_attention_feat/'
    svs_prefix = '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1'
    gene_data_filename = f'{root}/gene_data/{svs_prefix}_gene_data.pkl'
    cluster_data_filename = f'{root}/analysis/one_patient_top_128/meanstd_none_kmeans_euclidean_8_1_clustering/{svs_prefix}/{svs_prefix}_cluster_data.pkl'
    vst_filename = f'{root}/vst_dir/{svs_prefix}.tsv'

    save_filename = 'violin.png'
    with open(gene_data_filename, 'rb') as fp:
        gene_data_dict = pickle.load(fp)
    with open(cluster_data_filename, 'rb') as fp:
        cluster_data = pickle.load(fp)
    barcode_col_name = gene_data_dict['barcode_col_name']
    Y_col_name = gene_data_dict['Y_col_name']
    X_col_name = gene_data_dict['X_col_name']
    mpp = gene_data_dict['mpp']
    coord_df = gene_data_dict['coord_df']
    counts_df = gene_data_dict['counts_df']

    vst = pd.read_csv(vst_filename, sep='\t', index_col=0)
    vst = vst.subtract(vst.mean(axis=1), axis=0)

    barcodes = coord_df[barcode_col_name].values.tolist()
    stY = coord_df[Y_col_name].values.tolist()
    stX = coord_df[X_col_name].values.tolist()

    st_patch_size = int(
        pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))
    st_all_coords = np.array([stX, stY]).T
    st_all_coords[:, 0] -= st_patch_size // 2
    st_all_coords[:, 1] -= st_patch_size // 2
    st_all_coords = st_all_coords.astype(np.int32)

    vst = vst.T
    vst.index.name = 'barcode'
    valid_barcodes = set(vst.index.values.tolist())
    # print(len(valid_barcodes))

    # cluster_data_dict['coords_in_original']
    cluster_coords = cluster_data['coords_in_original']
    # cluster_data_dict['cluster_labels']
    cluster_labels = cluster_data['cluster_labels']
    # cluster_coords = cluster_data_dict['coords_in_original']
    # cluster_labels = cluster_data_dict['cluster_labels']

    cluster_barcodes = []
    innnvalid = 0
    iinds = []
    for iiii, (x, y) in enumerate(cluster_coords):
        ind = np.where((st_all_coords[:, 0] == x) & (
            st_all_coords[:, 1] == y))[0]
        if len(ind) > 0:
            barcoode = barcodes[ind[0]]
            if barcoode in valid_barcodes:
                cluster_barcodes.append(barcoode)
                iinds.append(iiii)
        else:
            innnvalid += 1
    cluster_labels = cluster_labels[iinds]
    cluster_coords = cluster_coords[iinds]
    vst1 = vst.loc[cluster_barcodes]
    counts_df1 = counts_df.T
    coord_df1 = coord_df.set_index(barcode_col_name)
    coord_df1.index.name = 'barcode'
    coord_df1 = coord_df1.loc[cluster_barcodes]
    stY = coord_df1[Y_col_name].values.tolist()
    stX = coord_df1[X_col_name].values.tolist()
    final_df = coord_df1.copy()


    cluster_label = 2
    selected_gene_names = ['C1R', 'C1S', 'SERPING1']
    gene_names = vst1.columns.values
    cluster_labels_unique = np.unique(cluster_labels)
    if cluster_label in cluster_labels_unique:
        ind1 = np.where(cluster_labels == cluster_label)[0]
        ind0 = np.where(cluster_labels != cluster_label)[0]
        vst11 = vst1.iloc[ind1][selected_gene_names] # num_spots x num_genes
        vst10 = vst1.iloc[ind0][selected_gene_names]

        res = ranksums(vst11, vst10)
        cohens = cohend(vst11, vst10).values.tolist()
        zscores = res.statistic
        pvalues = res.pvalue

        reject, pvals_corrected, alphacSidakfloat, alphacBonffloat = multipletests(
            pvalues, method='fdr_bh')

        vst2 = vst1[selected_gene_names]
        values = vst2.T.values.flatten()
        cc = [f'Cluster {c}' if c==cluster_label else 'Others' for c in cluster_labels] 
        gene_col = []
        cluster_col = []
        for gene_name in selected_gene_names:
            gene_col.extend([gene_name for _ in range(len(vst2))])
            cluster_col.extend(cc)
        df = pd.DataFrame({'gene_value': values, 'gene_col': gene_col, 'cluster_col': cluster_col})
        

        font_size = 30
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica'})
        # plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)

        g = sns.violinplot(data=df, x="gene_col", y="gene_value", hue="cluster_col")
        g.tick_params(pad = 10)
        g.set_xlabel(None)
        g.set_ylabel(None)

        plt.savefig(save_filename, bbox_inches='tight', transparent=True)
        plt.savefig(save_filename.replace('.png', '.svg'), bbox_inches='tight', transparent=True)
        plt.close()

    final_df['cluster_label'] = cluster_labels
    final_df1 = pd.concat([final_df, vst2], axis=1)
    final_df1.to_excel('final_data.xlsx')


# 20241218 CPTAC cancer search and mutation search plot
def main_20241218_CPTAC_comparision():

    import numpy as np
    import pandas as pd
    import pickle
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cbook import get_sample_data
    from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage, TextArea)
    from matplotlib.patches import Circle
    # sns.set_theme(style="whitegrid")

    topn = 5
    root = '/Volumes/data-1/temp_20240801'
    root = '/Volumes/Jiang_Lab/Data/Zisha_Zhong/temp_20240801'
    root = '/Volumes/data-1/CPTAC/check_CPTAC_search_cancer/YottixelPatches'
    root = f'/Volumes/data-1/CPTAC/check_CPTAC_search_cancer/YottixelPatches_intersection_topn{topn}'
    save_root = f'{SAVE_ROOT}/CPTAC_topn{topn}'
    save_root = f'{SAVE_ROOT}/Fig5_CPTAC_results'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)
    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
    dataset_names = ['CPTAC']
    dataset_names1 = ['CPTAC']

    writer = pd.ExcelWriter(os.path.join(SAVE_ROOT, 'Fig5.xlsx'))

    all_dfs = {}
    for di, dataset_name in enumerate(dataset_names): 
        
        methods = ['YottixelKimiaNet', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HiDARE_ProvGigaPath', 'CONCH', 'HiDARE_CONCH', 'UNI', 'HiDARE_UNI']
        label_names = ['Yottixel', 'RetCCL', 'SISH', 'HIPT', 'CLIP', 'PLIP', 'HERE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HERE_ProvGigaPath', 'CONCH', 'HERE_CONCH', 'UNI', 'HERE_UNI']
        methods = ['Yottixel', 'SISH_slide', 'RetCCL', 'HERE_CONCH']
        label_names = ['Yottixel', 'SISH', 'RetCCL', 'HERE']
        methods = ['RetCCL','SISH_slide','Yottixel', 'HERE_CONCH'][::-1]
        label_names = [ 'RetCCL','SISH', 'Yottixel', 'HERE'][::-1]
        if True:
            data = []
            dfs = []
            for method, label_name in zip(methods, label_names):
                method1 = 'Yottixel' if method == 'YottixelKimiaNet' else method
                filename = f'{root}/{dataset_name}_{method1}_feats_results1.csv'
                filename = f'{root}/mAP_mMV_{method1}.csv'
                # if not os.path.exists(filename) and 'kather100k' in dataset_name and ('Yottixel' in method or 'SISH' in method):
                #     filename = filename.replace('.csv', '_random100_random100.csv')
                # if True:# 'HiDARE_' in method1 or 'ProvGigaPath' in method1 or 'CONCH' in method1:
                #     filename = filename.replace(method1, f'{method1}_0')
                if not os.path.exists(filename):
                    print(filename, ' not existed')
                    break
                print(filename)
                df = pd.read_csv(filename, index_col=0)
                if dataset_name == 'NuCLS':
                    df = df.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                data.append(df.values[:2, :])

                df1 = pd.read_csv(filename, index_col=0)
                if dataset_name == 'NuCLS':
                    df1 = df1.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                df1 = df1.T
                df1['method'] = label_name
                df1.index.name = 'label'
                dfs.append(df1)
            df2 = pd.concat(dfs).reset_index()
            df3 = df2.copy()
            df3['dataset_name'] = dataset_names1[di]
            all_dfs[dataset_name] = df3


            if len(data) != len(methods):
                print('wrong data')
                break

            for jj, name in enumerate(['Acc', 'Percision']):
                name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'
                species = df.columns.values
                penguin_means1 = {
                    method: [float('{:.4f}'.format(v)) for v in data[mi][jj].tolist()]
                    for mi, method in enumerate(methods)
                }
                penguin_means = {
                    method: [v for v in data[mi][jj].tolist()]
                    for mi, method in enumerate(methods)
                }

                # plt.tight_layout()
                df1 =pd.DataFrame(penguin_means)
                df1.columns = label_names
                df2 = df1.T
                df2.columns = species
                df2.to_csv(f'{save_root}/{dataset_name}_{name1}.csv')

    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k'] 
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
    dataset_names = ['CPTAC']
    dataset_names1 = ['CPTAC']
    xticklabels = {
        'BCSS': ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Fat', 'Plasma', 'Other infil', 'Vessel'],
        'NuCLS': ['Lymphocyte', 'Macrophage', 'Stroma', 'Plasma', 'Tumor'],
        'PanNuke': ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial'],
        'Kather100K': ['Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus', 'Muscle', 'Mucosa', 'Stroma', 'Adeno epithelium'],
        # 'CPTAC': ['AML','BRCA','CCRCC','CM','COAD','GBM','HNSCC','LSCC','LUAD','OV','PDA','SAR','UCEC']
        'CPTAC': ['Leukemia','Breast','Clear Renal','Melanoma','Colon','Glioblastoma','Head Neck','Lung Squamous','Lung Adeno','Ovary','Pancreatic','Sarcoma','Endometrial']
    }

    kkk1=['AML','BRCA','CCRCC','CM','COAD','GBM','HNSCC','LSCC','LUAD','OV','PDA','SAR','UCEC']
    vvv1=['Leukemia','Breast','Clear Renal','Melanoma','Colon','Glioblastoma','Head Neck','Lung Squamous','Lung Adeno','Ovary','Pancreatic','Sarcoma','Endometrial']

    """
    acute myeloid leukemia (AML), 
    breast invasive carcinoma (BRCA), 
    clear cell renal cell carcinoma (CCRCC), 
    cutaneous melanoma (CM), colon adenocarcinoma (COAD), 
    glioblastoma multiforme (GBM), 
    head and neck squamous cell carcinoma (HNSCC), 
    lung squamous cell carcinoma (LSCC), 
    lung adenocarcinoma (LUAD), 
    ovarian serous cystadenocarcinoma (OV), 
    pancreatic ductal adenocarcinoma (PDA), 
    sarcoma (SAR), and 
    uterine corpus endometrial carcinoma (UCEC). 
    """
    palette = sns.color_palette('colorblind')
    palette = [
        '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', 
        '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', 
        '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690'
    ]

    datadata = pd.concat([v for k,v in all_dfs.items()],axis=0)
    # datadata.to_csv(os.path.join(save_root, f'ranking_meanstd_all.csv'))
    datadata11 = datadata.copy()
    datadata11 = datadata11.rename(columns={'Percision':'Precision', 'Acc':'Majority Vote Accuracy'})
    datadata11['label'] = datadata11['label'].map({kkkk:'{} ({})'.format(kkkk,vvvv) for kkkk,vvvv in zip(kkk1,vvv1)})
    datadata11.to_excel(writer, sheet_name='Fig 5a,5b CPTAC type')

    #get the ranking
    for name in ['Percision', 'Acc']:

        name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'

        # selected_methods = ['RetCCL', 'SISH', 'Yottixel', 'HERE']
        # palette_all = ['#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690']
        # selected_methods_all = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvG', 'CONCH', 'UNI']
        # selected_methods_all = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvG', 'HERE', 'UNI']
        # method_pallette_dict = {m:p for m,p in zip(selected_methods_all, palette_all[:len(selected_methods_all)])}
        # palette1 = [method_pallette_dict[m] for m in selected_methods]
        # palette1 = ['#686789','#E5E2B9','#A79A89','#E8D3C0']

        # boxplot
        plt.close('all')
        font_size = 30
        figure_height = 7
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
        ax = plt.gca()
        g=sns.boxplot(data=datadata, x="method",  palette=all_colors, y=name, showfliers=False, legend=False, ax=ax) 
        g.set(ylabel=None)
        g.set(xlabel=None)
        g.set(ylim=[-0.05, 1.05])  
        g.set(yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticklabels=['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.ylim([-0.05, 1.05])
        g.set_xticklabels(g.get_xticklabels(), rotation=15,ha="right")#, ha="right", va='center')

        g=sns.stripplot(data=datadata, palette=[(0,0,0),(0,0,0)],x="method", y=name, legend=False, marker="$\circ$", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
        g.set(ylabel='Mean Majority Vote Accuracy' if name =='Acc' else 'Average Precision')
        g.set(xlabel=None)
        
        plt.savefig(os.path.join(save_root, f'ranking_meanstd_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'ranking_meanstd_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
        plt.savefig(os.path.join(save_root, f'ranking_meanstd_{name1}.pdf'), bbox_inches='tight', transparent=True, format='pdf')
        datadata.to_csv(os.path.join(save_root, f'ranking_meanstd_{name1}.csv'))
        plt.close()

        all_df = None
        for dataset_name in dataset_names:
            df = pd.read_csv(f'{save_root}/{dataset_name}_{name1}.csv', index_col=0)
            
            df = df.sum(axis=1)
            if all_df is None:
                all_df = df.copy()
            else:
                all_df += df
        all_df=pd.DataFrame(all_df, columns=['score'])
        all_df = all_df.sort_values('score', ascending=False)
        all_df.to_csv(f'{save_root}/ranking_{name1}.csv')

        all_df.index.name = 'method'
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE_PLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov', 'HERE_CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'UNI']
        selected_methods = ['RetCCL', 'SISH', 'Yottixel', 'HERE']
        all_df = all_df[all_df.index.isin(selected_methods)].reset_index()
        all_df1 = all_df.copy()
        all_df1['score1'] = np.log(all_df1['score'] - 30)
        all_df1['score2'] = all_df1['score']

        # 
        hue_order = all_df['method'].values
        ylims = {
            'BCSS': [0, 1],
            'Kather100K': [0, 1],
            'PanNuke': [0, 1],
            'NuCLS': [0, 1],
            'CPTAC': [0, 1]
        }
        for di, dataset_name in enumerate(dataset_names):
            if dataset_name not in all_dfs:
                continue
            df2 = all_dfs[dataset_name]
            # Draw a nested barplot by species and sex
            plt.close()

            num_labels = len(df2['label'].value_counts())
            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            g = sns.catplot(
                data=df2, kind="bar",
                x="label", y=name, hue="method", hue_order=hue_order,
                errorbar="sd", palette=all_colors, height=6,legend=False,aspect=1.5
            )
            sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
            g.ax.yaxis.tick_right()
            g.ax.set_ylim(ylims[dataset_names1[di]])
            g.ax.yaxis.set_label_position("right")
            g.set_axis_labels("", 'Mean Majority Vote Accuracy' if name =='Acc' else 'Average Precision')
            print(name1, g.ax.get_yticklabels())
            g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode='anchor')
            g.ax.set_xticklabels([xticklabels[dataset_names1[di]][iii] for iii in range(len(g.ax.get_xticklabels()))], rotation=90, ha="right", va='center', rotation_mode='anchor')
            # plt.title(dataset_names1[di], fontsize=font_size)
            plt.savefig(os.path.join(save_root, '{}_{}.png'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, '{}_{}.svg'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='svg')
            plt.savefig(os.path.join(save_root, '{}_{}.pdf'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='pdf')
            df2.to_csv(os.path.join(save_root, '{}_{}.csv'.format(dataset_names[di], name1)))
            plt.close()


    # # 20241218 CPTAC mutation search plot
    # def main_20241218_CPTAC_mutation_search_comparision():
    import numpy as np
    import pandas as pd
    import pickle
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cbook import get_sample_data
    from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage, TextArea)
    from matplotlib.patches import Circle
    from common import CLASSIFICATION_DICT, REGRESSION_LIST, IGNORE_INDEX_DICT, ALL_CLASSIFICATION_DICT


    # sns.set_theme(style="whitegrid")    
    topn = 5
    root = '/Volumes/data-1/temp_20240801'
    root = '/Volumes/Jiang_Lab/Data/Zisha_Zhong/temp_20240801'
    root = '/Volumes/data-1/CPTAC/check_CPTAC_search_cancer/YottixelPatches'
    root = '/Volumes/data-1/CPTAC/check_CPTAC_search_mutation/YottixelPatches'
    root = f'/Volumes/data-1/CPTAC/check_CPTAC_search_mutation/YottixelPatches_intersection_topn{topn}'
    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
    dataset_names = ['CPTAC']
    dataset_names1 = ['CPTAC']

    all_dfs = {}
    for di, dataset_name in enumerate(dataset_names): 
        
        methods = ['YottixelKimiaNet', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HiDARE_ProvGigaPath', 'CONCH', 'HiDARE_CONCH', 'UNI', 'HiDARE_UNI']
        label_names = ['Yottixel', 'RetCCL', 'SISH', 'HIPT', 'CLIP', 'PLIP', 'HERE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HERE_ProvGigaPath', 'CONCH', 'HERE_CONCH', 'UNI', 'HERE_UNI']
        methods = ['Yottixel', 'SISH_slide', 'RetCCL', 'HERE_CONCH']
        label_names = ['Yottixel', 'SISH', 'RetCCL', 'HERE']
        methods = ['RetCCL','SISH_slide','Yottixel', 'HERE_CONCH'][::-1]
        label_names = [ 'RetCCL','SISH', 'Yottixel', 'HERE'][::-1]
        if True:
            data = []
            dfs = []
            for method, label_name in zip(methods, label_names):
                method1 = 'Yottixel' if method == 'YottixelKimiaNet' else method
                filename = f'{root}/{dataset_name}_{method1}_feats_results1.csv'
                filename = f'{root}/mAP_mMV_{method1}.csv'
                # if not os.path.exists(filename) and 'kather100k' in dataset_name and ('Yottixel' in method or 'SISH' in method):
                #     filename = filename.replace('.csv', '_random100_random100.csv')
                # if True:# 'HiDARE_' in method1 or 'ProvGigaPath' in method1 or 'CONCH' in method1:
                #     filename = filename.replace(method1, f'{method1}_0')
                if not os.path.exists(filename):
                    print(filename, ' not existed')
                    break
                print(filename)
                df = pd.read_csv(filename, index_col=0)
                if dataset_name == 'NuCLS':
                    df = df.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                data.append(df.values[:2, :])

                df1 = pd.read_csv(filename, index_col=0)
                if dataset_name == 'NuCLS':
                    df1 = df1.drop(['AMBIGUOUS', 'other_nucleus'], axis=1)
                df1 = df1.T
                df1['method'] = label_name
                df1.index.name = 'label'
                dfs.append(df1)
            df2 = pd.concat(dfs).reset_index()
            df3 = df2.copy()
            df3['dataset_name'] = dataset_names1[di]
            all_dfs[dataset_name] = df2


            if len(data) != len(methods):
                print('wrong data')
                break

            for jj, name in enumerate(['Acc', 'Percision']):
                name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'
                species = df.columns.values
                penguin_means1 = {
                    method: [float('{:.4f}'.format(v)) for v in data[mi][jj].tolist()]
                    for mi, method in enumerate(methods)
                }
                penguin_means = {
                    method: [v for v in data[mi][jj].tolist()]
                    for mi, method in enumerate(methods)
                }

                # plt.tight_layout()
                df1 =pd.DataFrame(penguin_means)
                df1.columns = label_names
                df2 = df1.T
                df2.columns = species
                df2.to_csv(f'{save_root}/mutation_{dataset_name}_{name1}.csv')

    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k'] 
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
    dataset_names = ['CPTAC']
    dataset_names1 = ['CPTAC']
    xticklabels = {
        'BCSS': ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Fat', 'Plasma', 'Other infil', 'Vessel'],
        'NuCLS': ['Lymphocyte', 'Macrophage', 'Stroma', 'Plasma', 'Tumor'],
        'PanNuke': ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial'],
        'Kather100K': ['Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus', 'Muscle', 'Mucosa', 'Stroma', 'Adeno epithelium'],
        # 'CPTAC': ['AML','BRCA','CCRCC','CM','COAD','GBM','HNSCC','LSCC','LUAD','OV','PDA','SAR','UCEC']
        'CPTAC': ['TP53','PIK3CA','PTEN','KRAS','ARID1A','BRAF','APC','IDH1','ATRX','CDH1']
    }
    palette = sns.color_palette('colorblind')
    palette = [
        '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', 
        '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', 
        '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690'
    ]

    datadata = pd.concat([v for k,v in all_dfs.items()],axis=0)
    datadata.to_excel(writer, sheet_name='Fig 5c,5d CPTAC mutation')

    #get the ranking
    for name in ['Percision', 'Acc']:

        name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'


        # boxplot
        plt.close('all')
        font_size = 30
        figure_height = 7
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
        ax = plt.gca()
        g=sns.boxplot(data=datadata, x="method",  palette=all_colors, y=name, showfliers=False, legend=False, ax=ax) 
        g.set_xticklabels(g.get_xticklabels(), rotation=15,ha="right")#, ha="right", va='center')

        g.set(ylabel=None)
        g.set(xlabel=None)
        g=sns.stripplot(data=datadata, palette=[(0,0,0),(0,0,0)],x="method", y=name, legend=False, marker="$\circ$", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
        g.set(ylabel='Mean Majority Vote Accuracy' if name =='Acc' else 'Average Precision')
        g.set(xlabel=None)
        
        plt.savefig(os.path.join(save_root, f'mutation_ranking_meanstd_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'mutation_ranking_meanstd_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
        plt.savefig(os.path.join(save_root, f'mutation_ranking_meanstd_{name1}.pdf'), bbox_inches='tight', transparent=True, format='pdf')
        datadata.to_csv(os.path.join(save_root, f'mutation_ranking_meanstd_{name1}.csv'))
        plt.close()



        all_df = None
        for dataset_name in dataset_names:
            df = pd.read_csv(f'{save_root}/mutation_{dataset_name}_{name1}.csv', index_col=0)
            
            df = df.sum(axis=1)
            if all_df is None:
                all_df = df.copy()
            else:
                all_df += df
        all_df=pd.DataFrame(all_df, columns=['score'])
        all_df = all_df.sort_values('score', ascending=False)
        all_df.to_csv(f'{save_root}/mutation_ranking_{name1}.csv')

        all_df.index.name = 'method'
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE_PLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov', 'HERE_CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'UNI']
        selected_methods = ['RetCCL', 'SISH', 'Yottixel', 'HERE']
        all_df = all_df[all_df.index.isin(selected_methods)].reset_index()
        all_df1 = all_df.copy()
        all_df1['score1'] = np.log(all_df1['score'] - 30)
        all_df1['score2'] = all_df1['score']

        # 
        hue_order = all_df['method'].values
        ylims = {
            'BCSS': [0, 1],
            'Kather100K': [0, 1],
            'PanNuke': [0, 1],
            'NuCLS': [0, 1],
            'CPTAC': [0, 1]
        }
        for di, dataset_name in enumerate(dataset_names):
            if dataset_name not in all_dfs:
                continue
            df2 = all_dfs[dataset_name]
            # Draw a nested barplot by species and sex
            plt.close()

            num_labels = len(df2['label'].value_counts())

            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            g = sns.catplot(
                data=df2, kind="bar",
                x="label", y=name, hue="method", hue_order=hue_order,
                errorbar="sd", palette=all_colors, height=6,legend=False,aspect=1.5
            )
            sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
            g.ax.yaxis.tick_right()
            g.ax.set_ylim(ylims[dataset_names1[di]])
            g.ax.yaxis.set_label_position("right")
            g.set_axis_labels("", 'Mean Majority Vote Accuracy' if name =='Acc' else 'Average Precision')
            print(name1, g.ax.get_yticklabels())
            g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode='anchor')
            g.ax.set_xticklabels([xticklabels[dataset_names1[di]][iii] for iii in range(len(g.ax.get_xticklabels()))], rotation=90, ha="right", va='center', rotation_mode='anchor')
            # plt.title(dataset_names1[di], fontsize=font_size)
            plt.savefig(os.path.join(save_root, 'mutation_{}_{}.png'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, 'mutation_{}_{}.svg'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='svg')
            plt.savefig(os.path.join(save_root, 'mutation_{}_{}.pdf'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='pdf')
            df2.to_csv(os.path.join(save_root, 'mutation_{}_{}.csv'.format(dataset_names[di], name1)))
            plt.close()


    writer.close()


def compare_attention_with_noattention():

    import pandas as pd
    import numpy as np
    import pickle
    import os
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import ranksums
    import matplotlib.pyplot as plt
    import seaborn as sns
    save_root = f'{SAVE_ROOT}/Extended Data Fig 2f attention_or_not_comparision'
    os.makedirs(save_root, exist_ok=True)


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

    for k,v in atten_values.items():
        atten_values[k] = np.array(v).reshape(-1)
    for k,v in noatten_values.items():
        noatten_values[k] = np.array(v).reshape(-1)
    df1 = pd.DataFrame(atten_values)
    df2 = pd.DataFrame(noatten_values)

    selected_methods = ['PLIP', 'ProvGigaPath', 'UNI', 'CONCH']
    df1 = pd.melt(df1, value_vars=['PLIP', 'ProvGigaPath', 'UNI', 'CONCH'], var_name='method', value_name='score')
    df1['hue'] = 'Attention'
    df2 = pd.melt(df2, value_vars=['PLIP', 'ProvGigaPath', 'UNI', 'CONCH'], var_name='method', value_name='score')
    df2['hue'] = 'Mean'
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    hue_order = ['Mean', 'Attention']
    palette_all = ['#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690']
    selected_methods_all = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'UNI']
    method_pallette_dict = {m:p for m,p in zip(selected_methods_all, palette_all[:len(selected_methods_all)])}
    palette = [method_pallette_dict[m] for m in selected_methods]
    palette = ['red', 'blue']
    palette = {kk: vv for kk,vv in zip(hue_order, MORANDI_colors[:len(hue_order)])}
    # import pdb
    # pdb.set_trace()
    print('palettepalette', palette)

    plt.close('all')
    font_size = 30
    figure_height = 7
    figure_width = 7
    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
    plt.tick_params(pad = 10)
    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
    ax = plt.gca()
    # g = sns.catplot(
    #     data=df, kind="box",
    #     x="method", y="score", hue="hue", hue_order=hue_order,
    #     errorbar="sd", palette=palette, height=6,legend=False,aspect=1.5
    # )
    # g=sns.catplot(data=df, x="method", y="score", hue="hue", kind="box", palette=palette, ax=ax, order=hue_order)
    g=sns.boxplot(data=df, x="method", y="score", hue="hue", hue_order=hue_order, showfliers=False, palette=palette, legend=True, ax=ax) 
    g.set(ylabel=None)
    g.set(xlabel=None)
    # g.legend.set_title("")
    # plt.legend(loc="lower right", title='')
    sns.move_legend(
        ax, "lower right",
        title=None
    )
    legend = ax.legend()
    legend.get_frame().set_alpha(0)
    # g.set_xticklabels(g.get_xticklabels(), rotation=15, ha="right", va='center', rotation_mode='anchor')
    g.set_xticklabels(g.get_xticklabels(), rotation=15,ha="right")#, ha="right", va='center')

    # g.set(ylabel=None)
    # g.set(xlabel=None)
    # plt.ylim([0.5, 5.5])  
    # plt.yticks(ticks=[1, 2, 3, 4, 5], labels=[1,2,3,4,5])
    # plt.yticks(np.arange(0.5, 5.5, 0.5))
    # sns.despine(top=False, right=False, bottom=False, left=False, ax=g.ax)
    # g.map_dataframe(sns.stripplot, x=group, y="score", hue="method", legend=False, dodge=True, 
    #     marker="$\circ$", ec="face", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3,
    #     order=hue_order)
    # g.map_dataframe(sns.stripplot, x='method', y="score", hue="hue", legend=False, dodge=True, 
    #     marker="$\\circ$", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3,
    #     order=hue_order)
    g=sns.stripplot(data=df, palette=[(0,0,0),(0,0,0)],x="method", y="score", hue="hue",hue_order=hue_order, legend=False, dodge=True, marker="$\circ$", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
    g.set(ylabel='Overall scores')
    g.set(xlabel=None)

    plt.savefig(f'{save_root}/with_attention_or_not.png', bbox_inches='tight', transparent=True, format='png')
    plt.savefig(f'{save_root}/with_attention_or_not.svg', bbox_inches='tight', transparent=True, format='svg')
    plt.savefig(f'{save_root}/with_attention_or_not.pdf', bbox_inches='tight', transparent=True, format='pdf')
    df.to_csv(f'{save_root}/with_attention_or_not.csv')
    df.to_excel(f'{SAVE_ROOT}/Extended Data Fig 2f.xlsx')
    plt.close('all') 
    


def plot_gene_mutation_and_regression_plots():

    import os
    import numpy as np
    import pandas as pd
    # from matplotlib import pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import ranksums, wilcoxon
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.colors
    import matplotlib.lines
    from matplotlib.transforms import Bbox, TransformedBbox
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.image import BboxImage
    from matplotlib.patches import Circle
    from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                    AnnotationBbox)
    from matplotlib.cbook import get_sample_data

    def cohend(d1, d2) -> pd.Series:
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = np.mean(d1, axis=0), np.mean(d2, axis=0)
        # return the effect size
        return (u1 - u2) / s


    save_root = f'{SAVE_ROOT}/gene_mutation_TCGA_and_CPTAC'
    save_root = f'{SAVE_ROOT}/Extended Data Fig 2(g,h) gene_mutation_TCGA_and_CPTAC'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)

    writer = pd.ExcelWriter(os.path.join(SAVE_ROOT, 'Extended Data Fig 2(g,h).xlsx'))
    root = '/Volumes/data-1/CPTAC/'

    TCGA_test_df = pd.read_csv(f'{root}/predictions_v2_TCGA_filterTrue_2/test/CONCH_prediction_scores.csv')
    # CPTAC_df = pd.read_csv(f'{root}/predictions_v2_filterTrue_2/CONCH_prediction_scores.csv')
    CPTAC_df = pd.read_csv(f'{root}/predictions_v3_filterTrue_2/CONCH_prediction_scores.csv') # intersection

    TCGA_test_df = TCGA_test_df.iloc[-1, 2:].reset_index()
    TCGA_test_df.columns = ['var', 'score']
    TCGA_test_df['dataset'] = 'TCGA (test)'
    CPTAC_df = CPTAC_df.iloc[-1, 2:].reset_index()
    CPTAC_df.columns = ['var', 'score']
    CPTAC_df['dataset'] = 'CPTAC'

    df = pd.concat([TCGA_test_df, CPTAC_df])
    gene_cls_vars = [v for v in df['var'].unique() if '_cls' in v]
    gene_reg_vars = ['Cytotoxic_T_Lymphocyte', 'TIDE_CAF', 'TIDE_Dys', 'TIDE_M2', 'TIDE_MDSC'] + \
        [v for v in df['var'].unique() if '_sum' in v]
    df1 = df[df['var'].isin(gene_cls_vars)].reset_index(drop=True)
    df2 = df[df['var'].isin(gene_reg_vars)].reset_index(drop=True)
    df1['var'] = [v.replace('_cls', '').replace('_sum','') for v in df1['var'].values]
    df2['var'] = [v.replace('_cls', '').replace('_sum','') for v in df2['var'].values]
    df2['group'] = 'Gene set regression (n=55)'
    df1 = df1[df1['var']!='GATA3'].reset_index(drop=True)

    palette = sns.color_palette('colorblind')
    palette = [
        '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', 
        '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', 
        '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690'
    ]
    palette = sorted(list(set(all_colors.values()))[::-1])


    # gene mutation barplots
    font_size = 30
    figure_height = 7
    figure_width = 7
    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
    plt.tick_params(pad = 10)
    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
    g = sns.catplot(
        data=df1, kind="bar",
        x="var", y="score", hue="dataset",
        errorbar="sd", palette=palette, height=6,legend=False,aspect=1.5
    )

    plt.axhline(y=0.5, color='blue', linestyle='dashed', linewidth=1.5)

    sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
    g.ax.yaxis.tick_right()
    g.ax.set_ylim([0, 1])
    g.ax.yaxis.set_label_position("right")
    g.set_axis_labels("", "AUC scores")
    g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode='anchor')
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')

    plt.savefig(os.path.join(save_root, 'mutation.png'), bbox_inches='tight', transparent=True, format='png')
    plt.savefig(os.path.join(save_root, 'mutation.svg'), bbox_inches='tight', transparent=True, format='svg')
    plt.savefig(os.path.join(save_root, 'mutation.pdf'), bbox_inches='tight', transparent=True, format='pdf')
    df1.to_csv(os.path.join(save_root, 'mutation.csv'))
    df1.to_excel(writer, sheet_name='Ext Fig 2g mutation')
    plt.close()



    plt.close('all')
    font_size = 28
    figure_height = 7
    figure_width = 7
    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
    plt.tick_params(pad = 10)
    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
    ax = plt.gca()
    g=sns.boxplot(data=df2, x="group", y="score", hue="dataset", showfliers=False, palette=palette, legend=True, ax=ax) 

    plt.axhline(y=0, color='blue', linestyle='dashed', linewidth=1.5)

    g.set(ylabel=None)
    g.set(xlabel=None)
    sns.move_legend(
        ax, "upper right",
        title=None,
        fontsize="x-small"
    )
    g=sns.stripplot(data=df2, palette=[(0,0,0),(0,0,0)],x="group", y="score", hue="dataset", legend=False, dodge=True, marker="$\circ$", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
    g.set(ylabel='Spearmanr correlation score')
    g.set(xlabel=None)

    plt.savefig(os.path.join(save_root, 'generegression.png'), bbox_inches='tight', transparent=True, format='png')
    plt.savefig(os.path.join(save_root, 'generegression.svg'), bbox_inches='tight', transparent=True, format='svg')
    plt.savefig(os.path.join(save_root, 'generegression.pdf'), bbox_inches='tight', transparent=True, format='pdf')
    df2.to_csv(os.path.join(save_root, 'generegression.csv'))
    df2.to_excel(writer, sheet_name='Ext Fig 2h set regression')
    plt.close()


    items = []
    print('\ngene mutation prediction:')
    for v in df1['dataset'].unique():
        values = df1[df1['dataset']==v]['score'].values
        U, pvalue = wilcoxon(values, 0.5*np.ones_like(values), alternative='two-sided')
        print(v, pvalue)
        items.append((f'{v}(Gene mutation classification (n={len(values)}))', pvalue))
    print('\ngene set regression prediction:')
    for v in df2['dataset'].unique():
        values = df2[df2['dataset']==v]['score'].values
        U, pvalue = wilcoxon(values, np.zeros_like(values), alternative='two-sided')
        print(v, pvalue) 
        items.append((f'{v}(Gene set regression (n=({len(values)})))', pvalue))
    dff = pd.DataFrame(items, columns=['cohort', 'p-value'])
    dff.to_excel(writer, sheet_name='p-value')

    writer.close()

"""
We follow the segmentation pipeline proposed in [CLAM] to extract the tissue regions in whole-slide images (WSIs).
Specifically, we first convert eash WSI from RGB to the HSV colour space and apply an median bluring to the saturation 
channel to smooth the edges. 
Then based on the thresholding segmentation on this channel, a binary mask is generated for the tissue region.
Further, an additional morphological closing operation was adopted to fill out the small gaps and holes.

TCGA-06-0124-01Z-00-DX2.b3bd2a52-1a9a-409e-8908-6a2f30878080
"""

def plot_segmentation_patching():  # run on Biowulf

    import sys,os,glob,shutil,json,pickle,h5py
    import numpy as np
    import cv2
    from PIL import Image
    import openslide
    from utils import _assertLevelDownsamples
    import openslide
    import pyvips

    svs_dir = '/data/zhongz2/tcga/TCGA-ALL2_256/svs'
    masks_dir = '/data/zhongz2/tcga/TCGA-ALL2_256/preset_tcga_256_orignalcode/masks'
    patches_dir = '/data/zhongz2/tcga/TCGA-ALL2_256/preset_tcga_256_orignalcode/patches'
    
    svs_prefix = 'TCGA-06-0124-01Z-00-DX2.b3bd2a52-1a9a-409e-8908-6a2f30878080'
    svs_prefix = 'TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01'
    svs_prefix = 'TCGA-86-8358-01Z-00-DX1.C50100F8-9414-4A06-BEDF-93E7B01B24D6'
    
    slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))

    im = cv2.imread(os.path.join(masks_dir, svs_prefix+'.jpg'))

    file = h5py.File(os.path.join(patches_dir, svs_prefix+'.h5'), 'r')
    all_coords = file['coords'][:]
    patch_size = file['coords'].attrs['patch_size']
    patch_level = file['coords'].attrs['patch_level']

    vis_level = 0 # slide.get_best_level_for_downsample(64)

    level_downsamples = _assertLevelDownsamples(slide)
    downsample_patch = level_downsamples[patch_level]
    downsample_seg = level_downsamples[vis_level]
    downsample = (int(downsample_seg[0])/int(downsample_patch[0]),  int(downsample_seg[1])/int(downsample_patch[1]))
    coords1 = np.copy(all_coords)
    coords1[:, 0] = coords1[:, 0] / downsample[0]
    coords1[:, 1] = coords1[:, 1] / downsample[1]
    coords1 = coords1.astype(np.int32)
    downsamples = slide.level_downsamples[vis_level]
    patch_size1 = tuple((np.array((patch_size, patch_size)) * slide.level_downsamples[patch_level]).astype(np.int32))
    patch_size2 = tuple(np.ceil((np.array(patch_size1)/np.array(downsamples))).astype(np.int32))

    file.close()

    im1 = im.copy()
    for x, y in coords1:
        cv2.rectangle(im1, (x, y), (x+patch_size2[0], y+patch_size2[1]), (0, 255, 0), 2)
    
    save_root = '/data/zhongz2/'
    os.makedirs(save_root, exist_ok=True)
    cv2.imwrite(f'{save_root}/{svs_prefix}_patching.png', im1)
    
    slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))
    im = np.array(slide.read_region((0,0),0,slide.level_dimensions[0]).convert('RGB'))
    for x, y in all_coords:
        cv2.rectangle(im, (x, y), (x+patch_size, y+patch_size), (0, 255, 0), 32)
    # img.save(save_filename)
    # cv2.imwrite(f'{save_root}/{svs_prefix}_patching.png', im)
    img=Image.fromarray(im)
    print(type(img), img.size)
    # img.save(save_filename)
    img_vips = pyvips.Image.new_from_array(img)
    # img_vips.dzsave(save_filename, tile_size=1024)
    save_filename = f'{save_root}/{svs_prefix}_patching.tif'
    img_vips.tiffsave(save_filename, compression="none",
        tile=True, tile_width=256, tile_height=256,
        pyramid=True,  bigtiff=True)

def plot_scalability():

    import sys,os,shutil,json,glob,pickle
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import pandas as pd
    import numpy as np
    import pickle
    import os
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import ranksums
    import matplotlib.pyplot as plt
    import seaborn as sns
    root = '/Volumes/data-1/temp_20241204_scalability/'
    save_root = f'{SAVE_ROOT}/scability'
    save_root = f'{SAVE_ROOT}/Extended Data Fig 3 scability analysis'
    os.makedirs(save_root, exist_ok=True)

    writer = pd.ExcelWriter(os.path.join(SAVE_ROOT, 'Extended Data Fig 3.xlsx'))

    with open(os.path.join(root, 'index_time.pkl'), 'rb') as fp:
        index_time = pickle.load(fp)
    with open(os.path.join(root, 'index_size.pkl'), 'rb') as fp:
        index_size = pickle.load(fp)

    key = 'HNSW+IVFPQ(32,128)'

    method = 'HERE_CONCH'
    num_patches = ['1e5', '1e6', '1e7', '1e8', 'TCGA_NCI_CPTAC']

    num_patches2 = []
    for v in num_patches:
        if v=='TCGA_NCI_CPTAC':
            v = '3.2E8'
        num_patches2.append(v.replace('e', 'E')) 

    # index time
    font_size = 24
    figure_height = 7
    figure_width = 7
    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
    plt.tick_params(pad = 10)
    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)

    # plt.bar(num_patches2, index_time['all_search_times'][key])
    dff = pd.DataFrame(index_time['all_search_times'][key]).T
    dff.columns = num_patches2

# colors = [
#     '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9', 
#     '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82', 
#     '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5', '#9AA690'
#     ]

    # Calculate the mean and standard deviation of each column
    means = dff.mean()
    stds = dff.std()
    # Plot the mean values
    plt.bar(means.index, means, yerr=stds, capsize=5, color=MORANDI_colors[0])

    plt.xlabel('Number of patches')
    plt.ylabel('Retrieval time (s)'.format(key))
    
    plt.savefig(f'{save_root}/index_time.png', bbox_inches='tight', transparent=True, format='png')
    plt.savefig(f'{save_root}/index_time.svg', bbox_inches='tight', transparent=True, format='svg')
    plt.savefig(f'{save_root}/index_time.pdf', bbox_inches='tight', transparent=True, format='pdf')
    dff.to_csv(f'{save_root}/index_time.csv')
    dff.to_excel(writer, sheet_name='Ext Fig 3a time (s)')
    plt.close('all')


    # index size
    font_size = 24
    figure_height = 7
    figure_width = 7
    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
    plt.tick_params(pad = 10)
    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)

    plt.bar(num_patches2, index_size['all_sizes'][key], color=MORANDI_colors[0])
    dff = pd.DataFrame(index_size['all_sizes'][key]).T
    dff.columns = num_patches2
    plt.xlabel('Number of patches')
    plt.ylabel('Index size (GB)'.format(key))

    plt.savefig(f'{save_root}/index_size.png', bbox_inches='tight', transparent=True, format='png')
    plt.savefig(f'{save_root}/index_size.svg', bbox_inches='tight', transparent=True, format='svg')
    plt.savefig(f'{save_root}/index_size.pdf', bbox_inches='tight', transparent=True, format='pdf')
    dff.to_csv(f'{save_root}/index_size.csv')
    dff.to_excel(writer, sheet_name='Ext Fig 3b size (Gb)')
    plt.close('all')

    writer.close()


def get_original_data_storage(): # do this on biowulf

    # get all file storage sizes
    import sys,os,glob,shutil
    import numpy as np
    import pandas as pd
    import openslide
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from natsort import natsorted
    import pickle
    import h5py

    save_filename = '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/original_data_storage_20250119.xlsx'
    if os.path.exists(save_filename):
        shutil.rmtree(save_filename, ignore_errors=True)

    data_dir = '/data/Jiang_Lab/Data/COMPASS_NGS_Cases_20240814/'
    postfix = '.ndpi'
    files = natsorted(glob.glob(os.path.join(data_dir, '*.ndpi')))
    with open('/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/all_scales_20241219_newKenData.pkl', 'rb') as fp:
        keys = list(pickle.load(fp).keys())
    svs_dir = '/data/zhongz2/KenData_20240814_256/svs'
    patches_dir = '/data/zhongz2/KenData_20240814_256/patches'
    # KenData
    KenData_keys = [k.replace('KenData_20240814_', '') for k in keys if 'KenData_20240814_' in k]
    items = []
    svs_prefixes = set()
    for ind, filename in enumerate(files):
        prefix = os.path.basename(filename).replace(postfix, '')
        prefix = prefix.replace(' ', '_')
        prefix = prefix.replace(',', '_')
        prefix = prefix.replace('&', '_')
        prefix = prefix.replace('+', '_')
        if prefix in KenData_keys and prefix not in svs_prefixes:
            filename2 = os.path.join(svs_dir, prefix+'.svs')
            with h5py.File(os.path.join(patches_dir, prefix+'.h5')) as file:
                num_patches = len(file['coords'][:])
            items.append([prefix, filename, os.path.getsize(os.path.realpath(filename)), filename2, os.path.getsize(os.path.realpath(filename2)), num_patches])
            svs_prefixes.add(prefix)
            print(prefix)
    df = pd.DataFrame(items, columns=['svs_prefix', 'orig_filepath', 'orig_filesize_inbytes', '20x_filepath', '20x_filesize_inbytes', 'num_patches'])
    df.to_excel(save_filename.replace('.xlsx', '_NCI.xlsx'))
    # TCGA
    items = []
    svs_prefixes = set()
    svs_dir = '/data/zhongz2/TCGA-COMBINED_256/svs'
    patches_dir = '/data/zhongz2/TCGA-COMBINED_256/patches'
    TCGAData_keys = [k.replace('TCGA-COMBINED_', '') for k in keys if 'TCGA-COMBINED_' in k]
    with open('/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/gdc_manifest.2024-06-24.txt', 'r') as fp:
        lines = [line.split('\t') for line in fp.readlines()[1:]]
    for line in lines:
        filename, size = line[1], line[3]
        prefix = filename.replace('.svs', '')
        if prefix in TCGAData_keys and prefix not in svs_prefixes:     
            with h5py.File(os.path.join(patches_dir, prefix+'.h5')) as file:
                num_patches = len(file['coords'][:])   
            filename2 = os.path.join(svs_dir, prefix+'.svs')
            items.append([prefix, filename, float(size), filename2, os.path.getsize(os.path.realpath(filename2)), num_patches])
            svs_prefixes.add(prefix)
    df = pd.DataFrame(items, columns=['svs_prefix', 'orig_filepath', 'orig_filesize_inbytes', '20x_filepath', '20x_filesize_inbytes', 'num_patches'])
    df.to_excel(save_filename.replace('.xlsx', '_TCGA.xlsx'))
    # ST
    invalid_prefixes = [
        '10x_Parent_Visium_Human_Glioblastoma_1.2.0',
        '10x_Targeted_Visium_Human_BreastCancer_Immunology_1.2.0'
    ]
    items = []
    svs_prefixes = set()
    svs_dir = '/data/zhongz2/ST_256/svs'
    patches_dir = '/data/zhongz2/ST_256/patches'
    ST_keys = [k[3:] for k in keys if 'ST_' == k[:3]]
    df = pd.read_excel('/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/ST_list.xlsx', index_col=0)
    df1 = pd.read_excel('/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/ST_list_cancer.xlsx', index_col=0)
    truepaths = []
    for ind, row in df.iterrows():
        prefix = row['slide_id']
        filename = row['TruePath'].split(' ')[0]
        if 'He_2020' in filename and 'Beibei' in filename:
            filename = filename.replace('/data/Jiang_Lab/datashare/Beibei/ST/Gudrun/He_2020_Breast.Cancer',
            '/data/Jiang_Lab/Data/Zisha_Zhong/He_2020_Breast.Cancer')
        truepaths.append(filename)
        if prefix in ST_keys and prefix not in svs_prefixes and prefix not in invalid_prefixes:    
            with h5py.File(os.path.join(patches_dir, prefix+'.h5')) as file:
                num_patches = len(file['coords'][:])   
            filename2 = os.path.join(svs_dir, prefix+'.svs')
            items.append([prefix, filename, os.path.getsize(os.path.realpath(filename)), filename2, os.path.getsize(os.path.realpath(filename2)), num_patches])
            svs_prefixes.add(prefix)
    df['TruePath'] = truepaths
    df = pd.DataFrame(items, columns=['svs_prefix', 'orig_filepath', 'orig_filesize_inbytes', '20x_filepath', '20x_filesize_inbytes', 'num_patches'])
    df.to_excel(save_filename.replace('.xlsx', '_ST.xlsx'))

    # CPTAC
    with open('/data/zhongz2/CPTAC/allsvs/allsvs.txt', 'r') as fp:
        lines = [line.strip().split('/')[-2:] for line in fp.readlines()]
    filenames = {line[1].replace('.svs', ''): '/data/zhongz2/CPTAC/allsvs/{}/{}'.format(line[0], line[1]) for line in lines}
    items = []
    svs_prefixes = set()
    svs_dir = '/data/zhongz2/CPTAC_256/svs'
    patches_dir = '/data/zhongz2/CPTAC_256/patches'
    original_svs_dir = '/data/zhongz2/CPTAC/allsvs'
    CPTAC_keys = [k[6:] for k in keys if 'CPTAC_' == k[:6]]
    for f in glob.glob(os.path.join(patches_dir, '*.h5')):
        prefix = os.path.splitext(os.path.basename(f))[0]
        if prefix in CPTAC_keys:
            with h5py.File(os.path.join(patches_dir, prefix+'.h5')) as file:
                num_patches = len(file['coords'][:])   
            filename = filenames[prefix]
            filename2 = os.path.join(svs_dir, prefix+'.svs')
            items.append([prefix, filename, os.path.getsize(os.path.realpath(filename)), filename2, os.path.getsize(os.path.realpath(filename2)), num_patches])
            print(prefix)
    df = pd.DataFrame(items, columns=['svs_prefix', 'orig_filepath', 'orig_filesize_inbytes', '20x_filepath', '20x_filesize_inbytes', 'num_patches'])
    df.to_excel(save_filename.replace('.xlsx', '_CPTAC.xlsx'))

    # combine
    import time
    if os.path.exists(save_filename):
        shutil.rmtree(save_filename, ignore_errors=True)
        time.sleep(1)

    writer = pd.ExcelWriter(save_filename)
    items = []
    total_storage = 0
    total_WSIs = 0
    total_patches = 0
    total_index_size = 0
    faiss_bin_filenames = {
        'CPTAC': '/data/zhongz2/CPTAC/assets/faiss_bins/all_data_feat_before_attention_feat_faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8_CPTAC_HERE_CONCH.bin',
        'NCI': '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/faiss_relatedV6/faiss_bins/all_data_feat_before_attention_feat_faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8_KenData_20240814_HERE_CONCH.bin',
        'TCGA': '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/faiss_relatedV6/faiss_bins/all_data_feat_before_attention_feat_faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8_TCGA-COMBINED_HERE_CONCH.bin',
        'ST': '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/faiss_relatedV6/faiss_bins/all_data_feat_before_attention_feat_faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8_ST_HERE_CONCH.bin'
    }
    for name in ['NCI', 'TCGA', 'ST', 'CPTAC']:
        name1 = name.replace('NCI', 'NCI Lab of Pathology')
        df = pd.read_excel(save_filename.replace('.xlsx', '_{}.xlsx'.format(name)), index_col=0)
        size = df['orig_filesize_inbytes'].sum()/1024/1024/1024/1024
        num_WSIs = len(df)
        num_patches = df['num_patches'].sum()
        index_size = os.path.getsize(faiss_bin_filenames[name])/1024/1024/1024 # GB
        items.append((name1, num_WSIs, num_patches, size, index_size))
        df.to_excel(writer, sheet_name=name1, index=False)
        total_storage += size
        total_WSIs += num_WSIs
        total_patches += num_patches
        total_index_size += index_size
    items.append(('total', total_WSIs, total_patches,total_storage, total_index_size))
    df = pd.DataFrame(items, columns=['cohort',  'Num_WSIs', 'Num_Patches','Storage (TB)', 'Index Size (GB)'])
    df.to_excel(writer, sheet_name='ALL', index=False)
    writer.close()





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
    for ki, key in enumerate(select_columns):
        if 'auc' in key or 'mse' in key or 'c_index' in key \
                or 'r2score' in key or 'pearsonr' in key or 'pearmanr' in key:
            pass
        else:  # surv, loss
            key += '_loss'

        font_size = 30
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('AUC scores' if 'cls' in task_type else 'Spearmanr correlation score')

        # plt.title('{} task: {}'.format(task_type, key), fontsize=font_size)
        plt.title(key.replace('_cls_auc_weighted', '') if 'auc' in key else key.replace('_sum_spearmanr_corr', ''), fontsize=font_size)
        plt.tick_params(pad = 10)
        print(key)

        best_label = None
        best_score = None
        plotted_data = {}
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
                             label=label, color=all_colors[label], #COMMON_COLORS[label],
                             linewidth=4)
                best_label = label
                best_score = mean_value
            else:
                plt.errorbar(epochs[indices], mean1[indices, ki], std1[indices, ki], fmt='-o', color=all_colors[label], #color=COMMON_COLORS[label],
                             label=label)
            plotted_data.update({f'{label}_mean': mean1[:, ki].tolist(), f'{label}_std': std1[:, ki].tolist()})
        # import pdb
        # pdb.set_trace()
        plotted_data1 = pd.DataFrame(plotted_data)
        plotted_data1['epoch'] = epochs
        plotted_data1.to_excel(os.path.join(save_root,
                                 '{}_{}_step{}.xlsx'.format(subset, key.replace('/', '_'), epoch_step)))
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
        plt.savefig(os.path.join(save_root,
                                 '{}_{}_step{}.png'.format(subset, key.replace('/', '_'), epoch_step)), bbox_inches='tight', transparent=True)
        plt.savefig(os.path.join(save_root,
                                 '{}_{}_step{}.svg'.format(subset, key.replace('/', '_'), epoch_step)), bbox_inches='tight', transparent=True, format='svg')

        if pdf_file_handle is not None:
            pdf_file_handle.savefig(fig)
        plt.close()
    return ranks, best_scores

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


def plot_training_curves_TCGA(results_dir, save_prefix):
    import sys, os, glob, shutil
    import pandas as pd
    import numpy as np
    import pdb
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_pdf import PdfPages
    from common import PAN_CANCER_SITES, CLASSIFICATION_DICT, REGRESSION_LIST

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

    task_types = ['cls', 'reg']
    accum_iters = [4]
    network_dims = {   # only for TCGA-ALL2
        # 'mobilenetv3': 1280,
        # 'CLIP': 512,
        'PLIP': 512,
        'ProvGigaPath': 1536,
        'CONCH': 512,
        'UNI': 1024
    }

    sub_epochs = [1]
    # results_dir = 'results_20241128_e100_noattention'
    # results_dir = 'results_20240724_e100'
    root = '/Volumes/data-1/temp29/debug'
    save_root = f'{SAVE_ROOT}/{save_prefix}' # Add UNI 
    if os.path.isdir(save_root):
        shutil.rmtree(save_root, ignore_errors=True)
    os.makedirs(save_root, exist_ok=True)

    for site_id, site_name in enumerate(all_sites):
        print('begin {}'.format(site_name))
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
                                '{}/{}/ngpus{}_accum{}_backbone{}_dropout0.25'.format(
                                   root, results_dir, num_gpus, accum_iter, backbone
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

                    ranks, best_scores = save_figure_to_one_pdf_new2(subset, task_type, task_name, mean_results, std_results,
                                                   select_columns, epoch_step, prefixes, save_dir,
                                                   pdf_file_handle=pdf_file_handle)
                    all_ranks[task_type][task_name] = ranks
                    all_best_scores[task_type][task_name] = best_scores
            pdf_file_handle.close()

            with open('{}/{}_all_results.pkl'.format(save_root, subset), 'wb') as fp:
                pickle.dump(all_results, fp)
                
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
                sortresults={}
                xxx = {}
                for prefix, all_mean_results_ in all_mean_results.items():
                    x = np.array(all_mean_results_).sum(axis=0).flatten()
                    best_epoch = np.argsort(x)[-1]
                    plt.scatter(best_epoch, x[best_epoch], s=144, c='red')
                    sortresults[prefix] = x[best_epoch]
                for prefix, all_mean_results_ in all_mean_results.items():
                    all_mean_results_ = all_mean_results[prefix]
                    x = np.array(all_mean_results_).sum(axis=0).flatten()
                    best_epoch = np.argsort(x)[-1]
                    label = shorten_prefix2(prefix)
                    # plt.plot(x, 'o-', label=label, color=COMMON_COLORS[label])
                    plt.plot(x, 'o-', label=label, color=all_colors[label])
                    xxx[label]= x.tolist()

                # import pdb
                # pdb.set_trace()
                xxx = pd.DataFrame(xxx)
                xxx.insert(0, 'epoch', np.arange(100))
                xxx.to_excel(savefilename.replace('.png', '.xlsx'), index=False)

                plt.xlabel('Epochs')
                plt.ylabel('Overall scores')
                plt.title(label_names[subset], fontsize=font_size)
                plt.grid()
                if subset == 'train':
                    leg=plt.legend(loc='lower right',handlelength=0, handletextpad=0, fancybox=True)
                    for item, text in zip(leg.legend_handles, leg.get_texts()):
                        print(item.get_color(), text)
                        text.set_color(item.get_color())
                        item.set_visible(False)
                # plt.tight_layout()
                plt.savefig(savefilename, bbox_inches='tight', transparent=True)
                plt.savefig(savefilename.replace('.png', '.svg'), bbox_inches='tight', transparent=True, format='svg')
                plt.close()

                print('\n')
                print('xxx', xxx)

            # # processing all_ranks
            # num_tasks = sum([len(all_ranks[task_type]) for task_type in task_types])  # len(all_ranks['cls']) + len(all_ranks['reg'])
            # num_prefixes = len(prefixes)
            # scores = np.zeros((num_prefixes, num_tasks), dtype=np.uint16)
            # ind = 0
            # for task_index, task_type in enumerate(task_types):
            #     for ki, key in enumerate(all_ranks[task_type].keys()):
            #         sort_inds = list(all_ranks[task_type][key][0].keys())
            #         ranks = list(all_ranks[task_type][key][0].values())
            #         scores[sort_inds, ind] = ranks
            #         ind += 1

            # plot_box_plot_for_check(scores, prefixes, title='{}'.format(subset),
            #                         save_filename=os.path.join('{}/{}_boxplot.png'.format(save_root, subset)))

            # for task_index, task_type in enumerate(task_types):
            #     savefilename111 = os.path.join('{}/{}_{}_best_scores.png'.format(save_root, subset, task_type))
            #     # fig = plt.figure(figsize=(36, 36) if task_type=='reg' else (16, 16))
            #     # ax = fig.add_subplot(111)
            #     font_size = 30
            #     font_size_label = 20
            #     figure_width = 7
            #     plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            #     plt.tick_params(pad = 10)
            #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figure_width, figure_width), frameon=False)

            #     xs = []
            #     ys = []
            #     color = ['cyan', 'blue', 'green', 'red']
            #     for kkks, vvvs in all_best_scores[task_type].items():
            #         for kkkk, vvvv in vvvs.items():
            #             xs.append(kkkk.replace('_sum_spearmanr_corr', '').replace('_cls_auc_weighted', '').replace('_auc_weighted', '').replace('CLS_', '').replace('_spearmanr_corr', ''))
            #             ys.append(vvvv)
            #             break

            #     xs = np.array(xs)
            #     ys = np.array(ys)
            #     ind1 = np.where(ys<0.6)[0]
            #     ind2 = np.where((ys>=0.6)&(ys<0.7))[0]
            #     ind3 = np.where((ys>=0.7)&(ys<0.8))[0]
            #     ind4 = np.where(ys>=0.8)[0]
            #     y_pos = np.arange(len(xs))
            #     for iii, indd in enumerate([ind1, ind2, ind3, ind4]):
            #         ax.barh(y_pos[indd], ys[indd], align='center', color=color[iii])
            #     ax.set_yticks(y_pos, labels=xs)
            #     ax.invert_yaxis()  # labels read top-to-bottom
            #     ax.set_xlabel('Tasks')
            #     ax.set_title('Spearman Correlation' if task_type == 'reg' else "AUC")

            #     plt.grid()
            #     plt.legend(loc='lower right')
            #     if task_type == 'reg':
            #         plt.yticks(fontsize=font_size)
            #         plt.xticks(fontsize=font_size)
            #     plt.savefig(savefilename111, bbox_inches='tight', transparent=True)
            #     plt.savefig(savefilename111.replace('.png', '.svg'), bbox_inches='tight', transparent=True, format='svg')
            #     plt.close()

    for subset, subset_name in label_names.items():
        if subset not in ['train', 'test']:
            continue

        if 'noattention' in results_dir: 
            writer = pd.ExcelWriter(os.path.join(SAVE_ROOT, f'Extended Data Fig 2(c,d,e,f)-{subset}-noattention.xlsx'))
        else:
            writer = pd.ExcelWriter(os.path.join(SAVE_ROOT, f'Extended Data Fig 2(c,d,e,f)-{subset}.xlsx'))
        with open('{}/{}_all_results.pkl'.format(save_root, subset), 'rb') as fp:
            data = pickle.load(fp)
        # data['cls'][task_name][method, split, epoch]
        for i, backbone in enumerate(network_dims.keys()):
            for split in range(5):
                items = {}
                for task_type in ['cls', 'reg']:
                    task_names = list(CLASSIFICATION_DICT.keys()) if task_type == 'cls' else REGRESSION_LIST
                    for ii, task_name in enumerate(task_names):
                        items.update({task_name: data[task_type][ii][i][split][:, 0].tolist()})
                df = pd.DataFrame(items)
                df.columns = [v.replace('_cls','').replace('_sum','') for v in df.columns]
                df.insert(0, 'epoch', np.arange(100))
                df.to_excel(writer, sheet_name='{}(split={})'.format(backbone, split), index=False)
        writer.close()



def plot_colorbar_for_heatmap():
    import numpy as np
    import matplotlib.pyplot as plt 

    # Generate random data for the heatmap
    data = np.random.rand(10, 10)  # 10x10 matrix with random values
    data[0,0] = 0
    data[9,9] = 1.0

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Display the heatmap with "jet" colormap
    heatmap = ax.imshow(data, cmap='jet', aspect='auto')

    # Add a colorbar
    cbar = plt.colorbar(heatmap)

    # Set colorbar tick positions and labels
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Min and max values of the data
    cbar.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Custom labels

    # Show the heatmap
    # plt.show()


    savefilename = 'colorbar_jet_heatmap.png'
    plt.savefig(savefilename, bbox_inches='tight', transparent=True)
    plt.savefig(savefilename.replace('.png', '.svg'), bbox_inches='tight', transparent=True, format='svg')
    plt.close('all')


if __name__ == '__main__':
    main_20240708_encoder_comparision()     # Extended Data Fig 1.xlsx, Extended Data Fig 2e.xlsx
    # compare_attention_with_noattention()    # Extended Data Fig 2f.xlsx
    # main_20241218_CPTAC_comparision()       # Fig5.xlsx
    # plot_gene_mutation_and_regression_plots() # Extended Data Fig 2(g,h).xlsx
    # plot_search_time_tcga_ncidata()         # Fig2.xlsx

    # plot_jinlin_evaluation_boxplots()       # Fig3&4.xlsx
    # Fig3_4()                                

    # plot_scalability()                      # Extended Data Fig 3.xlsx

    # ## run the following on Biowulf
    # # plot_segmentation_patching()         # Extended Data Fig 2b
    # # get_original_data_storage()           # Fig 1.xlsx
    # # plot_violin_Fig6a(): # run on biowulf  # Fig 6a

    # results_dir = 'results_20241128_e100_noattention'
    # plot_training_curves_TCGA(results_dir, 'Extended Data Fig 2 encoder training (Mean)')
    # results_dir = 'results_20240724_e100'
    # plot_training_curves_TCGA(results_dir, 'Extended Data Fig 2 encoder training (Attention)')    # Extended Data Fig 2(c,d,e,f).xlsx

    pass





"""
/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801   1.9T

/data/zhongz2/CPTAC/check_CPTAC_search_cancer/YottixelPatches_intersection_topn5  9.2M

/data/zhongz2/CPTAC/check_CPTAC_search_mutation/YottixelPatches_intersection_topn5  412M

/data/zhongz2/CPTAC/predictions_v2_TCGA_filterTrue_2  327M

/data/zhongz2/CPTAC/predictions_v3_filterTrue_2  38M

/data/zhongz2/temp_20241204_scalability/    110G

# check /data/Jiang_Lab/Data/Zisha_Zhong/results 164G
/data/zhongz2/temp29/debug/results_20241128_e100_noattention 
/data/zhongz2/temp29/debug/results_20240724_e100

"""














