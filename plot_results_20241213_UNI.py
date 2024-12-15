


import sys,os,shutil,glob,json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def Fig3():

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
    df2 = df1.merge(df, left_on='query', right_on='query', how='inner').reset_index()
    df = df2

    if 'AdaptiveHERE' not in df.columns:
        # get Adaptive HERE score
        df['AdaptiveHERE'] = df[['r0', 'r2', 'r4']].fillna(-1).max(axis=1)
        df.to_excel(excel_filename)

    df['r2_r4'] = df[['r2', 'r4']].fillna(-1).max(axis=1)

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


    save_root = '/Users/zhongz2/down/temp_20241215/Fig3_4'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)

    if True: # box plot
        df2 = df[[compared_method, 'WebPLIP', 'RetCCL', 'Yottixel', 'SISH']].rename(columns={compared_method: 'HERE', 'WebPLIP': 'PLIP'})
        df3 = pd.melt(df2, value_vars=['RetCCL', 'Yottixel', 'SISH', 'PLIP', 'HERE'], var_name='method', value_name='score')
        df3['court'] = [i for i in range(len(df2))]*5

        if True: 
            U, pvalue = wilcoxon(df2['PLIP'].values, df2['HERE'].values, alternative='two-sided')

            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            ax = plt.gca()

            num_group = 1
            palette = [(0, 0, 0), (0, 0, 0)]
            g=sns.boxplot(data=df3, x="method", y="score", showfliers=False, palette=palette, ax=ax) 
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
            sns.despine(top=False, right=False, bottom=False, left=False, ax=g)
            # g=sns.stripplot(data=df3, x="method", y="score", legend=False, marker="$\circ$", ec="face", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
            g=sns.stripplot(data=df3, x="method", y="score", legend=False, marker="$\circ$", s=10, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3)
            g.set(ylabel='Expert score')
            g.set(xlabel=None)
            g.set_xticklabels(['RetCCL', 'Yottixel', 'SISH', 'PLIP', 'HERE'], rotation=90, ha="right", va='center', rotation_mode='anchor')


            # plt.tight_layout()

            # for i in range(1):
            #     for p1, p2 in zip(g.collections[i].get_offsets().data, g.collections[i+num_group].get_offsets().data):
            #         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.5)

            # plt.text(0, 5.15, 'P={:.2e}'.format(pvalue), fontsize=30, color=(0, 0, 0))
            plt.savefig(f'{save_root}/overall_boxplot.png', bbox_inches='tight', transparent=True, format='png')
            plt.savefig(f'{save_root}/overall_boxplot.svg', bbox_inches='tight', transparent=True, format='svg')
            df3.to_excel(f'{save_root}/overall_boxplot.xlsx')
            plt.close()

            df4 = df2[['RetCCL', 'Yottixel', 'SISH', 'PLIP', 'HERE']]
            df4.insert(0, column='query', value=df['query'].values)

        # HERE histogram
        if True:
            for method_name in ['PLIP', 'HERE']:
                font_size = 30
                figure_height = 7
                figure_width = 7
                plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                plt.tick_params(pad = 10)
                fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
                ax = plt.gca()

                # g=sns.histplot(data=df3, y='score', hue='method', multiple="dodge", bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], shrink=0.8)
                # plt.xticks([0,10,20,30,40])
                g=sns.histplot(data=df2, y=method_name, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], shrink=0.8, ax=ax)
                plt.xticks([0,10,20,30,40,50])
                plt.ylim([0.5, 5.5])  
                g.set(yticklabels=[])
                g.set(ylabel=None)
                g.set(xlabel='count')
                plt.title(method_name)

                plt.savefig(f'{save_root}/overall_histplot_{method_name}.png', bbox_inches='tight', transparent=True, format='png')
                plt.savefig(f'{save_root}/overall_histplot_{method_name}.svg', bbox_inches='tight', transparent=True, format='svg')
                df2.to_excel(f'{save_root}/overall_histplot_{method_name}.xlsx')
                plt.close()
        # HERE histogram
        if True:
            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            ax = plt.gca()

            alpha = 0.7
            palette0 = np.array(COLOR_PALETTES['label'][1])*alpha+(1-alpha)
            palette1 = np.array(COLOR_PALETTES['label'][0])*alpha+(1-alpha)

            g=sns.histplot(data=df2, y='HERE', bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], shrink=0.8, ax=ax, color=palette0)
            g=sns.histplot(data=df2, y='PLIP', bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], shrink=0.8, ax=ax, color=palette1)
            plt.xticks([0,10,20,30,40,50])
            plt.ylim([0.5, 5.5])  
            g.set(yticklabels=[])
            g.set(ylabel=None)
            g.set(xlabel='count')

            plt.savefig(f'{save_root}/overall_histplot.png', bbox_inches='tight', transparent=True, format='png')
            plt.savefig(f'{save_root}/overall_histplot.svg', bbox_inches='tight', transparent=True, format='svg')
            df2.to_excel(f'{save_root}/overall_histplot.xlsx')
            plt.close()

        # HERE histogram (group plot)
        if True:
            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            ax = plt.gca()

            alpha = 0.7
            palette0 = np.array(COLOR_PALETTES['label'][1])*alpha+(1-alpha)
            palette1 = np.array(COLOR_PALETTES['label'][0])*alpha+(1-alpha)

            v0 = np.histogram(df2['HERE'].values, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])[0][::-1]
            v1 = np.histogram(df2['PLIP'].values, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])[0][::-1]
            court0 = np.arange(len(v0))
            court1 = np.arange(len(v1))
            method0 = ['HERE' for _ in range(len(v0))]
            method1 = ['PLIP' for _ in range(len(v1))]
            df4 = pd.DataFrame({'count': np.concatenate([v0, v1]), 'court': np.concatenate([court0, court1])})
            df4['method'] = method0 + method1
            g=sns.catplot(data=df4, x='court', y="count", hue="method", kind="bar", palette=[palette0, palette1], ax=ax, height=6,legend=False)
            # g=sns.barplot(df4, x="count", y="court", hue="method", palette=[palette0, palette1], ax=ax, legend=False)
            # g=sns.histplot(data=df2, y='HERE', bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], shrink=0.8, ax=ax, color=palette0)
            # plt.xticks([0,10,20,30,40,50])
            plt.yticks([0, 10, 20, 30, 40, 50])
            # plt.ylim([0.5, 5.5])  
            # g.set(yticklabels=[])
            # g.set(ylabel=None)
            # g.set(xlabel='count')
            # g.set(yticklabels=[0, 10, 20, 30, 40, 50])
            # g.set(xticklabels=[5, 4, 3, 2, 1])
            g.ax.yaxis.tick_right()
            # g.despine(top=False, right=False, left=False, bottom=False)
            sns.despine(top=False, right=False, left=False, bottom=False, ax=g.ax)

            g.ax.yaxis.set_label_position("right")
            g.set_axis_labels("", "count")
            # g.legend.set_title("")
            # g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="right", va="center", rotation_mode='anchor')
            g.ax.set_yticklabels(['0', '10', '20', '30', '40', '50'], rotation=90, ha="center", va="top", rotation_mode='anchor')
            # g.ax.set_xticklabels(['5', '4', '3', '2', '1'], rotation=90, ha="right", va='center', rotation_mode='anchor')
            g.set(xticklabels=[])

            plt.savefig(f'{save_root}/overall_histplot_group.png', bbox_inches='tight', transparent=True, format='png')
            plt.savefig(f'{save_root}/overall_histplot_group.svg', bbox_inches='tight', transparent=True, format='svg')
            # df2.to_excel(f'{save_root}/overall_histplot_group.xlsx')
            plt.close()





if __name__ == '__main__':
    main()


















