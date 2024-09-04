








# 20240708  added CONCH
def main_20240708_encoder_comparision():

    import numpy as np
    import pandas as pd
    import pickle
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cbook import get_sample_data
    from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                    TextArea)
    from matplotlib.patches import Circle
    # sns.set_theme(style="whitegrid")

    root = '/Volumes/data-1/temp_20240801'
    root = '/Volumes/Jiang_Lab/Data/Zisha_Zhong/temp_20240801'
    save_root = '/Users/zhongz2/down/temp_20240902/encoder_comparison'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)
    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']

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
                df2.to_csv(f'{save_root}/{dataset_name}_{name1}.csv')

    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k'] 
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
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

    #get the ranking
    # for name in ['Percision']:
    if True: 
        name = 'Percision'
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
        all_df.to_csv(f'{save_root}/ranking_{name1}.csv')

        all_df.index.name = 'method'
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'HERE_PLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'HERE_Prov', 'HERE_CONCH']
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH', 'UNI']
        all_df = all_df[all_df.index.isin(selected_methods)].reset_index()
        all_df1 = all_df.copy()
        all_df1['score1'] = np.log(all_df1['score'] - 30)
        all_df1['score2'] = all_df1['score']

        plt.close()
        font_size = 30
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        # plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
        g=sns.barplot(all_df1, x="method", y="score", hue="method", palette=palette, legend=False)
        # g.set_yscale("log")
        g.tick_params(pad=10)
        g.set_xlabel("")
        g.set_ylabel("Overall ranking")
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
        plt.savefig(os.path.join(save_root, f'ranking_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'ranking_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
        all_df.to_csv(os.path.join(save_root, f'ranking_{name1}.csv'))
        plt.close()

        plt.close()
        font_size = 30
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        # plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
        g=sns.barplot(all_df1, x="method", y="score2", hue="method", palette=palette, legend=False)
        # g.set_yscale("log")
        g.set_ylim([30, 38])
        g.tick_params(pad=10)
        g.set_xlabel("")
        g.set_ylabel("Overall ranking")
        # g.set_ylim([0, 1])
        # g.legend.set_title("")
        # g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=10, ha="right")
        # g.legend.remove()
        # g.set_xticklabels(g.get_xticklabels(), fontsize=9)
        print(name1, g.get_yticklabels())
        g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
        g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')
        for ci, tick_label in enumerate(g.get_xticklabels()):
            tick_label.set_color(palette[ci])
        # plt.tight_layout()
        plt.savefig(os.path.join(save_root, f'ranking_{name1}_v2.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'ranking_{name1}_v2.svg'), bbox_inches='tight', transparent=True, format='svg')
        # all_df.to_csv(os.path.join(save_root, f'ranking_{name1}.csv'))
        plt.close()

        # from compute_flops.py
        total_params_and_flops = {'PLIP': (151277313, 4413615360), 'CONCH': (395232769, 17738386944), 'ProvGigaPath': (1134953984, 228217640448), 'UNI': (303350784, 61603111936), 'Yottixel': (7978856, 2865546752), 'SISH': (7978856, 2865546752), 'MobileNetV3': (5483032, 225436416), 'HIPT': (21665664, 4607954304), 'CLIP': (151277313, 4413615360), 'RetCCL': (23508032, 4109464576)}
        all_df2 = pd.DataFrame(total_params_and_flops).T
        all_df2.columns = ['NumParams', 'FLOPs']
        all_df2 = all_df2.loc[all_df['method'].values]
        all_df2.index.name = 'method'
        all_df2 = all_df2[all_df2.index.isin(selected_methods)].reset_index()
        all_df2 = all_df.merge(all_df2, left_on='method', right_on='method')

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
        g=sns.barplot(all_df2, x="method", y="FLOPs", hue="method", palette=palette, legend=False) 
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
        all_df2.to_csv(os.path.join(save_root, f'flops_{name1}.csv'))
        plt.close()


        
        plt.close()
        font_size = 30
        figure_width = 7
        plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
        # plt.tick_params(pad = 10)
        fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
        # g=sns.scatterplot(all_df2, x="FLOPs", y="score", hue="method", palette=palette, legend=False)
        # g=sns.scatterplot(all_df2, x="FLOPs", y="score", hue="method", palette=all_df2['color'].values.tolist(), legend=False)
        g=sns.scatterplot(all_df2, x="FLOPs", y="score", hue="method", palette=[palette[0] for _ in range(len(all_df2))], legend=False)
        g.tick_params(pad=10)
        g.set_ylabel("Overall scores")
        g.set_xlabel(r"Total FLOPs")
        g.ticklabel_format(style='sci', axis='x')
        g.set_ylim([30, 38])
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

        all_df2['x'] = [2.5*17738386944, 0.55*228217640448, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944]
        all_df2['y'] = [35.521479+1.2, 37.098852, 35.521479+0.6, 35.521479, 35.521479-0.6, 32.989012, 32.989012-0.6, 32.989012-1.2, 32.989012-1.8, 32.989012-2.4]
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
            if tt == 'ProvGigaPath':
                cc = COMMON_COLORS[tt]
                tt = '{}\n    ({:.3f})'.format(tt, 33.241507276672884)
            if tt == 'CONCH':
                cc = COMMON_COLORS[tt]
                tt = '{} ({:.3f})'.format(tt, 34.164964506392046)
            if tt == 'PLIP':
                cc = COMMON_COLORS[tt]
                tt = '{} ({:.3f})'.format(tt, 29.2551421379361)
            g.annotate(tt, color=cc, size=18, xy=(row['FLOPs'], row['score']), ha='left', va='center', xytext=(row['x'], row['y']), \
                 arrowprops=dict(facecolor='black', width=1, headwidth=4, shrink=0.15))

        # plt.tight_layout()
        plt.savefig(os.path.join(save_root, f'ranking_vs_flops_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'ranking_vs_flops_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
        # all_df2.to_csv(os.path.join(save_root, f'flops_{name1}.csv'))
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
                errorbar="sd", palette=palette, height=6,legend=False,aspect=1.5
            )
            sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
            g.ax.yaxis.tick_right()
            g.ax.set_ylim(ylims[dataset_names1[di]])
            g.ax.yaxis.set_label_position("right")
            g.set_axis_labels("", name1 if 'mAP' not in name1 else 'Average precision')
            print(name1, g.ax.get_yticklabels())
            g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode='anchor')
            g.ax.set_xticklabels([xticklabels[dataset_names1[di]][iii] for iii in range(len(g.ax.get_xticklabels()))], rotation=90, ha="right", va='center', rotation_mode='anchor')
            plt.title(dataset_names1[di], fontsize=font_size)
            plt.savefig(os.path.join(save_root, '{}_{}_result1.png'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, '{}_{}_result1.svg'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='svg')
            df2.to_csv(os.path.join(save_root, '{}_{}_result1.csv'.format(dataset_names[di], name1)))
            plt.close()



    #get the ranking (for HERE methods, HERE_PLIP, PLIP, HERE_)
    palette = [
        '#008080', '#029370', 'purple', 'mediumpurple', 'blue', 'royalblue', 'gray', 'lightgray',
    ]
    for name in ['Percision']:
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
        g=sns.barplot(all_df, x="method", y="score", hue="method", palette=palette, legend=False)
        g.tick_params(pad=10)
        g.set_xlabel("")
        g.set_ylabel("Overall ranking")
        # g.set_ylim([0, 1])
        # g.legend.set_title("")
        # g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=10, ha="right")
        # g.legend.remove()
        # g.set_xticklabels(g.get_xticklabels(), fontsize=9)
        print(name1, g.get_yticklabels())
        g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
        g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')
        for ci, tick_label in enumerate(g.get_xticklabels()):
            tick_label.set_color(palette[ci])
        # plt.tight_layout()
        plt.savefig(os.path.join(save_root, f'HERE_ranking_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
        plt.savefig(os.path.join(save_root, f'HERE_ranking_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
        all_df.to_csv(os.path.join(save_root, f'HERE_ranking_{name1}.csv'))
        plt.close()


        if False:
            # from compute_flops.py
            total_params_and_flops = {'PLIP': (151277313, 4413615360), 'CONCH': (395232769, 17738386944), 'ProvGigaPath': (1134953984, 228217640448), 'UNI': (303350784, 61603111936), 'Yottixel': (7978856, 2865546752), 'SISH': (7978856, 2865546752), 'MobileNetV3': (5483032, 225436416), 'HIPT': (21665664, 4607954304), 'CLIP': (151277313, 4413615360), 'RetCCL': (23508032, 4109464576)}
            all_df2 = pd.DataFrame(total_params_and_flops).T
            all_df2.columns = ['NumParams', 'FLOPs']
            all_df2 = all_df2.loc[all_df['method'].values]
            all_df2.index.name = 'method'
            all_df2 = all_df2[all_df2.index.isin(selected_methods)].reset_index()

            plt.close()
            font_size = 30
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            # plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_width), frameon=False)
            g=sns.barplot(all_df2, x="method", y="FLOPs", hue="method", palette=palette, legend=False)
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
            
            for ci, tick_label in enumerate(g.get_xticklabels()):
                tick_label.set_color(palette[ci])
            # plt.tight_layout()
            plt.savefig(os.path.join(save_root, f'HERE_flops_{name1}.png'), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, f'HERE_flops_{name1}.svg'), bbox_inches='tight', transparent=True, format='svg')
            all_df2.to_csv(os.path.join(save_root, f'HERE_flops_{name1}.csv'))
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
                errorbar="sd", palette=palette, height=6,legend=False,aspect=1.5
            )
            sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
            g.ax.yaxis.tick_right()
            g.ax.set_ylim(ylims[dataset_names1[di]])
            g.ax.yaxis.set_label_position("right")
            g.set_axis_labels("", name1 if 'mAP' not in name1 else 'Average precision')
            print(name1, g.ax.get_yticklabels())
            g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode='anchor')
            g.ax.set_xticklabels([xticklabels[dataset_names1[di]][iii] for iii in range(len(g.ax.get_xticklabels()))], rotation=90, ha="right", va='center', rotation_mode='anchor')
            plt.title(dataset_names1[di], fontsize=font_size)
            plt.savefig(os.path.join(save_root, 'HERE_{}_{}_result1.png'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, 'HERE_{}_{}_result1.svg'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='svg')
            df2.to_csv(os.path.join(save_root, 'HERE_{}_{}_result1.csv'.format(dataset_names[di], name1)))
            plt.close()








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
    save_root = '/Users/zhongz2/down/temp_20240902/hashing_comparison2'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)
    # root = '/Users/zhongz2/down'
    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k']
    dataset_names1 = ['BCSS', 'BCSS','BCSS','BCSS', 'NuCLS', 'PanNuke', 'Kather100K']

    # dataset_names = ['bcss_512_0.8']
    # dataset_names = ['bcss_512_0.5', 'NuCLS', 'PanNuke', 'kather100k']
    # dataset_names1 = ['BCSS', 'NuCLS', 'PanNuke', 'Kather100K']
    
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

            all_dfs[dataset_name][fe_method] = df2


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
                df2.to_csv(f'{save_root}/{dataset_name}_{fe_method}_{name1}_binary_comparision.csv')


    color_keys = list(mcolors.CSS4_COLORS.keys())
    np.random.shuffle(color_keys)

    dataset_names = ['bcss_512_0.8', 'bcss_512_0.5', 'bcss_256_0.8', 'bcss_256_0.5', 'NuCLS', 'PanNuke', 'kather100k']
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
    fe_label_names = ['HERE_CONCN']
    palette = sns.color_palette('colorblind')
    for name in ['Percision']:
        name1 = 'mMV@5' if name == 'Acc' else 'mAP@5'
        hue_orders = {}
        for fe_method in fe_methods:
            #get the ranking 
            all_df = None
            for dataset_name in dataset_names:
                df = pd.read_csv(f'{save_root}/{dataset_name}_{fe_method}_{name1}_binary_comparision.csv', index_col=0)
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
            all_df.to_csv(f'{save_root}/{fe_method}_{name1}_binary_ranking.csv')

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
            label_names.reverse()
            all_df.index = label_names
            all_df.index.name = 'method'
            selected_methods = ['ITQ+LSH(32)', 'ITQ+LSH(64)', 'HNSW+IVFPQ(16,128)', 'HNSW+IVFPQ(16,256)', 'HNSW+IVFPQ(32,128)', 'HNSW+IVFPQ(32,256)', 'Original']
            all_df = all_df[all_df.index.isin(selected_methods)].reset_index()
            hue_order = all_df['method'].values
            hue_orders[fe_method] = hue_order
            
            font_size = 30
            figure_height = 7
            figure_width = 7
            plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
            plt.tick_params(pad = 10)
            fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
            g=sns.barplot(all_df, x="method", y="score", hue="method", palette='colorblind', legend=False)
            g.set_xlabel("")
            g.set_ylabel("Overall ranking")
            # g.legend.set_title("")
            g.set_yticklabels(g.get_yticklabels(), rotation=90, ha="right", va="center")
            g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')
            # for ci, tick_label in enumerate(g.get_xticklabels()):
            #     tick_label.set_color(palette[ci])
            # plt.tight_layout()
            plt.savefig(os.path.join(save_root, f'{fe_method}_{name1}_binary_ranking.png'), bbox_inches='tight', transparent=True, format='png')
            plt.savefig(os.path.join(save_root, f'{fe_method}_{name1}_binary_ranking.svg'), bbox_inches='tight', transparent=True, format='svg')
            all_df.to_csv(os.path.join(save_root, f'{fe_method}_{name1}_binary_ranking.csv'))
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
                # g.despine(top=False, right=False, left=False, bottom=False)
                sns.despine(top=True, right=False, left=True, bottom=False, ax=g.ax)
                g.ax.set_ylim(ylims[dataset_names1[di]])
                g.set_axis_labels("", name1 if 'mAP' not in name1 else 'Average precision')
                # g.legend.set_title("")
                g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="right", va="center")
                g.ax.set_xticklabels([xticklabels[dataset_names1[di]][iii] for iii in range(len(g.ax.get_xticklabels()))], rotation=90, ha="right", va='center', rotation_mode='anchor')
                # g.legend.remove()
                # g.set_titles(dataset_names1[di])
                # g.ax.set_title(dataset_names1[di], fontsize=font_size) 
                plt.savefig(os.path.join(save_root, '{}_{}_binary_result1.png'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='png')
                plt.savefig(os.path.join(save_root, '{}_{}_binary_result1.svg'.format(dataset_names[di], name1)), bbox_inches='tight', transparent=True, format='svg')
                df2.to_csv(os.path.join(save_root, '{}_{}_binary_result1.csv'.format(dataset_names[di], name1)))
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

                for ppi, proj_name in enumerate(['TCGA', 'NCIData', 'Kather100K']):
                    font_size = 30
                    figure_height = 7
                    figure_width = 7
                    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                    plt.tick_params(pad = 10)
                    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)

                    g=sns.barplot(data=df1, y='method', x=proj_name, errorbar=None, palette='colorblind', hue='method', legend=False)
                    sns.despine(top=False, right=False, bottom=False, left=False, ax=g)
                    plt.ylabel(None)
                    plt.xlabel('Storage size (Gb)')
                    g.set_yticklabels([])
                    plt.title(proj_name, fontsize=font_size)
                    # for ci, tick_label in enumerate(g.axes.get_yticklabels()):
                    #     tick_label.set_color(palette[ci])
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_storage_comparison_subplot_{proj_name}.png'), bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_storage_comparison_subplot_{proj_name}.svg'), bbox_inches='tight', transparent=True, format='svg')
                    df1.to_csv(os.path.join(save_root, f'{name1}_{fe_method}_storage_comparison_subplot_{proj_name}.csv'))
                    plt.close()

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
                        ax1.set_xticks([0, 1, 2, 3, 4, 5, 6], ['0', '1', '2', '3', '4', '5', '6'])
                        ax2.set_xlim(100, 155)  # outliers only
                        ax2.set_xticks([100, 125, 150], ['100', '125', '150'])
                    elif proj_name == 'NCIData':
                        ax1.set_xlim(0, 5)  # most of the data
                        ax1.set_xticks([0, 1, 2, 3, 4, 5], ['0', '1', '2', '3', '4', '5'])
                        ax2.set_xlim(100, 120)  # outliers only
                        ax2.set_xticks([100, 110, 120], ['100', '110', '120'])
                    elif proj_name == 'Kather100K':
                        ax1.set_xlim(0, 0.005)  # most of the data
                        ax1.set_xticks([0, 0.003], ['0', '3e-3'])
                        ax2.set_xlim(0.05, 0.1)  # outliers only
                        ax2.set_xticks([0.05, 0.075, 0.1], ['0.05', '0.075', '0.01'])
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

                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_storage_comparison_subplot_{proj_name}_v2.png'), bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_storage_comparison_subplot_{proj_name}_v2.svg'), bbox_inches='tight', transparent=True, format='svg')
                    df1.to_csv(os.path.join(save_root, f'{name1}_{fe_method}_storage_comparison_subplot_{proj_name}_v2.csv'))
                    plt.close()

                for ppi, proj_name in enumerate(['TCGA', 'NCIData', 'Kather100K']):

                    font_size = 30
                    figure_height = 7
                    figure_width = 7
                    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                    plt.tick_params(pad = 10)
                    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)

                    g=sns.barplot(data=df3, y='method', x=proj_name, errorbar=None, palette='colorblind', hue='method', legend=False)
                    # g.despine(top=False, right=False, left=False, bottom=False)
                    sns.despine(top=False, right=False, bottom=False, left=False, ax=g)
                    plt.ylabel(None)
                    plt.xlabel('Search time (s)')
                    g.set_yticklabels([])
                    # for ci, tick_label in enumerate(g.axes.get_yticklabels()):
                    #     tick_label.set_color(palette[ci])
                    plt.title(proj_name, fontsize=font_size)
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}.png'), bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}.svg'), bbox_inches='tight', transparent=True, format='svg')
                    df3.to_csv(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}.csv'))
                    plt.close()

                # log-scale
                for ppi, proj_name in enumerate(['TCGA', 'NCIData', 'Kather100K']):

                    font_size = 30
                    figure_height = 7
                    figure_width = 7
                    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                    plt.tick_params(pad = 10)
                    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)

                    g=sns.barplot(data=df3, y='method', x=proj_name, errorbar=None, palette='colorblind', hue='method', legend=False)
                    g.set_xscale("log")
                    # g.despine(top=False, right=False, left=False, bottom=False)
                    sns.despine(top=False, right=False, bottom=False, left=False, ax=g)
                    plt.ylabel(None)
                    plt.xlabel('Search time (s)')
                    g.set_yticklabels([])
                    # for ci, tick_label in enumerate(g.axes.get_yticklabels()):
                    #     tick_label.set_color(palette[ci])
                    plt.title(proj_name, fontsize=font_size)
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}_v2.png'), bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}_v2.svg'), bbox_inches='tight', transparent=True, format='svg')
                    df3.to_csv(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}_v2.csv'))
                    plt.close()

                # broken
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
                        ax2.set_xticks([0.005, 0.01, 0.015], ['5e-3', '0.01', '0.015'])
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
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}_v3.png'), bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}_v3.svg'), bbox_inches='tight', transparent=True, format='svg')
                    df3.to_csv(os.path.join(save_root, f'{name1}_{fe_method}_search_time_comparison_subplot_{proj_name}_v3.csv'))
                    plt.close()





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


    save_root = '/Users/zhongz2/down/temp_20240904/Fig3_4'
    if os.path.exists(save_root):
        os.system('rm -rf "{}"'.format(save_root))
    os.makedirs(save_root, exist_ok=True)

    if True: # box plot
        df2 = df[[compared_method, 'WebPLIP']].rename(columns={compared_method: 'HERE', 'WebPLIP': 'PLIP'})
        df3 = pd.melt(df2, value_vars=['PLIP', 'HERE'], var_name='method', value_name='score')
        df3['court'] = [i for i in range(len(df2))] + [i for i in range(len(df2))]

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
            for i in range(1):
                for p1, p2 in zip(g.collections[i].get_offsets().data, g.collections[i+num_group].get_offsets().data):
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.5)
            plt.text(0, 5.15, 'P={:.2e}'.format(pvalue), fontsize=30, color=(0, 0, 0))
            plt.savefig(f'{save_root}/overall_boxplot.png', bbox_inches='tight', transparent=True, format='png')
            plt.savefig(f'{save_root}/overall_boxplot.svg', bbox_inches='tight', transparent=True, format='svg')
            df3.to_excel(f'{save_root}/overall_boxplot.xlsx')
            plt.close()

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
            palette0 = np.array(COLOR_PALETTES['label'][0])*alpha+(1-alpha)
            palette1 = np.array(COLOR_PALETTES['label'][1])*alpha+(1-alpha)

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
            palette0 = np.array(COLOR_PALETTES['label'][0])*alpha+(1-alpha)
            palette1 = np.array(COLOR_PALETTES['label'][1])*alpha+(1-alpha)

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



    groups = ['label', 'cell_type', 'pattern2']
    groups = ['structure', 'cell type', 'cell shape', 'cytoplasm', 'scattered large cells']
    for col in groups:
        print(col, '='*20)
        print(df[col].value_counts())

    groups = ['structure', 'cell type', 'cell shape', 'cytoplasm', 'label']
    group_names = {
        'structure': 'tissue structure',
        'cell type': 'cell type',
        'cell shape': 'cellular shape',
        'cytoplasm': 'cytoplasm',
        'label': 'tissue composition'
    }
    
    # df = df[df['label'].notna()].reset_index(drop=True)
    hue_orders = {}
    for expname in ['overall']: #, 'R0_vs_R2R4']:
        if expname == 'overall':
            df1 = df[['WebPLIP', compared_method] + groups].rename(columns={'WebPLIP': 'PLIP', compared_method: 'HERE'})
            df2 = pd.melt(df1, value_vars=['PLIP','HERE'], id_vars=groups, var_name='method', value_name='score')
        else:
            df1 = df[df['r2_r4']>0][['r0', 'r2_r4'] + groups].rename(columns={'r0': 'R0', 'r2_r4':'R2(R4)'})
            df2 = pd.melt(df1, value_vars=['R0','R2(R4)'], id_vars=groups, var_name='method', value_name='score')
        df2['court'] = [i for i in range(len(df1))] + [i for i in range(len(df1))]

        hue_orders[expname] = {}
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
            hue_orders[expname][group] = hue_order
            if group == 'label' and expname != 'overall':
                hue_order = hue_orders['overall'][group]
            if expname == 'overall':
                # df4 = df3[group].value_counts().loc[hue_order].reset_index()
                # print('df4', df4)
                # df4 = pd.concat([df4.iloc[range(0, len(df4), 2), :], df4.iloc[range(1, len(df4), 2), :]], axis=0).reset_index(drop=True)
                # df4 = df4.set_index(group)
                df4 = df3[group].value_counts().loc[hue_order].reset_index()
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

            # boxplot-v2 (20240522)
            if True:
                font_size = 18
                figure_height = 7
                figure_width = 7
                plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                plt.tick_params(pad = 10)
                fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
                ax = plt.gca()
                g=sns.catplot(data=df3, x=group, y="score", hue="method", kind="box", palette=palette, ax=ax, order=hue_order, legend=False)
                # if group=='structure':
                #     g.fig.legend(labels=hue_order, ncol=2, loc='outside right')
                # plt.setp(ax.get_legend().get_texts(), fontsize='24') # for legend text
                # plt.setp(ax.get_legend().get_title(), fontsize='24') # for legend title
                g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=90, ha="right", va="center")
                g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')
                g.set(ylabel=None)
                g.set(xlabel=None)
                g.ax.set(ylim=[0.5, 5.5])  
                g.ax.set(yticks=[1, 2, 3, 4, 5])
                plt.ylim([0.5, 5.5])
                plt.yticks([1, 2,3,4,5], labels=[1,2,3,4,5])
                # plt.yticks(np.arange(0.5, 5.5, 0.5))
                sns.despine(top=False, right=False, bottom=False, left=False, ax=g.ax)
                # g.map_dataframe(sns.stripplot, x=group, y="score", hue="method", legend=False, dodge=True, 
                #     marker="$\circ$", ec="face", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3,
                #     order=hue_order)
                g.map_dataframe(sns.stripplot, x=group, y="score", hue="method", legend=False, dodge=True, 
                    marker="$\circ$", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3,
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
                plt.savefig(f'{save_root}/boxplot_{group}_{expname}_v2.png', bbox_inches='tight', transparent=True, format='png')
                plt.savefig(f'{save_root}/boxplot_{group}_{expname}_v2.svg', bbox_inches='tight', transparent=True, format='svg')
                plt.close()

                if group=='structure': # for the legend
                    font_size = 18
                    figure_height = 7
                    figure_width = 7
                    plt.rcParams.update({'font.size': font_size , 'font.family': 'Helvetica', 'text.usetex': False, "svg.fonttype": 'none'})
                    print(plt.rcParams)
                    plt.tick_params(pad = 10)
                    fig = plt.figure(figsize=(figure_width, figure_height), frameon=False)
                    ax = plt.gca()
                    g=sns.catplot(data=df3, x=group, y="score", hue="method", kind="box", palette=palette, ax=ax, order=hue_order)
                    # if group=='structure':
                    #     g.fig.legend(labels=hue_order, ncol=2, loc='outside right')
                    # plt.setp(ax.get_legend().get_texts(), fontsize='24') # for legend text
                    # plt.setp(ax.get_legend().get_title(), fontsize='24') # for legend title
                    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="right", va='center', rotation_mode='anchor')
                    g.set(ylabel=None)
                    g.set(xlabel=None)
                    plt.ylim([0.5, 5.5])  
                    plt.yticks(ticks=[1, 2, 3, 4, 5], labels=[1,2,3,4,5])
                    # plt.yticks(np.arange(0.5, 5.5, 0.5))
                    sns.despine(top=False, right=False, bottom=False, left=False, ax=g.ax)
                    # g.map_dataframe(sns.stripplot, x=group, y="score", hue="method", legend=False, dodge=True, 
                    #     marker="$\circ$", ec="face", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3,
                    #     order=hue_order)
                    g.map_dataframe(sns.stripplot, x=group, y="score", hue="method", legend=False, dodge=True, 
                        marker="$\circ$", s=5, linewidth=0.1, facecolor=(0, 0, 0), alpha=0.3,
                        order=hue_order)
                    g.set(ylabel='Expert score')
                    g.set(xlabel=group_names[group])
                    # g.map_dataframe(sns.catplot, data=df3, x="method", y="score", hue=group, kind="strip", palette='dark:.25', legend=False, dodge=True)
                    # sns.swarmplot(data=df3, x="method", y="score", hue=group, palette='dark:.25', legend=False, dodge=True)
                    # sns.lineplot(data=df3, x="method", y="score", hue=group, units="court", palette='dark:.7', estimator=None, legend=False)
                    # g.map_dataframe(sns.lineplot, x="method", y="score", hue=group, units="court", estimator=None)
                    # connect line
                    if False:
                        for i in range(num_group):
                            for p1, p2 in zip(g.ax.collections[i].get_offsets().data, g.ax.collections[i+num_group].get_offsets().data):
                                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=palette[i], alpha=0.2)
                    plt.savefig(f'{save_root}/boxplot_{group}_{expname}_v2_legend.png', bbox_inches='tight', transparent=True, format='png')
                    plt.savefig(f'{save_root}/boxplot_{group}_{expname}_v2_legend.svg', bbox_inches='tight', transparent=True, format='svg')
                    plt.close()












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





def plot_violin():


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

    # gene_data_filename = 'gene_data.pkl'
    # cluster_data_filename = 'cluster_data.pkl'
    # vst_filename = 'vst.tsv'

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


















if __name__ == '__main__':
    main_20240708_encoder_comparision()
    plot_search_time_tcga_ncidata()











