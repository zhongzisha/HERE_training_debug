








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
    save_root = '/Users/zhongz2/down/temp_20240830/encoder_comparison'
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
        methods = ['YottixelKimiaNet', 'RetCCL', 'DenseNet121', 'HIPT', 'CLIP', 'PLIP', 'HiDARE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HiDARE_ProvGigaPath', 'CONCH', 'HiDARE_CONCH']
        label_names = ['Yottixel', 'RetCCL', 'SISH', 'HIPT', 'CLIP', 'PLIP', 'HERE_PLIP', 'MobileNetV3', 'ProvGigaPath', 'HERE_ProvGigaPath', 'CONCH', 'HERE_CONCH']
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
    for name in ['Acc', 'Percision']:
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
        selected_methods = ['RetCCL', 'HIPT', 'SISH', 'CLIP', 'Yottixel', 'PLIP', 'MobileNetV3', 'ProvGigaPath', 'CONCH']
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
        for ci, tick_label in enumerate(g.get_xticklabels()):
            tick_label.set_color(palette[ci])
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

        COMMON_COLORS = {
            'mobilenetv3': 'orange',
            'MobileNetV3': 'orange',
            'CLIP': 'Green',
            'PLIP': 'gray',
            'ProvGigaPath': 'purple',
            'CONCH': 'blue'
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
        
        for ci, tick_label in enumerate(g.get_xticklabels()):
            tick_label.set_color(palette[ci])
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

        # mobilenetv3_4 32 26.966785516013267
        # CLIP_4 97 26.463153515114186
        # PLIP_4 66 29.2551421379361
        # ProvGigaPath_4 39 33.241507276672884
        # CONCH_4 53 34.164964506392046
        all_df2['marker_size_by_tcga_scores'] = [v*10 for v in [
            33.241507276672884, 34.164964506392046, 29.2551421379361,
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

        all_df2['x'] = [0.55*228217640448, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944, 2.5*17738386944]
        all_df2['y'] = [37.098852, 35.521479+0.6, 35.521479, 35.521479-0.6, 32.989012, 32.989012-0.6, 32.989012-1.2, 32.989012-1.8, 32.989012-2.4]
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
            sns.despine(top=False, right=False, left=False, bottom=False, ax=g.ax)
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
        'purple', 'mediumpurple', 'blue', 'royalblue', 'gray', 'lightgray'
    ]
    for name in ['Acc', 'Percision']:
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
        selected_methods = ['HERE_PLIP', 'PLIP', 'ProvGigaPath', 'CONCH', 'HERE_ProvGigaPath', 'HERE_CONCH']
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
            sns.despine(top=False, right=False, left=False, bottom=False, ax=g.ax)
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

    root = '/Volumes/data-1/temp_20240801'
    root = '/Volumes/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/'
    save_root = '/Users/zhongz2/down/temp_20240801/hashing_comparison2'
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
    for name in ['Acc', 'Percision']:
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
            for ci, tick_label in enumerate(g.get_xticklabels()):
                tick_label.set_color(palette[ci])
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
                sns.despine(top=False, right=False, left=False, bottom=False, ax=g.ax)
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
                sns.despine(top=False, right=False, left=False, bottom=False, ax=g.ax)
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











