




def check_results_forTCGA_v2_20250409(subset='test', model_name='CONCH', csv_filename=None, do_filter=False, save_root=None, results_dir=None):

    import os
    import pandas as pd
    import numpy as np
    from common import CLASSIFICATION_DICT, REGRESSION_LIST, IGNORE_INDEX_DICT, PAN_CANCER_SITES
    import torch
    from sklearn.metrics import confusion_matrix, f1_score, auc, roc_auc_score, roc_curve, classification_report, r2_score
    from scipy.stats import percentileofscore, pearsonr, spearmanr

        
    def softmax_stable(x):  # only 2-D
        x = np.exp(x - np.max(x, axis=1)[:, None])
        return x / x.sum(axis=1)[:, None]

    if save_root is not None:
        os.makedirs('{}/per-cancer'.format(save_root), exist_ok=True)

    all_labels = pd.read_csv(csv_filename, low_memory=False)
    all_labels['cancer_type'] = all_labels['PanCancerSiteID'].map({site_id+1: site_name for site_id, site_name in enumerate(PAN_CANCER_SITES)})
    all_labels['svs_prefix'] = [os.path.splitext(os.path.basename(row['DX_filename']))[0] for _, row in all_labels.iterrows()]
    all_labels = all_labels.set_index('svs_prefix')

    results = {}
    for svs_prefix, row in all_labels.iterrows():
        results_dict = torch.load(os.path.join(results_dir, svs_prefix+'.pt'))
        result = {}
        for k, v in CLASSIFICATION_DICT.items():
            result[k] = results_dict[k+'_logits']
        for k in REGRESSION_LIST:
            result[k] = results_dict[k+'_logits']
        results[svs_prefix] = result

    result_df = pd.DataFrame(results).T
    result_df.index.name = 'svs_prefix'
    result_df = result_df.reset_index(drop=False)
    result_df['cancer_type'] = all_labels.loc[result_df['svs_prefix'].values, 'cancer_type'].values
    result_df1 = result_df

    all_scores = {}

    for cancer_type in result_df1['cancer_type'].unique().tolist() + ['PanCancer']:

        if cancer_type == 'PanCancer':
            result_df2 = result_df1.copy().reset_index(drop=True)
        else:
            result_df2 = result_df1[result_df1['cancer_type'] == cancer_type].reset_index(drop=True)
        if len(result_df2) == 0:
            continue

        barcodes = []
        for svs_prefix in result_df2['svs_prefix'].values:
            found = []
            for v in all_labels.index.values:
                if v in svs_prefix:
                    found.append(v)
            if len(found) == 1:  # exact one match
                barcodes.append(found[0])
            elif len(found) == 0: # no match
                barcodes.append('')
            elif svs_prefix in found: # multi match, has one exact match
                barcodes.append(svs_prefix)
            else: 
                print(svs_prefix, found)
                barcodes.append('')
            
        result_df2['barcode'] = barcodes

        result_df2 = result_df2[result_df2['barcode'].isin(all_labels.index)].reset_index(drop=True)
        labels = all_labels.loc[result_df2['barcode'].values]

        if len(result_df2) ==0 or len(labels) == 0:
            continue

        if save_root is not None:
            labels.to_csv(f'{save_root}/per-cancer/{cancer_type}_{model_name}_labels.csv')
            result_df2.to_csv(f'{save_root}/per-cancer/{cancer_type}_{model_name}_predictions.csv')

        results = [len(labels)]
        all_gts = []
        all_preds = []
        for k, v in CLASSIFICATION_DICT.items():
            if k not in labels.columns:
                results.append(0)
                all_gts.append(0)
                all_preds.append(0)
                continue

            valid_ind = ~labels[k].isin([np.nan, IGNORE_INDEX_DICT[k]])
            gt = labels.loc[valid_ind, k]
            a,b = np.unique(gt.values, return_counts=True)
            print(k, a, b)
            if do_filter and (cancer_type != 'PanCancer') and ((len(a) == 0) or (len(a) == 1) or (len(b) == 2 and b[1] <= 5)):
                results.append(0)
                continue

            try:
                logits = np.concatenate(result_df2.loc[np.where(valid_ind)[0], k].values)
                probs = softmax_stable(logits)
                probs = np.delete(probs, IGNORE_INDEX_DICT[k], axis=1)
                probs = softmax_stable(probs)
                # logits = np.delete(logits, IGNORE_INDEX_DICT[k], axis=1)
                # probs = softmax_stable(logits)
                preds = np.argmax(probs, axis=1)
                auc = roc_auc_score(y_true=gt, y_score=probs[:, 1], average='weighted', multi_class='ovo',
                                        labels=np.arange(2))
                # cm = confusion_matrix(y_true=gt, y_pred=preds)
                results.append(auc)
                all_gts.append(gt)
                all_preds.append(preds)
            except Exception as error:
                print(k, error)
                results.append(0)
                all_gts.append(0)
                all_preds.append(0)

        import pdb
        pdb.set_trace()

        for k in REGRESSION_LIST:
            if k not in labels.columns and k[5:] not in labels.columns:
                results.append(0)
                continue
            try:
                logits = np.concatenate(result_df2.loc[:, k].values)
                # if 'TIDE_' == k[:5]:
                #     k = k[5:]
                gt = labels.loc[:, k]
                r2score = r2_score(gt, logits)
                pearson_corr, pearsonr_pvalue = pearsonr(gt, logits)
                spearmanr_corr, spearmanr_pvalue = spearmanr(gt, logits)
                results.append(spearmanr_corr)
            except Exception as error:
                print(k, error)
                results.append(0)

        all_scores[cancer_type] = np.array(results).reshape(1, -1)

    results = pd.DataFrame(np.concatenate([v for k,v in all_scores.items()]), columns=['N']+list(CLASSIFICATION_DICT.keys())+REGRESSION_LIST)
    results.index = list(all_scores.keys())
    results['N'] = results['N'].map(int)

    if save_root is not None:
        results.to_csv(f'{save_root}/{model_name}_prediction_scores.csv')
    
    return results


def do_results_TCGA_v2_20250409():

    best_splits = {
        'CONCH': 3,
        'UNI': 3,
        'ProvGigaPath': 1
    }
    best_epochs = {
        'CONCH': 53,
        'UNI': 58,
        'ProvGigaPath': 39
    }
    postfix = ''
    postfix = '_20250409'
    for subset in ['test']:# ['trainval', 'test']:
        for model_name in ['CONCH']: # ['UNI', 'ProvGigaPath', 'CONCH']:
            for do_filter in [True]: # [True, False]:
                split = best_splits[model_name]
                csv_filename = f'/data/zhongz2/temp29/debug/splits/{subset}-{split}.csv'
                save_root = '/data/zhongz2/CPTAC/predictions_v2_TCGA_filter{}_2{}/{}'.format(do_filter, postfix, subset)
                results_dir = f'/data/zhongz2/download/TCGA_{subset}{split}/{model_name}/pred_files'
                check_results_forTCGA_v2_20250409(subset=subset, model_name=model_name, \
                    csv_filename=csv_filename, do_filter=do_filter, save_root=save_root, \
                        results_dir=results_dir)

# 20250409
def check_TCGA_results():

    import pandas as pd
    df = pd.read_csv('/Volumes/data-1/CPTAC/predictions_v2_TCGA_filterTrue_2_20250409/test/CONCH_prediction_scores.csv')


if __name__ == '__main__':
    do_results_TCGA_v2_20250409()