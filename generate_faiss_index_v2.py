
import sys,os,shutil,io,time,json,pickle,gc,glob
import numpy as np
import h5py
import torch
import idr_torch
from common import BACKBONE_DICT
import faiss


def gen_randomly_samples_for_faiss_train_random10000(project_name='KenData', backbone='ProvGigaPath', dim2=256, HERE_ckpt_filename=None, save_dir=None, version=''):
    prefix = 'HERE_'
    save_filename = os.path.join(save_dir, f'randomly_background_samples_for_train_{project_name}_{prefix}{backbone}{version}.pkl')
    if os.path.exists(save_filename):
        return save_filename

    # project_name = 'KenData'
    # project_name = 'ST'
    # project_name = 'TCGA-COMBINED'
    # backbone = 'ProvGigaPath'
    dim1 = BACKBONE_DICT[backbone]  # 1536
    os.makedirs(save_dir, exist_ok=True)
    feats_dir = f'/data/zhongz2/{project_name}_256/feats/{backbone}/pt_files'

    # HERE_ProvGigaPath_ptfilename = f'/data/zhongz2/temp29/debug/results/ngpus2_accum4_backbone{backbone}_dropout0.25/split_1/snapshot_39.pt'
    # state_dict = torch.load(HERE_ProvGigaPath_ptfilename, map_location='cpu')
    state_dict = torch.load(HERE_ckpt_filename, map_location='cpu')
    attention_net_W = state_dict['MODEL_STATE']['attention_net.0.weight'].T
    attention_net_b = state_dict['MODEL_STATE']['attention_net.0.bias']

    feats_filenames = glob.glob(os.path.join(feats_dir, '*.pt'))
    selected_inds = np.random.choice(np.arange(len(feats_filenames)), size=min(500, len(feats_filenames)), replace=False)
    randomly_selected_data1 = {}
    randomly_selected_data2 = {}
    embeddings1 = []
    embeddings2 = []
    count = 0
    for _, i in enumerate(selected_inds):
        if count == 100:
            break
        svs_prefix = os.path.basename(feats_filenames[i]).replace('.h5', '')
        feats0 = torch.load(feats_filenames[i])
        if feats0.shape[0]<100:
            continue
        selected_inds2 = np.random.choice(np.arange(len(feats0)), size=feats0.shape[0]//2 if feats0.shape[0]<100 else 100, replace=False)
        feats0 = feats0[selected_inds2].reshape(len(selected_inds2), -1)

        feats = feats0.float() @ attention_net_W + attention_net_b
        feats = feats.numpy()
        feats /= np.linalg.norm(feats, axis=1)[:, None]
        embeddings1.append(feats.reshape(-1, dim2))

        feats = feats0.numpy()
        feats /= np.linalg.norm(feats, axis=1)[:, None]
        embeddings2.append(feats.reshape(-1, dim1))
        count += 1

    embeddings1 = np.concatenate(embeddings1, axis=0)
    print(project_name, embeddings1.shape)
    embeddings2 = np.concatenate(embeddings2, axis=0)
    print(project_name, embeddings2.shape)
    randomly_selected_data1[project_name] = {'embeddings': embeddings1}
    randomly_selected_data2[project_name] = {'embeddings': embeddings2}
    
    with open(save_filename, 'wb') as fp:
        pickle.dump({f'HERE_{backbone}': randomly_selected_data1, backbone: randomly_selected_data2}, fp)

    return save_filename

def add_feats_to_faiss(project_name='KenData', backbone='ProvGigaPath', HERE_ckpt_filename=None, save_dir=None, train_data_filename=None, version=''): # HERE_ProvGigaPath
    if not os.path.exists(train_data_filename):
        raise ValueError("first generate train data")
    prefix = 'HERE_'
    project_names = [project_name]
    d = BACKBONE_DICT[backbone]
    if prefix != '':
        d = 256  # HERE_*
    os.makedirs(save_dir, exist_ok=True)
    feats_dir = f'/data/zhongz2/{project_name}_256/feats/{backbone}/pt_files'
    patches_dir = f'/data/zhongz2/{project_name}_256/patches'

    # HERE_ProvGigaPath_ptfilename = f'/data/zhongz2/temp29/debug/results/ngpus2_accum4_backbone{backbone}_dropout0.25/split_1/snapshot_39.pt'
    state_dict = torch.load(HERE_ckpt_filename, map_location='cpu')
    attention_net_W = state_dict['MODEL_STATE']['attention_net.0.weight'].T
    attention_net_b = state_dict['MODEL_STATE']['attention_net.0.bias']

    faiss_bins_dir = '{}/faiss_bins/'.format(save_dir)
    os.makedirs(faiss_bins_dir, exist_ok=True)

    # load randomly samples to train
    with open(train_data_filename, 'rb') as fp:
        randomly_train_data = pickle.load(fp)
        train_data_float32 = randomly_train_data[f'{prefix}{backbone}'][project_name]['embeddings']
        del randomly_train_data

    #
    ITQ_Dims = [32, 64, 128, 256]
    Ms = [4, 8, 16, 32, 64]
    nlists = [128, 256, 512, 1024, 2048]
    #
    ITQ_Dims = [32, 64, 128]
    Ms = [8, 16, 32]
    nlists = [128, 256, 512]
    #
    ITQ_Dims = []
    Ms = [32]
    nlists = [128]

    faiss_types = [('IndexFlatIP', None), ('IndexFlatL2', None)]
    faiss_types.extend(
        [(f'IndexBinaryFlat_ITQ{dd}_LSH', dd) for dd in ITQ_Dims])
    for m in Ms:
        for nlist in nlists:
            faiss_types.append(
                (f'IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8', m, nlist))

    faiss_types1 = []
    for params in faiss_types:
        faiss_type = params[0]
        done = True
        for proj_id, project_name in enumerate(project_names):
            save_filename = f'{faiss_bins_dir}/all_data_feat_before_attention_feat_faiss_{faiss_type}_{project_name}_{prefix}{backbone}.bin'
            if project_name == 'KenData' and faiss_type == 'IndexFlatL2':
                continue
            if project_name == 'TCGA-COMBINED' and faiss_type == 'IndexFlatL2':
                continue
            if not os.path.exists(save_filename):
                done = False
                break
        if not done:
            faiss_types1.append(params)
    faiss_types = faiss_types1
    if len(faiss_types) == 0:
        return

    print('faiss_types', faiss_types)
    index_splits = np.array_split(
        np.arange(len(faiss_types)), indices_or_sections=idr_torch.world_size)

    faiss_types_sub = [faiss_types[i] for i in index_splits[idr_torch.rank]]
    for params in faiss_types_sub:
        faiss_type = params[0]

        quantizer = None
        binarizer = None

        if 'ITQ' in faiss_type:
            binarizer = faiss.index_factory(d, "ITQ{},LSH".format(params[1]))

        if 'HNSW' in faiss_type:
            quantizer = faiss.IndexHNSWFlat(d, params[1])

        if binarizer is not None:
            binarizer.train(train_data_float32)

        for proj_id, project_name in enumerate(project_names):
            if project_name == 'KenData' and (faiss_type == 'IndexFlatL2' or faiss_type == 'IndexFlatIP'):
                continue
            if project_name == 'TCGA-COMBINED' and (faiss_type == 'IndexFlatL2' or faiss_type == 'IndexFlatIP'):
                continue

            save_filename = f'{faiss_bins_dir}/all_data_feat_before_attention_feat_faiss_{faiss_type}_{project_name}_{prefix}{backbone}.bin'
            if os.path.exists(save_filename):
                continue

            h5filenames = sorted(glob.glob(patches_dir + '/*.h5'))

            if faiss_type == 'IndexFlatL2':
                index = faiss.IndexFlatL2(d)
            elif faiss_type == 'IndexFlatIP':
                index = faiss.IndexFlatIP(d)
            elif 'IndexBinaryFlat_ITQ' in faiss_type:
                index = faiss.IndexBinaryFlat(params[1])
            elif 'HNSW' in faiss_type:
                index = faiss.IndexIVFPQ(quantizer, d, params[2], params[1], 8)
                index.train(train_data_float32)
            else:
                print('wrong faiss type')
                sys.exit(0)

            for file_index, h5filename in enumerate(h5filenames):

                svs_prefix = os.path.basename(h5filename).replace('.h5', '')

                feats = torch.load(os.path.join(feats_dir, svs_prefix+'.pt'))
                feats = feats.float() @ attention_net_W + attention_net_b
                feats = feats.cpu().numpy().reshape(-1, d)

                feats /= np.linalg.norm(feats, axis=1)[:, None]

                if file_index % 500 == 0:
                    print(project_name, svs_prefix)

                if faiss_type == 'IndexBinaryFlat':
                    # feats is [-1, 1]
                    # [-1, 1] --> [0, 256]
                    feats = (feats + 1.) * 128
                    feats = np.clip(np.round(feats), 0, 256).astype(np.uint8)

                if 'Binary' in faiss_type:
                    feats = binarizer.sa_encode(feats)

                index.add(feats) 

            print('saving faiss index')
            if 'Binary' in faiss_type:
                with open(save_filename, 'wb') as fp:
                    pickle.dump({'binarizer': binarizer,
                                 'index': faiss.serialize_index_binary(index)}, fp)
            else:
                faiss.write_index(index, save_filename)
            del index
            gc.collect()



def merge_background_samples_for_deployment():
    import pickle
    import numpy as np
    with open('randomly_1000_data_with_ProvGigaPath.pkl', 'rb') as fp:
        data = pickle.load(fp)
    del data['ProvGigaPath']
    del data['HERE_PLIP']['KenData']
    del data['HERE_PLIP']['ST']
    for method in ['HERE_ProvGigaPath', 'HERE_CONCH', 'HERE_PLIP']:
        if method not in data:
            data[method] = {}
        version = 'V4'
        # if method == 'HERE_ProvGigaPath':
        #     version = 'V2'
        #     version = 'V4'
        # if method == 'HERE_CONCH':
        #     version = 'V3'
        #     version = 'V4' # 20240805 using 10000 samples for training
        for project_name in ['KenData', 'ST']:
            if project_name in data[method]:
                continue
            with open(f'randomly_background_samples_for_train_{project_name}_{method}{version}.pkl', 'rb') as fp:
                data1 = pickle.load(fp)   
            data[method][project_name] = data1[method][project_name]['embeddings']
    with open('randomly_1000_data_with_PLIP_ProvGigaPath_CONCH.pkl', 'wb') as fp:
        pickle.dump(data, fp)

def main():     
    version = 'V4'
    project_name='KenData'
    backbone='CONCH'
    dim2=256
    BEST_SPLIT=3
    BEST_EPOCH=53
    HERE_ckpt_filename=f'/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone{backbone}_dropout0.25/split_{BEST_SPLIT}/snapshot_{BEST_EPOCH}.pt'
    save_dir=f'/data/zhongz2/temp_20240801/faiss_related{version}'
    train_data_filename = gen_randomly_samples_for_faiss_train_random10000(project_name=project_name, backbone=backbone, dim2=dim2, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, version=version)
    add_feats_to_faiss(project_name=project_name, backbone=backbone, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, train_data_filename=train_data_filename)

    version = 'V4'
    project_name='ST'
    backbone='CONCH'
    dim2=256
    BEST_SPLIT=3
    BEST_EPOCH=53
    HERE_ckpt_filename=f'/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone{backbone}_dropout0.25/split_{BEST_SPLIT}/snapshot_{BEST_EPOCH}.pt'
    save_dir=f'/data/zhongz2/temp_20240801/faiss_related{version}'
    train_data_filename = gen_randomly_samples_for_faiss_train_random10000(project_name=project_name, backbone=backbone, dim2=dim2, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, version=version)
    add_feats_to_faiss(project_name=project_name, backbone=backbone, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, train_data_filename=train_data_filename)



    # HERE_ProvGigaPath
    version = 'V4'
    project_name='KenData'
    backbone='ProvGigaPath'
    dim2=256
    BEST_SPLIT=1
    BEST_EPOCH=39
    HERE_ckpt_filename=f'/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone{backbone}_dropout0.25/split_{BEST_SPLIT}/snapshot_{BEST_EPOCH}.pt'
    save_dir=f'/data/zhongz2/temp_20240801/faiss_related{version}'
    train_data_filename = gen_randomly_samples_for_faiss_train_random10000(project_name=project_name, backbone=backbone, dim2=dim2, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, version=version)
    add_feats_to_faiss(project_name=project_name, backbone=backbone, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, train_data_filename=train_data_filename)

    version = 'V4'
    project_name='ST'
    backbone='ProvGigaPath'
    dim2=256
    BEST_SPLIT=1
    BEST_EPOCH=39
    HERE_ckpt_filename=f'/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone{backbone}_dropout0.25/split_{BEST_SPLIT}/snapshot_{BEST_EPOCH}.pt'
    save_dir=f'/data/zhongz2/temp_20240801/faiss_related{version}'
    train_data_filename = gen_randomly_samples_for_faiss_train_random10000(project_name=project_name, backbone=backbone, dim2=dim2, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, version=version)
    add_feats_to_faiss(project_name=project_name, backbone=backbone, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, train_data_filename=train_data_filename)

    # HERE_PLIP
    version = 'V4'
    project_name='KenData'
    backbone='PLIP'
    dim2=256
    BEST_SPLIT=3
    BEST_EPOCH=66
    HERE_ckpt_filename=f'/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone{backbone}_dropout0.25/split_{BEST_SPLIT}/snapshot_{BEST_EPOCH}.pt'
    save_dir=f'/data/zhongz2/temp_20240801/faiss_related{version}'
    train_data_filename = gen_randomly_samples_for_faiss_train_random10000(project_name=project_name, backbone=backbone, dim2=dim2, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, version=version)
    add_feats_to_faiss(project_name=project_name, backbone=backbone, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, train_data_filename=train_data_filename)

    version = 'V4'
    project_name='ST'
    backbone='PLIP'
    dim2=256
    BEST_SPLIT=3
    BEST_EPOCH=66
    HERE_ckpt_filename=f'/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone{backbone}_dropout0.25/split_{BEST_SPLIT}/snapshot_{BEST_EPOCH}.pt'
    save_dir=f'/data/zhongz2/temp_20240801/faiss_related{version}'
    train_data_filename = gen_randomly_samples_for_faiss_train_random10000(project_name=project_name, backbone=backbone, dim2=dim2, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, version=version)
    add_feats_to_faiss(project_name=project_name, backbone=backbone, HERE_ckpt_filename=HERE_ckpt_filename, save_dir=save_dir, train_data_filename=train_data_filename)

