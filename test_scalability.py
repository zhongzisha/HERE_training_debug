



import sys,os,shutil,glob
import numpy as np
import h5py


def closest_subset_sum(count_dict, target):
    target = int(float(target))
    # Sort the dictionary by values in descending order
    sorted_items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    
    subset = {}
    current_sum = 0

    # Select items greedily until the sum is close to the target
    for key, value in sorted_items:
        if current_sum + value <= target:
            subset[key] = value
            current_sum += value
        if current_sum == target:
            break

    # Fine-tune: Try to find a better combination if current_sum is not exactly target
    if current_sum < target:
        for key, value in sorted_items:
            if key not in subset and current_sum + value <= target:
                subset[key] = value
                current_sum += value
                if current_sum == target:
                    break

    return subset


def main():
        
    project_name = 'KenData_20240814'
    all_coords = {}
    h5filenames = sorted(glob.glob(os.path.join(f'/data/zhongz2/{project_name}_256/patches/*.h5')))

    for svs_prefix_id, h5filename in enumerate(h5filenames):
        
        svs_prefix = os.path.basename(h5filename).replace('.h5', '')

        with h5py.File(h5filename, 'r') as file:
            coords = file['coords'][()].astype(np.int32)

        all_coords[svs_prefix] = len(coords)


    for target in ['1e5', '1e6', '1e7', '1e8']:

        result = closest_subset_sum(all_coords, target=target)

        save_dir = f'/data/zhongz2/KenData_20240814_{target}_256'
        os.makedirs(os.path.join(save_dir, 'patches'), exist_ok=True)

        os.system('ln -sf "/data/zhongz2/KenData_20240814/svs" "{}"'.format(save_dir))
        os.system('ln -sf "/data/zhongz2/KenData_20240814/feats" "{}"'.format(save_dir))

        for k,v in result.items():
            os.system('ln -sf "/data/zhongz2/KenData_20240814/patches/{}.h5" "{}/patches/"'.format(k, save_dir))



def check_memory_usage():

    import faiss
    import sys,glob,os,shutil
    import psutil

    mapper_dict = {
        # 'IndexFlatIP': 'Original',
        # 'IndexBinaryFlat_ITQ32_LSH': 'ITQ+LSH(32)',
        # 'IndexBinaryFlat_ITQ64_LSH': 'ITQ+LSH(64)',
        # 'IndexBinaryFlat_ITQ128_LSH': 'ITQ+LSH(128)',
        # 'IndexHNSWFlat_m8_IVFPQ_nlist128_m8': 'HNSW+IVFPQ(8,128)',
        # 'IndexHNSWFlat_m8_IVFPQ_nlist256_m8': 'HNSW+IVFPQ(8,256)',
        # 'IndexHNSWFlat_m16_IVFPQ_nlist128_m8': 'HNSW+IVFPQ(16,128)',
        # 'IndexHNSWFlat_m16_IVFPQ_nlist256_m8': 'HNSW+IVFPQ(16,256)',
        'IndexHNSWFlat_m32_IVFPQ_nlist128_m8': 'HNSW+IVFPQ(32,128)',
        # 'IndexHNSWFlat_m32_IVFPQ_nlist256_m8': 'HNSW+IVFPQ(32,256)'
    }

    method = 'HERE_CONCH'
    num_patches = ['1e5', '1e6', '1e7', '1e8', 'TCGA_NCI_CPTAC']

    faiss_bins_dir = '/data/zhongz2/temp_20241204_scalability/faiss_relatedV20240908/faiss_bins'

    mem = psutil.virtual_memory().used/1024/1024/1024

    all_mems = {}
    all_sizes = {}
    all_real_sizes = {}
    for index_name, index_NAME in mapper_dict.items():

        mems = []
        sizes = []
        real_sizes = []
        for num_patch in num_patches:
            if num_patch == 'TCGA_NCI_CPTAC':
                faiss_bin_filename = os.path.join(faiss_bins_dir, f'all_data_feat_before_attention_feat_faiss_{index_name}_TCGA_NCI_CPTAC_HERE_CONCH.bin')
            else:
                faiss_bin_filename = os.path.join(faiss_bins_dir, f'all_data_feat_before_attention_feat_faiss_{index_name}_KenData_20240814_{num_patch}_HERE_CONCH.bin')
            # mem1 = psutil.virtual_memory().used/1024/1024/1024
            index = faiss.read_index(faiss_bin_filename)
            # mem2 = psutil.virtual_memory().used/1024/1024/1024
            # mems.append(mem2 - mem1)
            sizes.append(os.path.getsize(faiss_bin_filename)/1e9)
            real_sizes.append(index.ntotal)
        all_mems[index_NAME] = mems
        all_sizes[index_NAME] = sizes
        all_real_sizes[index_NAME] = real_sizes



    from matplotlib import pyplot as plt

    key = 'HNSW+IVFPQ(32,128)'
    fig, axes = plt.subplots(nrows=1, ncols=1)
    y = all_sizes[key]
    plt.bar(num_patches, all_sizes[key])
    for i in range(len(num_patches)):
        if i!=len(num_patches)-1:
            continue
        plt.text(i, y[i], str(all_real_sizes[key][i]), ha = 'center')
    plt.xlabel('number of patches')
    plt.ylabel('Index size (Gb)'.format(key))
    plt.title('Comparison on database scalability')
    plt.savefig('/data/zhongz2/temp_20241204_scalability/result.png')
    plt.close('all')




