



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















