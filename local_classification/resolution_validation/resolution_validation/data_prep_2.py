#!/usr/bin/env python
# coding: utf-8

# In[4]:


import mrcfile as mrc
import numpy as np
import os
from sklearn.model_selection import train_test_split
import h5py
import math

from common import common_configuration as com_cfg
from monores import monores_configuration as mres_cfg
from resmap import resmap_configuration as resm_cfg

def isPowerOf2(input):
   return input != 0 and (input & (input - 1) == 0)

def remove_empty_cubes(label_cubes, map_cubes):
    # Remove map cubes containing no non-zero values and the corresponding 
    # label cubes.
    empty_cubes = []
    for cube_i in range(map_cubes.shape[0]):
        if not np.count_nonzero(map_cubes[cube_i, ...]):
            empty_cubes.append(cube_i)
    map_cubes = np.delete(map_cubes, empty_cubes, 0)
    label_cubes = np.delete(label_cubes, empty_cubes, 0)
    return label_cubes, map_cubes, empty_cubes


def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def nearest_power_two(val):
    # Returns the first power of two greater than or equal to value val if val 
    # is positive.
    if val <= 0:
        raise ValueError("val must be positive")
    power = 1
    while power < val:
        power *= 2
    return power

        
def pad_to_power_2_cube(arr, remainder_decisions):
    # Return a power of two cube padded with zeros for any additional 
    # elements appended. Use remainder_decisions to add the ith dimension to 
    # before if true or after if false for the respective dimension.
    cube_edge_length = nearest_power_two(max(arr.shape))
    dim_deltas = cube_edge_length - np.array(arr.shape)
    padding = []
    for count, dim_delta in enumerate(dim_deltas):
        before = dim_delta // 2 
        after = before
        if dim_delta % 2:  # odd -> append the remainder according to remainder_decisions
            if remainder_decisions[count]:
                before += 1
            else:
                after += 1
        tup = (before, after)
        padding.append(tup)
    return np.pad(arr, tuple(padding), mode='constant', constant_values=0.0)


def pad(label, map):
    # Pad label and map. Resolve odd dimensions with a random array of 
    # remainder choices. 
    if label.shape != map.shape:
        raise ValueError("invalid shape")
    remainder_decisions = np.random.choice([False, True], label.ndim)
    return (pad_to_power_2_cube(label.data, remainder_decisions), 
                pad_to_power_2_cube(map.data, remainder_decisions))


def mask_elements(target, mask):
    # Mask a target
    if target.shape != mask.shape:
        raise ValueError("invalid shape")
    return np.multiply(target, mask)


def normalize_map(target):
    # Computes feature scaling normalization on a map target data.
    # https://en.wikipedia.org/wiki/Normalization_(statistics)
    min = target.min()
    difference = target.max() - min
    target[:] = ((target - min) / difference)
    return target


def bin_label(target, bounds, bounding_scheme):
    # Bin a target file target_file to the bounds list using np.digitize(...)
    # to map input values to a bucket class in the range [1 - (len(bins) - 1)].
    # Subtract 1 to shift digitized index to align with 0-9 resolution 
    # classification scheme.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html
    target[:] = np.digitize(target, bounds, right=bounding_scheme) - 1
    return target


def pad2(label, map):
    if label.shape != map.shape:
        raise ValueError("invalid shape")
    
    # get max size among all dimension in tuple np.shape(masked_map)
    old_shape = max(np.shape(map))
    new_shape = old_shape

    # check if size is a power of 2
    # if not proceed with padding
    if (not isPowerOf2(old_shape)):
        exponent = math.ceil(math.log(old_shape, 2))
        new_shape = int(math.pow(2, exponent))

        # create temp with new size, all elements = 0
        # copy all elements from target to temp
        # assign temp to target
        temp_map = np.zeros((new_shape, new_shape, new_shape))
        temp_label = np.zeros((new_shape, new_shape, new_shape))
        
        temp_map[:old_shape, :old_shape, :old_shape] = map
        temp_label[:old_shape, :old_shape, :old_shape] = label
        map = temp_map
        label = temp_label  
    return label, map, 

def rebuild(target, empty_cubes, num_16x16x16_cubes, postpad_shape, orig_shape):
    rebuild_target = np.zeros((num_16x16x16_cubes, 16, 16, 16))

    start = 0
    end = 0
    count = 0
    for empty_cube in empty_cubes:
        end = empty_cube
        for i in range (start, end):
            rebuild_target[i] = target[count]
            count += 1
        start = end + 1

    # Uncubify
    rebuild_target = uncubify(rebuild_target, postpad_shape)

    dim1 = orig_shape[0]
    dim2 = orig_shape[1]
    dim3 = orig_shape[2]

    rebuild_target = rebuild_target[:dim1, :dim2, :dim3]
    return rebuild_target

def check_rebuild_result(post_process, orig):
    if np.array_equal(post_process, orig):
        print("          Rebuild is successful!")
    else:
        print("          Rebuild failed!")

def process_data(emdb_path_dict, train_repo_path, test_repo_path, validation_repo_path, bin_boundaries, bin_scheme):
    # Process maps and labels into their respespective repositories for 
    # training, validation, and testing.
    print(" create train, validation, and test repositories...")
    train_repository = h5py.File(train_repo_path, 'w')
    validation_repository = h5py.File(validation_repo_path, 'w')
    test_repository = h5py.File(test_repo_path, 'w')

    non_zero_cube_count = 0

    for id in emdb_path_dict:
        label_file = mrc.open(emdb_path_dict[id]["label"], mode="r")
        map_file = mrc.open(emdb_path_dict[id]["map"], mode="r")
        mask_file = mrc.open(emdb_path_dict[id]["mask"], mode="r")

        label = np.array(label_file.data)
        map = np.array(map_file.data)
        mask = np.array(mask_file.data)

        label_file.close()
        map_file.close()
        mask_file.close()
        
        print(" " + id)
        print("     bin_label()...")
        label = bin_label(label, bin_boundaries, bin_scheme)
        
        print("     normalize_map()...")
        map = normalize_map(map)
        
        print("     mask_elements(label, ...)...")
        label = mask_elements(label, mask)
        print("     mask_elements(map, ...)...")
        map = mask_elements(map, mask)
        
        ###-------NEW CODE-----###
        orig_masked_label = label
        orig_masked_map = map
        ###------END OF NEW CODE-------###
        
        print("     pad(label, map, ...)...")
        #label, map = pad(label, map)
        label, map = pad2(label, map)
        
        print("     cubify(label, ...)...")
        label_cubes = cubify(label, com_cfg.MODEL_INPUT_SHAPE)
        print("     cubify(map, ...)...")
        map_cubes = cubify(map, com_cfg.MODEL_INPUT_SHAPE)

        if label_cubes.shape != map_cubes.shape:
            raise ValueError("Unequal label and map shapes")

        # TODO: removal of cubes prevents protein reconstruction. Determine a 
        # methodology for storing cube location information in the repository
        # for later reconstruction
        print("     remove_empty_cubes(label_cubes, map_cubes)...")
        label_cubes, map_cubes, empty_cubes = remove_empty_cubes(label_cubes, map_cubes)

    
        ###-------NEW CODE-----###
        num_cubes = int(np.shape(label)[0]**3 / (16**3))
        postpad_shape = np.shape(label)
        orig_shape = np.shape(orig_masked_map)
        
        print("     rebuild label...")
        rebuild_label = rebuild(label_cubes, empty_cubes, num_cubes, postpad_shape, orig_shape)
        print("     check_rebuild_result(post_process, orig)...")
        check_rebuild_result(rebuild_label, orig_masked_label)
        print("     rebuild map...")
        rebuild_map = rebuild(map_cubes, empty_cubes, num_cubes, postpad_shape, orig_shape)
        print("     check_rebuild_result(post_process, orig)...")
        check_rebuild_result(rebuild_map, orig_masked_map)
        ###------END OF NEW CODE-------###
        
        non_zero_cube_count += label_cubes.shape[0]

        print("     train_test_split(label_cubes, map_cubes...)...")
        label_cubes_train, label_cubes_test, map_cubes_train, map_cubes_test = train_test_split(
            label_cubes, map_cubes, test_size=com_cfg.TEST_SIZE_RATIO, random_state=326) # random_state is arbitrary
        
        print("     train_test_split(label_cubes_train, map_cubes_train...)...")
        label_cubes_train, label_cubes_train_validation, map_cubes_train, map_cubes_train_validation = train_test_split(
            label_cubes_train, map_cubes_train, test_size=com_cfg.VALIDATION_SET_RATIO, random_state=326) # again, arbitrary

        # Create id_groupings using h5py to store the map and label data for the 
        # current id in the train, validation, and test repositories.
        print("     append datasets to train repository")
        train_repository_id_group = train_repository.create_group(id)
        train_repository_id_group.create_dataset("map", data=map_cubes_train)
        train_repository_id_group.create_dataset("label", data=label_cubes_train)
        
        print("     append datasets to validation repository")
        validation_repository_id_group = validation_repository.create_group(id)
        validation_repository_id_group.create_dataset("map", data=map_cubes_train_validation)
        validation_repository_id_group.create_dataset("label", data=label_cubes_train_validation)

        print("     append datasets to test repository")
        test_repository_id_group = test_repository.create_group(id)
        test_repository_id_group.create_dataset("map", data=map_cubes_test)
        test_repository_id_group.create_dataset("label", data=label_cubes_test)

        print(" total non_zero_cube_count == " + str(non_zero_cube_count))
    
    print(" count train, validation, and test sets, map and label cubes, respectively...")
    train_map_count = 0
    train_label_count = 0
    validation_map_count = 0
    validation_label_count = 0
    test_map_count = 0
    test_label_count = 0
    
    for id in emdb_path_dict:
        train_map_count += train_repository[id]["map"].shape[0]
        train_label_count += train_repository[id]["label"].shape[0]
        validation_map_count += validation_repository[id]["map"].shape[0]
        validation_label_count += validation_repository[id]["label"].shape[0]
        test_map_count += test_repository[id]["map"].shape[0]
        test_label_count += test_repository[id]["label"].shape[0]
    
    print(" train_map_count == " + str(train_map_count) + " && train_label_count == " + str(train_label_count))
    if(train_map_count != train_label_count):
        raise ValueError("Unequal train label and map counts")
    
    print(" validation_map_count == " + str(validation_map_count) + " && validation_label_count == " + str(validation_label_count))
    if(validation_map_count != validation_label_count):
        raise ValueError("Unequal validation label and map counts")
    
    print(" test_map_count == " + str(test_map_count) + " && test_label_count == " + str(test_label_count))
    if(test_map_count != test_label_count):
        raise ValueError("Unequal test label and map counts")
    
    print(" build aggregated train, validation, and test datasets...")
    train_repository.create_dataset("maps", shape=((train_map_count,) + com_cfg.MODEL_INPUT_SHAPE), dtype=np.float32)
    train_repository.create_dataset("labels", shape=((train_label_count,) + com_cfg.MODEL_INPUT_SHAPE), dtype=np.int32)
    validation_repository.create_dataset("maps", shape=((validation_map_count,) + com_cfg.MODEL_INPUT_SHAPE), dtype=np.float32)
    validation_repository.create_dataset("labels", shape=((validation_label_count,) + com_cfg.MODEL_INPUT_SHAPE), dtype=np.int32)
    test_repository.create_dataset("maps", shape=((test_map_count,) + com_cfg.MODEL_INPUT_SHAPE), dtype=np.float32)
    test_repository.create_dataset("labels", shape=((test_label_count,) + com_cfg.MODEL_INPUT_SHAPE), dtype=np.int32)
    
    print(" populate aggregate datasets...")
    accumulated_train_map_index = 0
    accumulated_train_label_index = 0
    accumulated_validation_map_index = 0
    accumulated_validation_label_index = 0
    accumulated_test_map_index = 0
    accumulated_test_label_index = 0

    # Aggregate train, validation, and test datasets into respective maps and 
    # labels datasets. Transpose to remove id grouping from the data structure.
    for id in emdb_path_dict:
        for cube_i in range(train_repository[id]["map"].shape[0]):
            train_repository["maps"][cube_i + accumulated_train_map_index] = train_repository[id]["map"][cube_i]
        accumulated_train_map_index += train_repository[id]["map"].shape[0]
        
        for cube_i in range(train_repository[id]["label"].shape[0]):
            train_repository["labels"][cube_i + accumulated_train_label_index] = train_repository[id]["label"][cube_i]
        accumulated_train_label_index += train_repository[id]["label"].shape[0]

        for cube_i in range(validation_repository[id]["map"].shape[0]):
            validation_repository["maps"][cube_i + accumulated_validation_map_index] = validation_repository[id]["map"][cube_i]
        accumulated_validation_map_index += validation_repository[id]["map"].shape[0]
        
        for cube_i in range(validation_repository[id]["label"].shape[0]):
            validation_repository["labels"][cube_i + accumulated_validation_label_index] = validation_repository[id]["label"][cube_i]
        accumulated_validation_label_index += validation_repository[id]["label"].shape[0]
        
        for cube_i in range(test_repository[id]["map"].shape[0]):
            test_repository["maps"][cube_i + accumulated_test_map_index] = test_repository[id]["map"][cube_i]
        accumulated_test_map_index += test_repository[id]["map"].shape[0]
        
        for cube_i in range(test_repository[id]["label"].shape[0]):
            test_repository["labels"][cube_i + accumulated_test_label_index] = test_repository[id]["label"][cube_i]
        accumulated_test_label_index += test_repository[id]["label"].shape[0]

    train_repository.close()
    validation_repository.close()
    test_repository.close()

def main():
    # Process monores and resmap data 
    print("data_prep.py")
    
    print("process monores...")
    process_data(mres_cfg.EMDB_FILE_PATH_DICT, mres_cfg.TRAIN_REPOSITORY, 
                 mres_cfg.TEST_REPOSITORY, mres_cfg.VALIDATION_REPOSITORY, 
                 mres_cfg.LABEL_BIN_BOUNDARIES, mres_cfg.LABEL_BIN_BOUNDS_SCHEME)
    
    print("process resmap...")
    process_data(resm_cfg.EMDB_FILE_PATH_DICT, resm_cfg.TRAIN_REPOSITORY, 
                 resm_cfg.TEST_REPOSITORY, resm_cfg.VALIDATION_REPOSITORY, 
                 resm_cfg.LABEL_BIN_BOUNDARIES, resm_cfg.LABEL_BIN_BOUNDS_SCHEME)


if __name__ == "__main__":
    main()


# In[ ]:




