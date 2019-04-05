#!/usr/bin/env python
# coding: utf-8

# import mrcfile as mrc
# import os
# import monores_configuration as mres_cfig
#  
#  
# def main():
#     print("build a structure for holding our 3 dimensional tensor...\n")
#     for id in EMDB_FILE_PATH_DICT
#     mrc.open(mres_cfig.EMDB_FILE_PATH_DICT['map'])
#     print("Binning label...\n")
#     print("Masking...\n")
#     print("Feature scaling...\n")
#     print("Binning...\n")
#     print("padding...\n")
#     print("Slicing...\n")
#     print("Removing zero cubes...\n")
#     print()

# In[233]:


import numpy as np
import mrcfile as mrc
import pandas as pd
import math
import os
from monores import monores_configuration as mres_cfg
from resmap import resmap_configuration as resm_cfg
from common import common_configuration as com_cfg


# In[234]:


def isPowerOf2(input):
    return input != 0 and (input & (input - 1) == 0)


# In[235]:


def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)


# In[236]:


def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)


# In[237]:


def create_bins(label):
    minVal = label.min()
    maxVal = label.max()
    
    # Maintains monotonic sequence in bin boundaries when attempting to run data
    # preparation on previousily prepared data.
    if minVal > 0:
        minVal = 0
    if maxVal <= 10:
        maxVal = 11
    return np.array([minVal - 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10, maxVal])


# In[238]:


def bin_label(label, bins):
    bin_bounding_scheme = True
    label = np.digitize(label, bins, right=bin_bounding_scheme)
    return label


# In[239]:


def normalize_map(target):
    # Computes feature scaling normalization on a map target data.
    # https://en.wikipedia.org/wiki/Normalization_(statistics)
    target[:] = (target - target.min()) / (target.max() - target.min())
    return target


# In[240]:


def mask_element(target, mask):
    if target.shape != mask.shape:
        raise ValueError("invalid shape")
    return target * mask


# In[241]:


def pad(target):
    # get max size among all dimension in tuple np.shape(masked_map)
    old_shape = max(np.shape(target))
    new_shape = old_shape

    # check if size is a power of 2
    # if not proceed with padding
    if (not isPowerOf2(old_shape)):
        exponent = math.ceil(math.log(old_shape, 2))
        new_shape = int(math.pow(2, exponent))

        # create temp with new size, all elements = 0
        # copy all elements from target to temp
        # assign temp to target
        temp = np.zeros((new_shape, new_shape, new_shape))
        temp[:old_shape, :old_shape, :old_shape] = target
        target = temp
        
    return target


# In[242]:


def numOfCubes(target):
    return int(np.shape(target)[0]**3 / (16**3))


# In[243]:


def remove_empty_cubes(map, label):
    empty_cubes = []
    for cube_i in range(map.shape[0]):
        non_zero = np.count_nonzero(map[cube_i])
        if non_zero == 0:
            empty_cubes.append(cube_i)
        
    map = np.delete(map, empty_cubes, 0)
    label = np.delete(label, empty_cubes, 0)
    return map, label, empty_cubes
    


# In[244]:


def rebuild(target, empty_cubes, num_16x16x16_cubes, postpad_shape, prepad_shape):
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

    dim1 = prepad_shape[0]
    dim2 = prepad_shape[1]
    dim3 = prepad_shape[2]

    rebuild_target = rebuild_target[:dim1, :dim2, :dim3]
    return rebuild_target


# In[247]:


def preprocess_data(emdb_path_dict):
    for id in emdb_path_dict:
        print("   id = {}".format(id))
        map_file = mrc.open(emdb_path_dict[id]['map'])
        mask_file = mrc.open(emdb_path_dict[id]['mask'])
        label_file = mrc.open(emdb_path_dict[id]['label'])
        
        map = np.array(map_file.data)
        mask = np.array(mask_file.data)
        label = np.array(label_file.data)
        
        map_file.close()
        mask_file.close()
        label_file.close()
        
        print("      Binning label...\n")
        bins = create_bins(label)
        label = bin_label(label, bins)

        print("      Normalizing map...\n")
        map = normalize_map(map)
        
        print("      Masking label...\n")
        masked_label = mask_element(label, mask)
        
        print("      Masking map...\n")
        masked_map = mask_element(map, mask)
        
        orig_masked_map = masked_map
        orig_masked_label = masked_label
        
        # Prepad shape tuple
        prepad_shape = np.shape(masked_map)
    
        print("      Padding...\n")
        masked_map = pad(masked_map)
        masked_label = pad(masked_label)
        
        # Postpad shape tuple
        postpad_shape = np.shape(masked_map)
        
        # find number of 16x16x16
        num_cubes = numOfCubes(masked_map)
        
        print("      Cubifying (cutting into multiple cubes of 16x16x16)...\n")
        slice_shape = (16,16,16)
        masked_label = cubify(masked_label, slice_shape)
        masked_map = cubify(masked_map, slice_shape)
        
        print("      Removing empty cubes...\n")
        # keep track of all indices of empty cubes in post-cubify map and label
        # remove empty cubes
        empty_cubes = []
        masked_map, masked_label, empty_cubes = remove_empty_cubes(masked_map, masked_label)

        print("      Rebuilding original map and label...\n")
        rebuild_map = rebuild(masked_map, empty_cubes, num_cubes, postpad_shape, prepad_shape)
        rebuild_label = rebuild(masked_label, empty_cubes, num_cubes, postpad_shape, prepad_shape)
        
        print("      Check if rebuilding map and label is successful...\n")
        result = np.array_equal(rebuild_map, orig_masked_map)
        print("         rebuild_map == orig_masked_map = {}".format(result))
        if result:
            print("            Rebuilding map SUCCESSFUL!\n")
        result = np.array_equal(rebuild_label, orig_masked_label)
        print("         rebuild_label == orig_masked_label = {}".format(result))
        if result:
            print("            Rebuilding label SUCCESSFUL!\n")
            
        print("\n---------------------\n")


# In[ ]:


def main():
    print("Process Monores maps and labels...\n")
    preprocess_data(mres_cfg.EMDB_FILE_PATH_DICT)
    print("\n--------------END OF Monores--------------\n")
    
    print("Process Resmap maps and labels...\n")
    preprocess_data(resm_cfg.EMDB_FILE_PATH_DICT)
    print("\n--------------END OF Monores--------------\n")
    
if __name__ == "__main__":
    main()    




