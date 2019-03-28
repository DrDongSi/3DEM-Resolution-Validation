import mrcfile as mrc
import os
from collections import defaultdict

def build_label_id_set(label_folder_paths, label_file_ext):
    # Builds a set of label id's from label_folder_paths. We assume a
    # label_folder_path is well-formed and the last folder name is the id.
    label_id_set = set()

    for label_folder_path in label_folder_paths:
        id = os.path.basename(os.path.normpath(label_folder_path))
        label_id_set.add(id)

    return label_id_set


def build_emdb_file_path_dict(id_set, label_folder_paths, map_folder_paths, mask_folder_paths, label_file_ext, map_file_ext, mask_file_ext):
    # Builds a dictionary of dictionaries containing file paths to labels, 
    # maps, and masks keyed by id and label, map or mask. Assumes the files
    # follows the id_specifier.type format.
    file_dict = defaultdict(dict)

    for label_folder_path in label_folder_paths:
        id = os.path.basename(os.path.normpath(label_folder_path))
        if id in id_set:
            file_dict[id]["label"] = str(label_folder_path + id + "_label" + label_file_ext)
    
    for map_folder_path in map_folder_paths:
        id = os.path.basename(os.path.normpath(map_folder_path))
        if id in id_set:
            file_dict[id]["map"] = str(map_folder_path + id + "_map" + map_file_ext)

    for mask_folder_path in mask_folder_paths:
        id = os.path.basename(os.path.normpath(mask_folder_path))
        if id in id_set:
            file_dict[id]["mask"] = str(mask_folder_path + id + "_mask" + mask_file_ext)

    return file_dict

def abs_max_val(emdb_file_path_dict, file_key):
    # Returns the absolute maximum value in a set of mrc files within the 
    # emdb_file_path_dict with the file key file_key.
    maxs = []
    for id in emdb_file_path_dict:
        file = mrc.open(emdb_file_path_dict[id][file_key], mode='r')
        maxs.append(file.data.max())
        file.close()
    return max(maxs)


def abs_min_val(emdb_file_path_dict, file_key):
    # Returns the absolute minimum value in a set of mrc files within the 
    # emdb_file_path_dict with the file key file_key.
    mins = []
    for id in emdb_file_path_dict:
        file = mrc.open(emdb_file_path_dict[id][file_key], mode='r')
        mins.append(file.data.min())
        file.close()
    return min(mins)
