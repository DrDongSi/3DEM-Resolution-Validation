import os
import sys # give me access to the parent directory by appending it to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import glob
import numpy as np
from time import time

from common import common_configuration as common_config
from common import common_utilities as utils

global LABEL_FILE_EXTENSION
global MAP_FILE_EXTENSION
global MASK_FILE_EXTENSION
LABEL_FILE_EXTENSION = ".mrc"
MAP_FILE_EXTENSION = ".map"
MASK_FILE_EXTENSION = ".mrc"

global BASE_DIR # monores root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

global FOLDER_DATA
FOLDER_DATA = os.path.join(BASE_DIR, 'data')

global FOLDER_LOGS
FOLDER_LOGS = os.path.join(BASE_DIR, './logs/log_{}'.format(time()))
if not os.path.exists(FOLDER_LOGS):
    os.makedirs(FOLDER_LOGS)

global FILEPATH_MODEL
FILEPATH_MODEL = os.path.join(BASE_DIR, 'model_weights_{}.h5'.format(time()))

global FILEPATH_CHECKPOINTS
FILEPATH_CHECKPOINTS = os.path.join(BASE_DIR, 'checkpoints_weights_{}.h5'.format(time()))

global LABEL_FOLDER_PATHS
LABEL_FOLDER_PATHS = glob.glob(os.path.join(FOLDER_DATA, '*/'))

global LABEL_ID_SET
LABEL_ID_SET = utils.build_label_id_set(LABEL_FOLDER_PATHS, LABEL_FILE_EXTENSION)

global MASK_FOLDER_PATHS
MASK_FOLDER_PATHS = glob.glob(os.path.join(FOLDER_DATA, '*/'))

global EMDB_ID_FOLDER_PATHS
global EMDB_FILE_PATH_DICT
EMDB_FILE_PATH_DICT = utils.build_emdb_file_path_dict(LABEL_ID_SET, 
                                                      LABEL_FOLDER_PATHS, 
                                                      common_config.MAP_FOLDER_PATHS, 
                                                      MASK_FOLDER_PATHS,
                                                      LABEL_FILE_EXTENSION, 
                                                      MAP_FILE_EXTENSION, 
                                                      MASK_FILE_EXTENSION)

global TRAIN_REPOSITORY
global VALIDATION_REPOSITORY
global TEST_REPOSITORY
TRAIN_REPOSITORY = os.path.join(FOLDER_DATA, 'train_repository.h5')
VALIDATION_REPOSITORY = os.path.join(FOLDER_DATA, 'validation_repository.h5')
TEST_REPOSITORY = os.path.join(FOLDER_DATA, 'test_repository.h5')

global LABEL_MIN_VAL
global LABEL_MAX_VAL
LABEL_MIN_VAL = utils.abs_min_val(EMDB_FILE_PATH_DICT, "label")
LABEL_MAX_VAL = utils.abs_max_val(EMDB_FILE_PATH_DICT, "label")

# Maintains monotonic sequence in bin boundaries when attempting to run data
# preparation on previousily prepared data.
if LABEL_MIN_VAL > 0:
    LABEL_MIN_VAL = 0
if LABEL_MAX_VAL <= 10:
    LABEL_MAX_VAL = 11

global LABEL_BIN_BOUNDARIES
global LABEL_BIN_BOUNDS_SCHEME
LABEL_BIN_BOUNDARIES = np.array([LABEL_MIN_VAL - 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10, LABEL_MAX_VAL])
LABEL_BIN_BOUNDS_SCHEME = True