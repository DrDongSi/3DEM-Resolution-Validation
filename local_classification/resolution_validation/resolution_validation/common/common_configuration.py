import glob
import os

global BASE_DIR # common root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

global FOLDER_DATA
FOLDER_DATA = os.path.join(BASE_DIR, 'data')

global MAP_FOLDER_PATHS
MAP_FOLDER_PATHS = glob.glob(os.path.join(FOLDER_DATA, '*/'))

global MODEL_INPUT_SHAPE
MODEL_INPUT_SHAPE = (16,16,16)

global TEST_SIZE_RATIO
global VALIDATION_SIZE_RATIO
TEST_SIZE_RATIO = 0.2 # 80% for training -> 20% for test
VALIDATION_SET_RATIO = .10    # 10% of 80% for the validation set

global LAYER_FILTERS
LAYER_FILTERS = [64, 128, 256, 512, 1024]
global DROPOUT_RATES
DROPOUT_RATES = [0.5, 0.5, 0.5, 0.5, 0.5]