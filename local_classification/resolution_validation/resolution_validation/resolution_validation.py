from common import common_utilities
from common import common_configuration as com_cfg
from monores import monores_configuration as mres_cfg
from resmap import resmap_configuration as resm_cfg
import data_prep

#from tensorflow import reshape
from keras.utils import to_categorical
from keras.utils import Sequence
from keras.layers import Input
from keras.layers import Conv3D
from keras.layers import SpatialDropout3D
from keras.layers import MaxPooling3D
from keras.layers import UpSampling3D
from keras.layers import concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import numpy as np
import h5py


class res_val_seq(Sequence):
    def __init__(self, repository_file, batch_size):
        self.repo = h5py.File(repository_file, mode='r')
        self.batch_size = batch_size

    def __del__(self):
        self.repo.close()

    def __len__(self):
        return int(np.ceil(self.repo['maps'].shape[0] / float(self.batch_size)))

    def __getitem__(self, batch_id):
        offset = batch_id * self.batch_size
        batch_maps = np.array(self.repo['maps'][offset : offset + self.batch_size])
        batch_labels = np.array(self.repo['labels'][offset : offset + self.batch_size])
        batch_maps = np.expand_dims(batch_maps, axis=4)
        batch_labels = to_categorical(batch_labels, num_classes=10, dtype='int8')
        return batch_maps, batch_labels

def build_train_and_evaluate(train_repo, validation_repo, test_repo, folder_logs, filepath_checkpoints, filepath_model):
    print("build training and validation sequences...")
    train_sequence = res_val_seq(train_repo, batch_size=32)
    validation_sequence = res_val_seq(validation_repo, batch_size=32)

    print("build model...")
    model_input = Input(shape=(com_cfg.MODEL_INPUT_SHAPE + (1,)), dtype='float32')

    left_conv_forward_zero_first = Conv3D(filters=com_cfg.LAYER_FILTERS[0] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(model_input)
    print("left_conv_forward_zero_first.shape == " + str(left_conv_forward_zero_first.shape) + " && dtype == " + str(left_conv_forward_zero_first.dtype))
    left_conv_forward_zero_second = Conv3D(filters=com_cfg.LAYER_FILTERS[1] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(left_conv_forward_zero_first)
    print("left_conv_forward_zero_second.shape == " + str(left_conv_forward_zero_second.shape) + " && dtype == " + str(left_conv_forward_zero_second.dtype))
    left_dropout_zero = SpatialDropout3D(rate=com_cfg.DROPOUT_RATES[0])(left_conv_forward_zero_second)
    print("left_dropout_zero.shape == " + str(left_dropout_zero.shape) + " && dtype == " + str(left_dropout_zero.dtype))
    left_pool_down_zero = MaxPooling3D(pool_size=2, strides=2, padding='same')(left_dropout_zero)
    print("left_pool_down_zero.shape == " + str(left_pool_down_zero.shape) + " && dtype == " + str(left_pool_down_zero.dtype))

    left_conv_forward_one_first = Conv3D(filters=com_cfg.LAYER_FILTERS[1] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(left_pool_down_zero)
    print("left_conv_forward_one_first.shape == " + str(left_conv_forward_one_first.shape) + " && dtype == " + str(left_conv_forward_one_first.dtype))
    left_conv_forward_one_second = Conv3D(filters=com_cfg.LAYER_FILTERS[2] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(left_conv_forward_one_first )
    print("left_conv_forward_one_second.shape == " + str(left_conv_forward_one_second.shape) + " && dtype == " + str(left_conv_forward_one_second.dtype))
    left_dropout_one = SpatialDropout3D(rate=com_cfg.DROPOUT_RATES[1])(left_conv_forward_one_second)
    print("left_dropout_one.shape == " + str(left_dropout_one.shape) + " && dtype == " + str(left_dropout_one.dtype))
    left_pool_down_one = MaxPooling3D(pool_size=2, strides=2, padding='same')(left_dropout_one)
    print("left_pool_down_one.shape == " + str(left_pool_down_one.shape) + " && dtype == " + str(left_pool_down_one.dtype))

    left_conv_forward_two_first = Conv3D(filters=com_cfg.LAYER_FILTERS[2] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(left_pool_down_one)
    print("left_conv_forward_two_first.shape == " + str(left_conv_forward_two_first.shape) + " && dtype == " + str(left_conv_forward_two_first.dtype))
    left_conv_forward_two_second = Conv3D(filters=com_cfg.LAYER_FILTERS[3] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(left_conv_forward_two_first)
    print("left_conv_forward_two_second.shape == " + str(left_conv_forward_two_second.shape) + " && dtype == " + str(left_conv_forward_two_second.dtype))
    left_dropout_two = SpatialDropout3D(rate=com_cfg.DROPOUT_RATES[2])(left_conv_forward_two_second)
    print("left_dropout_two.shape == " + str(left_dropout_two.shape) + " && dtype == " + str(left_dropout_two.dtype))
    left_pool_down_two = MaxPooling3D(pool_size=2, strides=2, padding='same')(left_dropout_two)
    print("left_pool_down_two.shape == " + str(left_pool_down_two.shape) + " && dtype == " + str(left_pool_down_two.dtype))

    left_conv_forward_three_first = Conv3D(filters=com_cfg.LAYER_FILTERS[3] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(left_pool_down_two)
    print("left_conv_forward_three_first.shape == " + str(left_conv_forward_three_first.shape) + " && dtype == " + str(left_conv_forward_three_first.dtype))
    left_conv_forward_three_second = Conv3D(filters=com_cfg.LAYER_FILTERS[4] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(left_conv_forward_three_first)
    print("left_conv_forward_three_second.shape == " + str(left_conv_forward_three_second.shape) + " && dtype == " + str(left_conv_forward_three_second.dtype))
    left_dropout_three = SpatialDropout3D(rate=com_cfg.DROPOUT_RATES[3])(left_conv_forward_three_second)
    print("left_dropout_three.shape == " + str(left_dropout_three.shape) + " && dtype == " + str(left_dropout_three.dtype))
    left_pool_down_three = MaxPooling3D(pool_size=2, strides=2, padding='same')(left_dropout_three)
    print("left_pool_down_three.shape == " + str(left_pool_down_three.shape) + " && dtype == " + str(left_pool_down_three.dtype))

    root_conv_forward_four_first = Conv3D(filters=com_cfg.LAYER_FILTERS[4] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(left_pool_down_three)
    print("root_conv_forward_four_first.shape == " + str(root_conv_forward_four_first.shape) + " && dtype == " + str(root_conv_forward_four_first.dtype))
    root_conv_forward_four_second = Conv3D(filters=com_cfg.LAYER_FILTERS[4] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(root_conv_forward_four_first)
    print("root_conv_forward_four_second.shape == " + str(root_conv_forward_four_second.shape) + " && dtype == " + str(root_conv_forward_four_second.dtype))
    root_dropout_four = SpatialDropout3D(rate=com_cfg.DROPOUT_RATES[4])(root_conv_forward_four_second)
    print("root_dropout_four.shape == " + str(root_dropout_four.shape) + " && dtype == " + str(root_dropout_four.dtype))

    right_up_sample_three = UpSampling3D(size=2)(root_dropout_four)
    print("right_up_sample_three.shape == " + str(right_up_sample_three.shape) + " && dtype == " + str(right_up_sample_three.dtype))
    right_conv_up_three = Conv3D(filters=com_cfg.LAYER_FILTERS[3] , kernel_size=2, padding='same', activation='relu', kernel_initializer='he_normal')(right_up_sample_three)
    print("right_conv_up_three.shape == " + str(right_conv_up_three.shape) + " && dtype == " + str(right_conv_up_three.dtype))
    right_merge_threes = concatenate([left_dropout_three, right_conv_up_three], axis=4)
    print("right_merge_threes.shape == " + str(right_merge_threes.shape) + " && dtype == " + str(right_merge_threes.dtype))
    right_conv_forward_three_first = Conv3D(filters=com_cfg.LAYER_FILTERS[3] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(right_merge_threes)
    print("right_conv_forward_three_first.shape == " + str(right_conv_forward_three_first.shape) + " && dtype == " + str(right_conv_forward_three_first.dtype))
    right_conv_forward_three_second = Conv3D(filters=com_cfg.LAYER_FILTERS[3] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(right_conv_forward_three_first)
    print("right_conv_forward_three_second.shape == " + str(right_conv_forward_three_second.shape) + " && dtype == " + str(right_conv_forward_three_second.dtype))
    
    right_up_sample_two = UpSampling3D(size=2)(right_conv_forward_three_second)
    print("right_up_sample_two.shape == " + str(right_up_sample_two.shape) + " && dtype == " + str(right_up_sample_two.dtype))
    right_conv_up_two = Conv3D(filters=com_cfg.LAYER_FILTERS[2] , kernel_size=2, padding='same', activation='relu', kernel_initializer='he_normal')(right_up_sample_two)
    print("right_conv_up_two.shape == " + str(right_conv_up_two.shape) + " && dtype == " + str(right_conv_up_two.dtype))
    right_merge_twos = concatenate([left_dropout_two, right_conv_up_two], axis=4)
    print("right_merge_twos.shape == " + str(right_merge_twos.shape) + " && dtype == " + str(right_merge_twos.dtype))
    right_conv_forward_two_first = Conv3D(filters=com_cfg.LAYER_FILTERS[2] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(right_merge_twos)
    print("right_conv_forward_two_first.shape == " + str(right_conv_forward_two_first.shape) + " && dtype == " + str(right_conv_forward_two_first.dtype))
    right_conv_forward_two_second = Conv3D(filters=com_cfg.LAYER_FILTERS[2] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(right_conv_forward_two_first)
    print("right_conv_forward_two_second.shape == " + str(right_conv_forward_two_second.shape) + " && dtype == " + str(right_conv_forward_two_second.dtype))
    
    right_up_sample_one = UpSampling3D(size=2)(right_conv_forward_two_second)
    print("right_up_sample_one.shape == " + str(right_up_sample_one.shape) + " && dtype == " + str(right_up_sample_one.dtype))
    right_conv_up_one = Conv3D(filters=com_cfg.LAYER_FILTERS[1] , kernel_size=2, padding='same', activation='relu', kernel_initializer='he_normal')(right_up_sample_one)
    print("right_conv_up_one.shape == " + str(right_conv_up_one.shape) + " && dtype == " + str(right_conv_up_one.dtype))
    right_merge_ones = concatenate([left_dropout_one, right_conv_up_one], axis=4)
    print("right_merge_ones.shape == " + str(right_merge_ones.shape) + " && dtype == " + str(right_merge_ones.dtype))
    right_conv_forward_one_first = Conv3D(filters=com_cfg.LAYER_FILTERS[1] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(right_merge_ones)
    print("right_conv_forward_one_first.shape == " + str(right_conv_forward_one_first.shape) + " && dtype == " + str(right_conv_forward_one_first.dtype))
    right_conv_forward_one_second = Conv3D(filters=com_cfg.LAYER_FILTERS[1] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(right_conv_forward_one_first)
    print("right_conv_forward_one_second.shape == " + str(right_conv_forward_one_second.shape) + " && dtype == " + str(right_conv_forward_one_second.dtype))

    right_up_sample_zero = UpSampling3D(size=2)(right_conv_forward_one_second)
    print("right_up_sample_zero.shape == " + str(right_up_sample_zero.shape) + " && dtype == " + str(right_up_sample_zero.dtype))
    right_conv_up_zero = Conv3D(filters=com_cfg.LAYER_FILTERS[0] , kernel_size=2, padding='same', activation='relu', kernel_initializer='he_normal')(right_up_sample_zero)
    print("right_conv_up_zero.shape == " + str(right_conv_up_zero.shape) + " && dtype == " + str(right_conv_up_zero.dtype))
    right_merge_zeros = concatenate([left_dropout_zero, right_conv_up_zero], axis=4)
    print("right_merge_zeros.shape == " + str(right_merge_zeros.shape) + " && dtype == " + str(right_merge_zeros.dtype))
    right_conv_forward_zero_first = Conv3D(filters=com_cfg.LAYER_FILTERS[0] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(right_merge_zeros)
    print("right_conv_forward_zero_first.shape == " + str(right_conv_forward_zero_first.shape) + " && dtype == " + str(right_conv_forward_zero_first.dtype))
    right_conv_forward_zero_second = Conv3D(filters=com_cfg.LAYER_FILTERS[0] , kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(right_conv_forward_zero_first)
    print("right_conv_forward_zero_second.shape == " + str(right_conv_forward_zero_second.shape) + " && dtype == " + str(right_conv_forward_zero_second.dtype))
    
    logits = Conv3D(filters=10, kernel_size=1, padding='same', activation='softmax', kernel_initializer = 'he_normal')(right_conv_forward_zero_second)
    print("logits.shape == " + str(logits.shape) + " && dtype == " + str(logits.dtype))
    model = Model(inputs=[model_input], outputs=[logits])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'categorical_crossentropy'])
    model.summary()

    print("fit model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)
    tensor_board = TensorBoard(log_dir=folder_logs)
    checkpoints = ModelCheckpoint(filepath=filepath_checkpoints, verbose=1, save_best_only=True)
    model.fit_generator(train_sequence, steps_per_epoch=300, epochs=1000000, validation_data=validation_sequence, callbacks=[early_stop, tensor_board, checkpoints])
    model.save(filepath_model)
    
    print("build evaluation sequence...")
    test_sequence = res_val_seq(test_repo, batch_size=32)

    print("evaluate model...")
    results = model.evaluate_generator(test_sequence)
    print("results == " + str(results))

def main():
    data_prep.main()    # prep the data beforehand
    print("resolution_validation.py")
    
    print("build_train_and_evaluate(monoRes...)")
    build_train_and_evaluate(mres_cfg.TRAIN_REPOSITORY, mres_cfg.VALIDATION_REPOSITORY, 
                             mres_cfg.TEST_REPOSITORY, mres_cfg.FOLDER_LOGS, 
                             mres_cfg.FILEPATH_CHECKPOINTS, mres_cfg.FILEPATH_MODEL)
    
    print("build_train_and_evaluate(resMap...)")
    build_train_and_evaluate(resm_cfg.TRAIN_REPOSITORY, resm_cfg.VALIDATION_REPOSITORY, 
                             resm_cfg.TEST_REPOSITORY, resm_cfg.FOLDER_LOGS, 
                             resm_cfg.FILEPATH_CHECKPOINTS, resm_cfg.FILEPATH_MODEL)



if __name__ == "__main__":
    main()

