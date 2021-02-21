import numpy as np
import tensorflow as tf
import zipfile as zip_
from pydicom import dcmread
from sklearn.model_selection import GroupKFold
import preprocessing.utils as utl


def normalize(x: np.ndarray):
    return  x / x.max()



def kfold_train_val_split(
        src_zip_path: str, 
        src_lbl_path: str, 
        cross_valid_n_splits: int = 2
    ):
    src_zip = zip_.ZipFile(src_zip_path)
    df_base = get_data(src_zip, src_lbl_path)

    # Prepare cross validation sets    
    group_kfold = GroupKFold(cross_valid_n_splits=2)
    group_kfold.get_n_splits(df_base.file.values, df_base.y.values, df_base.subject.values)

    # generate data
    for train_index, test_index in group_kfold.split(df_base.file.values, df_base.y.values, df_base.subject.values):        
        X_train, X_test = df_base.file.values[train_index], df_base.file.values[test_index]
        y_train, y_test = df_base.y.values[train_index], df_base.y.values[test_index]
        yield X_train, X_test, y_train, y_test
    

def brain_dataset(
        X: np.ndarray,
        y: np.ndarray,
        src_zip_path: str,         
        brain_slice_pos: tuple, 
        brain_segments: tuple = (8,8),        
    ):

    src_zip = zip_.ZipFile(src_zip_path)
    for file, label in zip(X, y):
        source_pixel_array = utl.get_brain_image_pixel_array(src_zip, file)
        target_pixel_array = get_brain_slice(source_pixel_array, brain_slice_pos, brain_segments)
        yield label, target_pixel_array


def brain_tf_dataset(data: brain_dataset, img_height: int = 128, img_width: int = 128):

    dataset = (
        tf
        .data
        .Dataset
        .from_generator(
            data,        
            output_signature=(            
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(img_height, img_width), dtype=tf.float64)
            )
        )
    )

    return dataset