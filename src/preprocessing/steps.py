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
    df_base = utl.get_data(src_zip, src_lbl_path)

    # Filter bad data
    df_base = df_base[df_base.subject != 105]
    df_base = df_base[-df_base.y.isnull()]

    # Prepare cross validation sets    
    group_kfold = GroupKFold(n_splits=cross_valid_n_splits)
    group_kfold.get_n_splits(df_base.file.values, df_base.y.values, df_base.subject.values)

    # generate data
    for train_index, test_index in group_kfold.split(df_base.file.values, df_base.y.values, df_base.subject.values):        
        X_train, X_test = df_base.file.values[train_index], df_base.file.values[test_index]
        y_train, y_test = df_base.y.values[train_index], df_base.y.values[test_index]
        yield X_train, X_test, y_train, y_test