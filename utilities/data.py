"""This module contains helper functions for data processing"""

import os
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

def one_hot_encode(y) -> np.ndarray:
    """One hot encode utility tool"""
    encoded = np.zeros((len(y), 10))

    for idx, val in enumerate(y):
        encoded[idx][val] = 1

    return encoded

def split_df(df: pd.DataFrame, cols: list, **kwargs) -> pd.DataFrame:
    """Split DataFrame with x number of samples from each labels
    params:
        df: Pandas Dataframe, dataframe to get samples from
        cols: list, column names (which information from df to copy)
        kwargs: format labelname=number of samples, e.g. fire=250 -> 250 'fire' samples
    return:
        Pandas Dataframe, new dataframe with the number of desired samples
    """
    df1 = pd.DataFrame()

    for k, v in kwargs.items():
        df2 = df.loc[df['Label'] == k][cols].sample(n=v)
        df1 = pd.concat([df1, df2])

    df1.index = range(len(df1))
    return df1

def create_callback(_dir: str, log_dir: str) -> tf.keras.callbacks.TensorBoard:
    """Creates a TensorBoard callback instance to store log files
    Format: '_dir/log_dir/current_datetime/'
    params:
        _dir: str, target directory to store log files
        log_dir: str name of log directory (e.g. efficientnet_model_1)
    return:
        Tensorboard log file
    """
    log_file = _dir + '/' + log_dir + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_file
    )
    
    return tensorboard_callback

def walk_through_dir(path: str) -> None:
    """Walk through an image classification directory
    and find out how many files (images) are in each subdirectory
    params:
        path: str, path to target directory
    return:
       None
    """
    for dirpath, dirnames, filenames in os.walk(path):
        print(f'There are {len(dirnames)} directories and {len(filenames)} images in "{dirpath}"')
