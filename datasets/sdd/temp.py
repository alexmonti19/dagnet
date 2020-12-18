import numpy as np
import pathlib

dir_path = pathlib.Path('.').absolute() / 'sdd_npy'
dir_path_original = pathlib.Path('.').absolute() / 'sdd_npy_original'

if __name__ == '__main__':
    all_data_orig = np.load(dir_path_original / 'all_data.npy', allow_pickle=True)
    all_data = np.load(dir_path / 'all_data.npy', allow_pickle=True)

    train_original = np.load(dir_path_original / 'train.npy', allow_pickle=True)
    train = np.load(dir_path / 'train.npy', allow_pickle=True)

    val_original = np.load(dir_path_original / 'validation.npy', allow_pickle=True)
    val = np.load(dir_path / 'validation.npy', allow_pickle=True)

    test_original = np.load(dir_path_original / 'test.npy', allow_pickle=True)
    test = np.load(dir_path / 'test.npy', allow_pickle=True)

    ...

