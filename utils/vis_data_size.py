import numpy as np
import os

datasets = ['AUSLAN', 'ArabicDigits', 'CMUsubject16', 'CharacterTrajectories', 'ECG', 'JapaneseVowels', 'KickvsPunch',
            'Libras', 'NetFlow', 'UWave', 'Wafer', 'WalkvsRun']
root_path = '/data3/zyx/project/tsc/data/archives/mts_archive'
for dataset in datasets:
    x_train = np.load(os.path.join(root_path, dataset, 'x_train.npy'))
    print(dataset, 'train size', x_train.shape)
    x_test = np.load(os.path.join(root_path, dataset, 'x_test.npy'))
    print(dataset, 'test size', x_test.shape)