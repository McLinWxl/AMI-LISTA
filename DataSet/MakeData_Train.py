import h5py
from DoaDataGenerator.data_generator import DataGenerator
import os
import numpy as np
abs_path = os.path.abspath(os.path.dirname(__file__))

configs = {
    'dataset_path': f'{abs_path}/Data/',
    'Start': -60,
    'End': 60,
    'Interval': 1,
    'num_sensor': 8,
    'num_snapshot': 256,
    }

Angles = np.arange(configs['Start'], configs['End'] + configs['Interval'], configs['Interval'])
num_meshes = len(Angles)
DOA11 = []
DOA22 = []
k1 = np.arange(1, 41, 1)
for i in range(len(k1)):
    DOA1 = np.arange(configs['Start'], configs['End'] - k1[i] + 1)
    DOA2 = np.arange(configs['Start'] + k1[i], configs['End'] + 1)
    DOA11 = np.concatenate([DOA11, DOA1])
    DOA22 = np.concatenate([DOA22, DOA2])
DOAs = np.vstack([DOA11, DOA22]).T

DG = DataGenerator(DOAs, is_train=True, repeat=30)
RawData, Label = DG.get_raw_label()

# with h5py.File(f'{configs["dataset_path"]}TrainData.h5', 'w') as f:
#     f.create_dataset('RawData', data=RawData)
#     f.create_dataset('LabelPower', data=Label)



