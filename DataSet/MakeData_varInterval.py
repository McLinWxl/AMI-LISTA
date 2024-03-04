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
    'SNR': 0,
    'MC': 1000,
    'num_snapshot': 256,
    }

intervels = np.arange(2, 42, 2)
DOAs = np.zeros((121, 2)) + 0.5
for i in range(len(intervels)):
    DOAs[i, 1] = 0.5 + intervels[i]

Angles = np.arange(configs['Start'], configs['End'] + configs['Interval'], configs['Interval'])
num_meshes = len(Angles)

RawData = np.zeros((len(intervels), configs['MC'], configs['num_sensor'], configs['num_snapshot']), dtype=np.complex64)
Label = np.zeros((len(intervels), configs['MC'], num_meshes, 1), dtype=np.float32)
for i in range(len(intervels)):
    DoA_pairs = DOAs[i].reshape(1, 2)
    DG = DataGenerator(DoA_pairs, is_train=False, snr_db=configs['SNR'], repeat=configs['MC'], num_snapshot=configs['num_snapshot'])
    raw_data, label = DG.get_raw_label()
    RawData[i] = raw_data
    Label[i] = label

with h5py.File(f'{configs["dataset_path"]}TestData_varSeparation.h5', 'w') as f:
    f.create_dataset('RawData', data=RawData)
    f.create_dataset('LabelPower', data=Label)

