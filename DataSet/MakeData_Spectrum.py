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
    'num_sensor': 4,
    'num_snapshot': 256,
    }

Angles = np.arange(configs['Start'], configs['End'] + configs['Interval'], configs['Interval'])
num_meshes = len(Angles)

DOAs = np.array([
     [-30.5, 30.5],
     [-15.5, 15.5],
     [-10.5, 10.5],
     [-5.5, 5.5],
     [-2.5, 2.5],
     [0.5, 10.5],
     [0.5, 35.5],
     ])

DG = DataGenerator(DOAs, is_train=False, snr_db=0, repeat=1, num_sensors=configs['num_sensor'], num_snapshot=configs['num_snapshot'])
RawData, Label = DG.get_raw_label()

with h5py.File(f'{configs["dataset_path"]}TestSpectrum_{configs["num_sensor"]}.h5', 'w') as f:
    f.create_dataset('RawData', data=RawData)
    f.create_dataset('LabelPower', data=Label)