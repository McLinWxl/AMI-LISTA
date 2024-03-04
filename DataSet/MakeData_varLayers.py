import h5py
from DoaDataGenerator.data_generator import DataGenerator
import os
import numpy as np

abs_path = os.path.abspath(os.path.dirname(__file__))

interval = '10'

configs = {
    'dataset_path': f'{abs_path}/Data/',
    'Start': -60,
    'End': 60,
    'Interval': 1,
    'num_sensor': 8,
    'num_snapshot': 256,
    'MC': 100,
    'SNR': 0,
}

Angles = np.arange(configs['Start'], configs['End'] + configs['Interval'], configs['Interval'])
num_meshes = len(Angles)

if interval == '35':
    DOAs = np.array([
        [-0.5, 34.5],
        [-0.4, 34.6],
        [-0.3, 34.7],
        [-0.2, 34.8],
        [-0.1, 34.9],
        [0.0, 35.0],
        [0.1, 35.1],
        [0.2, 35.2],
        [0.3, 35.3],
        [0.4, 35.4],
        [0.5, 35.5],
    ])
elif interval == '10':
    DOAs = np.array([
        [-0.5, 9.5],
        [-0.4, 9.6],
        [-0.3, 9.7],
        [-0.2, 9.8],
        [-0.1, 9.9],
        [0.0, 10.0],
        [0.1, 10.1],
        [0.2, 10.2],
        [0.3, 10.3],
        [0.4, 10.4],
        [0.5, 10.5],
    ])
elif interval == '7':
    DOAs = np.array([
        [-0.5, 6.5],
        [-0.4, 6.6],
        [-0.3, 6.7],
        [-0.2, 6.8],
        [-0.1, 6.9],
        [0.0, 7.0],
        [0.1, 7.1],
        [0.2, 7.2],
        [0.3, 7.3],
        [0.4, 7.4],
        [0.5, 7.5],
    ])


DG = DataGenerator(DOAs, is_train=False, snr_db=configs['SNR'], repeat=configs['MC'])
RawData, Label = DG.get_raw_label()
RawData = RawData.reshape(1, configs['MC']*len(DOAs), configs['num_sensor'], configs['num_snapshot'])
Label = Label.reshape(1, configs['MC']*len(DOAs), num_meshes, 1)

with h5py.File(f'{configs["dataset_path"]}TestData_varLayers_{interval}.h5', 'w') as f:
    f.create_dataset('RawData', data=RawData)
    f.create_dataset('LabelPower', data=Label)
