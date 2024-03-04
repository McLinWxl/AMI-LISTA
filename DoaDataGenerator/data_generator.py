import h5py
import matplotlib.pyplot as plt
import numpy as np

class DataGenerator:
    def __init__(self, DOAs, snr_db=None, is_train=False, repeat=1, num_sensors=8, num_snapshot=256, sensor_interval=0.5, wavelength=1.0, num_meshes=121, bias=60.0):
        """

        :param DOAs: Sample, Sources
        :param num_sensors:
        :param num_snapshot:
        :param snr_db:
        :param sensor_interval:
        :param wavelength:
        """
        self.DOAs = np.repeat(DOAs[np.newaxis, :, :], repeat, axis=0)
        self.num_sensors = num_sensors
        self.num_snapshot = num_snapshot
        self.snr_db = snr_db
        self.sensor_interval = sensor_interval
        self.wavelength = wavelength
        self.num_samples, self.num_sources = DOAs.shape
        self.is_train = is_train
        self.num_meshes = num_meshes
        # print(f'mode: {"train" if self.is_train else "test"}')

        data_all = np.zeros((repeat, self.num_samples, self.num_sources, self.num_sensors, self.num_snapshot),
                            dtype=np.complex64)
        signal_wave_all = np.zeros((repeat, self.num_samples, self.num_sources, 1, self.num_snapshot), dtype=np.complex64)
        self.data = np.zeros((repeat, self.num_samples, self.num_sensors, self.num_snapshot), dtype=np.complex64)
        self.labels = np.zeros((repeat, self.num_samples, self.num_meshes, 1), dtype=np.float32)
        for r in range(repeat):
            for i in range(self.num_samples):
                for j in range(self.num_sources):
                    data_all[r, i, j], _, signal_wave_all[r, i, j] = self.signal_generate(self.DOAs[r, i, j])
                    signal_power = np.matmul(signal_wave_all[r, i, j], signal_wave_all[r, i, j].conj().T).real / self.num_snapshot
                    if np.floor(self.DOAs[r, i, j]) == np.ceil(self.DOAs[r, i, j]):
                        self.labels[r, i, int(self.DOAs[r, i, j] + bias), 0] = 1
                    else:
                        # calculate the left and right power by the distance to the two nearest mesh
                        left = int(np.floor(self.DOAs[r, i, j]) + bias)
                        right = int(np.ceil(self.DOAs[r, i, j]) + bias)
                        right_power = np.square(self.DOAs[r, i, j] - left + bias) / np.square(right - left) * signal_power
                        left_power = np.square(right - self.DOAs[r, i, j] - bias) / np.square(right - left) * signal_power
                        # left_power = (self.DOAs[r, i, j] - left + bias) / (right - left) * signal_power
                        # right_power = (right - self.DOAs[r, i, j] - bias) / (right - left) * signal_power
                        self.labels[r, i, left, 0] = left_power
                        self.labels[r, i, right, 0] = right_power
                    snr_db = self.snr_db if not self.is_train else np.random.uniform(-10, 0)
                    data_all[r, i, j, :, :] = self.add_awgn(data_all[r, i, j, :, :], signal_power, snr_db)
                    self.data[r, i, :, :] += data_all[r, i, j, :, :]

    def get_raw_label(self):
        data = self.data.reshape(-1, self.num_sensors, self.num_snapshot)
        labels = self.labels.reshape(-1, self.num_meshes, 1)
        return data, labels

    def signal_generate(self, DOA):
        steering_vector = np.exp(
            1j * 2 * np.pi * self.sensor_interval * np.arange(self.num_sensors)[:, np.newaxis] * np.sin(
                np.deg2rad(DOA)) / self.wavelength)
        signal_wave = (np.random.randn(1, self.num_snapshot) + 1j * np.random.randn(1, self.num_snapshot))
        return np.matmul(steering_vector, signal_wave), steering_vector, signal_wave

    def add_awgn(self, signal, power, snr_db):
        # print(snr_db)
        noise_power = power / (10 ** (snr_db / 10))
        noise2add = np.sqrt(noise_power) * (
                np.random.normal(0, np.sqrt(2) / 2, signal.shape)
                + 1j * np.random.normal(0, np.sqrt(2) / 2, signal.shape))
        return signal + noise2add


if __name__ == '__main__':
    DOAs = np.array([[-30, 40], [60, -10], [-10, 15]])
    DG = DataGenerator(DOAs, 0, repeat=30)
    data, label = DG.get_raw_label()
