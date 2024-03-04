import h5py
import torch.utils.data
from DoaMethods.functions import denoise_covariance, min_max_norm, ReadRaw
import numpy
from configs import name, UnfoldingMethods, DataMethods, ModelMethods

import DoaMethods.configs
DoaMethods.configs.configs(name=name, UnfoldingMethods=UnfoldingMethods, DataMethods=DataMethods, ModelMethods=ModelMethods)


class MakeDataset(torch.utils.data.Dataset):

    def __init__(self, raw_data, label=None, **kwargs):
        """
        If is simulated, manifold and label are required. Else, sensor_interval is required to calculate manifold.
        :param raw_data:
        :param label:
        :param manifold:
        :param kwargs:
        """
        assert len(raw_data.shape) == 3  # (samples, num_sensors, num_snapshots)
        self.samples, self.num_sensors, _ = raw_data.shape
        # calculate the number of snapshots depending on the 3rd dimension of raw_data of non-zero elements
        self.num_snapshots = numpy.count_nonzero(raw_data[0, 0])
        self.raw_data = raw_data
        self.num_sources = kwargs.get('num_sources', 2)
        D_start = kwargs.get('D_start', -60)
        D_stop = kwargs.get('D_stop', 60)
        interval = kwargs.get('interval', 1)
        self.theta = numpy.arange(D_start, D_stop + 1, interval).reshape(-1)
        self.num_meshes = len(self.theta)

        self.sensor_interval = kwargs.get('sensor_interval', 0.5)
        self.wavelength = kwargs.get('wavelength', 1)
        self.manifold = self.cal_manifold(self.num_sensors, self.sensor_interval, self.wavelength)
        self.label = label.reshape(self.samples, self.num_meshes, 1) if label is not None else numpy.zeros((self.samples, self.num_meshes, 1))
        self.dictionary = self.cal_dictionary()
        self.covariance_matrix_clean = self.cal_covariance_matrix_clean()
        self.covariance_matrix_denoised = self.cal_covariance_matrix_denoised()
        self.covariance_vector = self.cal_covariance_vector()
        self.psudo_spectrum = self.cal_psuedo_spectrum()

    def cal_manifold(self, num_sensors, sensor_interval, wavelength):
        return numpy.exp(1j * numpy.pi * 2 * sensor_interval * numpy.arange(num_sensors)[:, numpy.newaxis] * numpy.sin(numpy.deg2rad(self.theta)) / wavelength)

    def cal_dictionary(self):
        dictionary = numpy.zeros((self.num_sensors**2, self.num_meshes), dtype=numpy.complex64)
        for i in range(self.num_sensors):
            s = numpy.exp(-1j * numpy.pi * 2 * self.sensor_interval * i * numpy.sin(numpy.deg2rad(self.theta)) / self.wavelength)
            B = numpy.diag(s)
            phi = numpy.matmul(self.manifold, B)
            dictionary[i*self.num_sensors:(i+1)*self.num_sensors, :] = phi
        return dictionary

    def cal_covariance_matrix_clean(self):
        covariance_matrix = numpy.matmul(self.raw_data, self.raw_data.conj().transpose(0, 2, 1)) / self.num_snapshots
        # Normalize the covariance matrix up to 1
        a = numpy.max(numpy.abs(covariance_matrix), axis=(1, 2), keepdims=True)
        # covariance_matrix_norm = covariance_matrix / numpy.max(numpy.abs(covariance_matrix), axis=(1, 2), keepdims=True)
        return covariance_matrix

    def cal_covariance_matrix_denoised(self):
        covariance_matrix = denoise_covariance(self.cal_covariance_matrix_clean())
        covariance_matrix_norm = covariance_matrix
        return covariance_matrix_norm

    def cal_covariance_vector(self):
        covariance_vector = self.cal_covariance_matrix_denoised().transpose(0, 2, 1).reshape(self.samples, self.num_sensors ** 2, 1)
        return covariance_vector

    def cal_psuedo_spectrum(self):
        psudo_spectrum = numpy.zeros((self.samples, 2, self.num_meshes), dtype=numpy.float32)
        covariance_matrix_clean = self.cal_covariance_matrix_clean()
        covariance_vector = covariance_matrix_clean.transpose(0, 2, 1).reshape(self.samples, self.num_sensors ** 2, 1)
        temp = numpy.matmul(self.dictionary.conj().transpose(), covariance_vector).transpose(0, 2, 1).reshape(self.samples, 1, self.num_meshes)
        # temp = temp / numpy.linalg.norm(temp, axis=1, keepdims=True, ord=2)
        psudo_spectrum[:, 0, :] = numpy.real(temp).reshape(self.samples, self.num_meshes)
        psudo_spectrum[:, 1, :] = numpy.imag(temp).reshape(self.samples, self.num_meshes)
        psudo_spectrum[:, 0, :] = min_max_norm(psudo_spectrum[:, 0, :].reshape(self.samples, 1, self.num_meshes).transpose(0, 2, 1)).transpose(0, 2, 1).reshape(self.samples, self.num_meshes)
        return psudo_spectrum

    def __getitem__(self, index):
        if name in UnfoldingMethods:
            return self.covariance_vector[index], self.label[index]
        elif name in DataMethods:
            return self.psudo_spectrum[index], self.label[index]
        else:
            raise ValueError("Wrong name!")

    def __len__(self):
        return self.samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    raw, label = ReadRaw("../Dataset_old/Data/TestSpectrum.h5")
    dataset = MakeDataset(raw)
    covariance_vector_clean = dataset.cal_covariance_matrix_denoised()
    psedo_spectrum = dataset.cal_psuedo_spectrum()
    plt.plot(psedo_spectrum[0, 0, :])
    plt.plot(psedo_spectrum[0, 1, :])
    plt.plot(label[0, 0, :])
    plt.show()
    a= 1
