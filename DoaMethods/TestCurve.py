import itertools

import h5py
import numpy as np
import torch
from rich.progress import track

import DoaMethods
from DoaMethods.MakeDataset import MakeDataset
from DoaMethods.functions import ReadModel, timer, find_peak, Spect2DoA, Spect2DoA_no_insert
from configs import is_insert_superresolution, name, UnfoldingMethods, DataMethods, ModelMethods

DoaMethods.configs.configs(name=name, UnfoldingMethods=UnfoldingMethods, DataMethods=DataMethods,
                           ModelMethods=ModelMethods)


class TestCurve:

    def __init__(self, dir_test, resolution=1, num_sources=2, num_meshes=121):
        test_mat = h5py.File(dir_test, 'r')
        self.raw_data = test_mat["RawData"][()]
        assert len(self.raw_data.shape) == 4
        self.num_sources = num_sources
        self.num_meshes = num_meshes
        self.num_lists, self.samples, self.num_sensors, self.num_snapshots = self.raw_data.shape
        self.label = test_mat["LabelPower"][()].reshape(self.num_lists, self.samples, 1, -1)
        self.covariance_matrix_clean = np.zeros(
            (self.num_lists, self.samples, self.num_sensors, self.num_sensors)) + 1j * np.zeros(
            (self.num_lists, self.samples, self.num_sensors, self.num_sensors))
        self.covariance_vector = np.zeros((self.num_lists, self.samples, self.num_sensors ** 2, 1)) + 1j * np.zeros(
            (self.num_lists, self.samples, self.num_sensors ** 2, 1))
        self.pseudo_spectrum = np.zeros((self.num_lists, self.samples, 2, num_meshes))
        for i in range(self.num_lists):
            dataset = MakeDataset(self.raw_data[i], self.label[i])
            self.covariance_matrix_clean[i] = dataset.cal_covariance_matrix_clean()
            self.covariance_vector[i] = dataset.cal_covariance_vector()
            self.pseudo_spectrum[i] = dataset.cal_psuedo_spectrum()
            if i == 0:
                self.dictionary = dataset.dictionary

        self.angles = np.linspace(-0.5 * (self.num_meshes - 1) * resolution, 0.5 * (self.num_meshes - 1) * resolution,
                                  self.num_meshes)
        # Find the non-zero index of the label
        self.DOA_train = np.zeros((self.num_lists, self.samples, self.num_sources))
        for i in range(self.num_lists):
            for j in range(self.samples):
                true_angle = np.where(self.label[i, j] != 0)[1]
                if len(true_angle) == num_sources:
                    self.DOA_train[i, j] = np.where(self.label[i, j] != 0)[1] - (self.num_meshes - 1) / 2
                elif len(true_angle) == num_sources * 2:
                    self.DOA_train[i, j] = Spect2DoA(self.label[i, j].reshape(1, self.num_meshes, 1),
                                                     num_sources=num_sources,
                                                     start_bias=int((num_meshes - 1) / 2)).reshape(-1)
                    # a = 1
                else:
                    raise ValueError("Wrong label!")

    @timer
    def test_model(self, name, model_dir, **kwargs):
        device = kwargs.get('device', 'cpu')
        self.num_layers = kwargs.get('num_layers', 4)
        dictionary = torch.from_numpy(self.dictionary)
        model = ReadModel(name=name, dictionary=dictionary, num_layers=self.num_layers, device=device).load_model(
            model_dir)
        model.eval()
        prediction = torch.zeros((self.num_lists, self.samples, self.num_meshes, 1))
        prediction_layers = torch.zeros((self.num_lists, self.samples, self.num_layers, self.num_meshes, 1))
        with torch.no_grad():
            if name in UnfoldingMethods:
                for list_idx in track(range(self.num_lists), description="Ufolding"):
                    for idx in range(self.samples):
                        cor_array_item = torch.unsqueeze(torch.from_numpy(self.covariance_vector[list_idx, idx]), dim=0)
                        prediction[list_idx, idx], prediction_layers[list_idx, idx] = model(cor_array_item)
                        # plt.plot(prediction[list_idx, idx].reshape(-1))
                        # plt.show()
            elif name == 'DCNN':
                for list_idx in track(range(self.num_lists), description="DCNN"):
                    for idx in range(self.samples):
                        cor_array_item = torch.unsqueeze(torch.from_numpy(self.pseudo_spectrum[list_idx, idx]), dim=0)
                        prediction[list_idx, idx] = model(cor_array_item)

        return prediction, prediction_layers

    def test_alg(self, name, **kwargs):
        prediction = np.zeros((self.num_lists, self.samples, self.num_meshes, 1))
        algorithm = DoaMethods.ModelMethods.ModelMethods(dictionary=self.dictionary)
        if name == 'ISTA':
            for list_idx in track(range(self.num_lists), description="ISTA"):
                for idx in range(self.samples):
                    prediction[list_idx, idx] = (algorithm.ISTA(self.covariance_vector[list_idx, idx]))
        elif name == 'MUSIC':
            for list_idx in track(range(self.num_lists), description="ISTA"):
                for idx in range(self.samples):
                    prediction[list_idx, idx] = (algorithm.MUSIC(self.covariance_matrix_clean[list_idx, idx]))
        elif name == 'SBL':
            for list_idx in track(range(self.num_lists), description="SBL"):
                for idx in range(self.samples):
                    prediction[list_idx, idx] = (algorithm.SBL(self.raw_data[list_idx, idx]))
        elif name == 'MVDR':
            for list_idx in track(range(self.num_lists), description="MVDR"):
                for idx in range(self.samples):
                    prediction[list_idx, idx] = (algorithm.MVDR(self.covariance_matrix_clean[list_idx, idx]))
        else:
            raise ValueError("Wrong name!")
        return prediction

    def find_peak(self, predict, is_insert=False):
        num_lists, num_id, _, _ = predict.shape
        peak = np.zeros((num_lists, num_id, 2))
        for list_idx, idx in itertools.product(range(num_lists), range(num_id)):
            aps = predict[list_idx, idx]
            peak[list_idx, idx] = find_peak(predict[list_idx, idx].reshape(1, self.num_meshes, 1),
                                            num_sources=2, is_insert=is_insert).reshape(-1)
        return peak

    def calculate_error(self, peak):
        """
        :param peak: (num_lists, samples, 2)
        :return: error, RMSE, NMSE, prob
        """
        num_list, num_id, _ = peak.shape
        RMSE = np.zeros(num_list)
        NMSE = np.zeros(num_list)
        prob = np.zeros(num_list)
        predict_st = peak
        DOA_train_st = self.DOA_train
        error_ = np.zeros((num_list, num_id, self.num_sources))
        for snr in range(num_list):
            for itt in range(num_id):
                predict_st[snr, itt] = np.sort(peak[snr, itt])
                DOA_train_st[snr, itt] = np.sort(self.DOA_train[snr, itt])
        for snr in range(num_list):
            error = np.abs(np.sort(predict_st[snr]) - DOA_train_st[snr])
            error_[snr] = (predict_st[snr]) - DOA_train_st[snr]
            for idx in range(num_id):
                for i in range(self.num_sources):
                    prob[snr] += np.sum(error[idx, i] <= 4.4)
                    if error[idx, i] > 4.4:
                        error[idx, i] = 10
            RMSE[snr] = np.sqrt(np.mean(error ** 2))
            NMSE[snr] = 10 * np.log10(np.mean(error ** 2) / np.mean(DOA_train_st[snr] ** 2))
            prob[snr] = prob[snr] / num_id

        return error_, RMSE, NMSE, prob / self.num_sources
