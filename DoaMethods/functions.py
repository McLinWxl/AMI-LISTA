import heapq
import numpy as np
import scipy.signal
import torch
import DoaMethods
import h5py


def Spect2DoA(Spectrum, num_sources=2, height_ignore=0, start_bias=60):
    """
    :param Spectrum: (num_samples, num_meshes, 1)
    :param num_sources:
    :param height_ignore:
    :param start_bias:
    :return: (num_samples, num_sources)
    """
    num_samples, num_meshes, _ = Spectrum.shape
    angles = np.zeros((num_samples, num_sources))
    for num in range(num_samples):
        li_0 = Spectrum[num, :].reshape(-1)
        li_0[li_0 < 0] = 0
        li = np.sqrt(li_0)
        angle = np.zeros(num_sources) - 5
        peaks_idx = np.zeros(num_sources)
        grids_mesh = np.arange(num_meshes) - start_bias
        peaks, _ = scipy.signal.find_peaks(li, height=height_ignore)
        max_spectrum = heapq.nlargest(num_sources, li[peaks])
        for i in range(len(max_spectrum)):
            peaks_idx[i] = np.where(li == max_spectrum[i])[0][0]
            angle[i] = (
                li[int(peaks_idx[i] + 1)] / (li[int(peaks_idx[i] + 1)]
                                             + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i] + 1)]
                + li[int(peaks_idx[i])] / (li[int(peaks_idx[i] + 1)]
                                           + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i])]
                if li[int(peaks_idx[i] - 1)] < li[int(peaks_idx[i] + 1)]
                else li[int(peaks_idx[i] - 1)] / (li[int(peaks_idx[i] - 1)]
                                                  + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i] - 1)]
                     + li[int(peaks_idx[i])] / (li[int(peaks_idx[i] - 1)]
                                                + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i])]
            )
        angles[num] = angle.reshape(-1)
    return np.sort(angles, axis=1)[::-1]


def Spect2DoA_no_insert(Spectrum, num_sources=2, height_ignore=0, start_bias=60):
    """
    :param Spectrum: (num_samples, num_meshes, 1)
    :param num_sources:
    :param height_ignore:
    :param start_bias:
    :return: (num_samples, num_sources)
    """
    num_samples, num_meshes, _ = Spectrum.shape
    angles = np.zeros((num_samples, num_sources))
    grids_mesh = np.arange(num_meshes) - start_bias
    for num in range(num_samples):
        li_0 = Spectrum[num, :].reshape(-1)
        li_0[li_0 < 0] = 0
        li = np.sqrt(li_0)
        angle = np.zeros(num_sources) - 5
        peaks, _ = scipy.signal.find_peaks(li, height=height_ignore)
        max_spectrum = heapq.nlargest(num_sources, li[peaks])
        for i in range(len(max_spectrum)):
            angle[i] = grids_mesh[np.where(li == max_spectrum[i])[0][0]]
        angles[num] = angle.reshape(-1)
    return np.sort(angles, axis=1)[::-1]


def DoA2Spect(DoA, num_meshes=121, num_sources=2, start_bias=60):
    """
    :param DoA: (num_samples, num_sources)
    :param num_meshes:
    :param num_sources:
    :param start_bias:
    :return: (num_samples, num_meshes, 1)
    """
    num_samples, _ = DoA.shape
    spectrum = np.zeros((num_samples, num_meshes, 1))
    for num in range(num_samples):
        for i in range(num_sources):
            spectrum[num, int(DoA[num, i] + start_bias)] = 1
    return spectrum

# Spect2DoA = Spect2DoA(np.random.rand(10, 121, 1))

def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result

    return func_wrapper


def ReadRaw(path):
    """
    Read .h5 file
    :param path: path of .h5 file
    :return: raw_data
    """
    with h5py.File(path, 'r') as f:
        raw_data = f["RawData"][()]
        assert len(raw_data.shape) == 3
        label = f["LabelPower"][()].reshape(raw_data.shape[0], 1, -1)
    return raw_data, label


class ReadModel:
    def __init__(self, name, dictionary, num_layers, device='cpu', **kwargs):
        is_train = kwargs.get('is_train', False)
        if name == 'LISTA':
            model = (DoaMethods.UnfoldingMethods.LISTA(dictionary=dictionary, num_layers=num_layers)
                     .to(device))
        elif name == 'AMI':
            model = (DoaMethods.UnfoldingMethods.AMI_LISTA(dictionary=dictionary, num_layers=num_layers)
                     .to(device))
        elif name == 'ALISTA':
            model = (DoaMethods.UnfoldingMethods.ALISTA(dictionary=dictionary, num_layers=num_layers, is_train=is_train)
                     .to(device))
        elif name == 'ALISTA-SS':
            model = (DoaMethods.UnfoldingMethods.ALISTA_SS(dictionary=dictionary, num_layers=num_layers, is_train=is_train)
                     .to(device))
        elif name == 'CPSS':
            model = (DoaMethods.UnfoldingMethods.CPSS_LISTA(dictionary=dictionary, num_layers=num_layers)
                     .to(device))
        elif name == 'DCNN':
            model = (DoaMethods.DataMethods.DCNN().to(device))
        else:
            raise ValueError("No such model")
        self.model = model
        # Print trainable parameters
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name} Total number of trainable parameters : {num_trainable_params}")


        # print(f"{name} Total number of parameters : {sum(p.numel() for p in model.parameters())}")

    def get_model(self):
        return self.model

    def load_model(self, modelPath):
        state_dict = torch.load(modelPath, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict['model'])
        return self.model


def denoise_covariance(covariance_matrix, num_sources=2):
    """
    Minus the noise variance (estimated by the smallest eigenvalue) from the covariance matrix.
    :param covariance_matrix:
    :param num_sources:
    :return: Denoised covariance vector
    """
    nums, M, M = covariance_matrix.shape
    covariance_matrix_clean = np.zeros((nums, M, M)) + 1j * np.zeros((nums, M, M))
    for i in range(nums):
        eigvalue = np.linalg.eigvals(covariance_matrix[i])
        smallest_eigvalue = heapq.nsmallest(int(M - num_sources), eigvalue)
        noise_variance = np.mean(smallest_eigvalue)
        noise_matrix = noise_variance * np.eye(M)
        covariance_matrix_clean[i] = covariance_matrix[i] - noise_matrix
    return covariance_matrix_clean


def soft_threshold(x, theta):
    return x.sgn() * torch.nn.functional.relu(x.abs() - theta)


def support_selection(x, theta, p):
    """
    :param x:
    :param theta: Threshold
    :param p: Possibility of the support
    :return:
    """
    x_abs = x.abs()
    threshold = torch.quantile(x_abs, 1 - p, dim=1, keepdims=True)
    bypass = torch.logical_and(torch.ge(x_abs, threshold), torch.ge(x_abs, theta))
    output = torch.where(bypass, x_abs, soft_threshold(x_abs, theta))
    a = 1
    return output


def min_max_norm(x):
    """
    :param x: [batch, data, 1]
    :return:
    """

    if isinstance(x, np.ndarray):
        batch, data, _ = x.shape
        x += 1e-20
        for i in range(batch):
            x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
    elif isinstance(x, torch.Tensor):
        batch, data, _ = x.shape
        # x = 1e-20 + x
        for i in range(batch):
            x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]))
    return x



def find_peak(spectrum, num_sources=2, height_ignore=0, start_bias=60, is_insert=False):
    numTest, num_mesh, _ = spectrum.shape
    angles = np.zeros((num_sources, numTest))
    for num in range(numTest):
        li = spectrum[num, :].reshape(-1)
        if is_insert:
            angle = Spect2DoA(spectrum[num, :].reshape(1, num_mesh, 1), num_sources=num_sources, height_ignore=height_ignore, start_bias=start_bias)
        else:
            angle = Spect2DoA_no_insert(spectrum[num, :].reshape(1, num_mesh, 1), num_sources=num_sources, height_ignore=height_ignore, start_bias=start_bias)
        angles[:, num] = angle.reshape(-1)
    return np.sort(angles, axis=0)[::-1]
