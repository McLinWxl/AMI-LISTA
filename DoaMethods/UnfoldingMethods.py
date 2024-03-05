import matplotlib.pyplot as plt
import numpy as np
import torch
from DoaMethods.functions import support_selection, soft_threshold


class LISTA(torch.nn.Module):

    def __init__(self, dictionary, **kwargs):
        """
        :param dictionary: **required**
        :param num_layers: 10
        :param device: 'cpu'
        """
        super(LISTA, self).__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        self.num_meshes = dictionary.shape[1]
        self.num_layers = kwargs.get('num_layers', 10)
        self.device = kwargs.get('device', 'cpu')
        num_sensors_powered = self.num_sensors * self.num_sensors

        self.We = torch.nn.Parameter(torch.randn([self.num_layers, self.num_meshes, num_sensors_powered]) +
                                     1j * torch.randn([self.num_layers, self.num_meshes, num_sensors_powered]),
                                     requires_grad=True)
        self.Wg = torch.nn.Parameter(torch.randn([self.num_layers, self.num_meshes, self.num_meshes]) +
                                     1j * torch.randn([self.num_layers, self.num_meshes, self.num_meshes]),
                                     requires_grad=True)
        self.theta = torch.nn.Parameter(0.01 * torch.ones(self.num_layers), requires_grad=True)

        self.num_sensors_2p = num_sensors_powered
        self.relu = torch.nn.ReLU()
        self.dictionary = dictionary

    def forward(self, covariance_vector, device="cpu"):
        dictionary = self.dictionary.to(torch.complex64)
        covariance_vector = covariance_vector.reshape(-1, self.num_sensors_2p, 1).to(torch.complex64).to(self.device)
        covariance_vector = covariance_vector / torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        batchSize = covariance_vector.shape[0]
        x_eta = torch.matmul(dictionary.conj().T, covariance_vector).real.float()
        # x_eta /= torch.norm(x_eta, dim=1, keepdim=True)
        covariance_vector = covariance_vector.to(device)
        x_layers_virtual = torch.zeros(batchSize, self.num_layers, self.num_meshes, 1)

        for t in range(self.num_layers):
            We = self.We[t]
            Wg = self.Wg[t]
            z = torch.matmul(We, covariance_vector) + torch.matmul(Wg, (x_eta + 1j * torch.zeros_like(x_eta)))
            x_abs = torch.abs(z)
            # apply soft-thresholding on xabs, return xabs
            x_eta = self.relu(x_abs - self.theta[t])
            x_norm = x_eta.norm(dim=1, keepdim=True)
            x_eta = x_eta / (torch.sqrt(torch.tensor(2.)) * (x_norm + 1e-20))
            x_layers_virtual[:, t] = x_eta
        return x_eta, x_layers_virtual


class AMI_LISTA(torch.nn.Module):
    def __init__(self, dictionary, **kwargs):
        """
        :param dictionary **required**
        :param num_layers: 10
        :param device: 'cpu
        :param mode: None ('tied', 'single', or 'both')
        """
        super(AMI_LISTA, self).__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        M2 = self.num_sensors ** 2
        self.num_meshes = dictionary.shape[1]
        self.M2 = M2
        self.num_layers = kwargs.get('num_layers', 10)
        self.device = kwargs.get('device', 'cpu')
        self.mode = kwargs.get('mode', None)

        print(f'mode: {self.mode}')
        if not self.mode:
            self.W1 = torch.nn.Parameter(torch.eye(M2).repeat(self.num_layers, 1, 1)
                                         + 1j * torch.zeros([self.num_layers, M2, M2]), requires_grad=True)
            self.W2 = torch.nn.Parameter(torch.eye(M2).repeat(self.num_layers, 1, 1)
                                         + 1j * torch.zeros([self.num_layers, M2, M2]), requires_grad=True)
        elif self.mode == 'tied':
            self.W1 = torch.nn.Parameter(torch.eye(M2) + 1j * torch.zeros([M2, M2]), requires_grad=True)
            self.W2 = torch.nn.Parameter(torch.eye(M2) + 1j * torch.zeros([M2, M2]), requires_grad=True)
        elif self.mode == 'single':
            self.W = torch.nn.Parameter(torch.eye(M2).repeat(self.num_layers, 1, 1)
                                        + 1j * torch.zeros([self.num_layers, M2, M2]), requires_grad=True)
        elif self.mode == 'both':
            self.W = torch.nn.Parameter(torch.eye(M2) + 1j * torch.zeros([M2, M2]), requires_grad=True)
        self.theta = torch.nn.Parameter(0.001 * torch.ones(self.num_layers), requires_grad=True)
        self.gamma = torch.nn.Parameter(0.001 * torch.ones(self.num_layers), requires_grad=True)
        self.leakly_relu = torch.nn.LeakyReLU()
        self.dictionary = dictionary
        self.relu = torch.nn.ReLU()

    def forward(self, covariance_vector: torch.Tensor):
        dictionary = self.dictionary.to(torch.complex64)
        covariance_vector = covariance_vector.reshape(-1, self.M2, 1).to(self.device).to(torch.complex64)
        covariance_vector = covariance_vector / torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        batch_size = covariance_vector.shape[0]
        x0 = torch.matmul(dictionary.conj().T, covariance_vector).real.float()
        # x0 /= torch.norm(x0, dim=1, keepdim=True)
        x_real = x0
        x_layers_virtual = torch.zeros(batch_size, self.num_layers, self.num_meshes, 1).to(self.device)
        for layer in range(self.num_layers):
            identity_matrix = (torch.eye(self.num_meshes) + 1j * torch.zeros([self.num_meshes, self.num_meshes])).to(
                self.device)
            if not self.mode:
                W1 = self.W1[layer]
                W2 = self.W2[layer]
            elif self.mode == 'tied':
                W1 = self.W1
                W2 = self.W2
            elif self.mode == 'single':
                W1 = self.W[layer]
                W2 = self.W[layer]
            elif self.mode == 'both':
                W1 = self.W
                W2 = self.W
            else:
                raise Exception('mode error')
            W1D = torch.matmul(W1, dictionary)
            W2D = torch.matmul(W2, dictionary)
            Wt = identity_matrix - self.gamma[layer] * torch.matmul(W2D.conj().T, W2D)
            We = self.gamma[layer] * W1D.conj().T
            s = torch.matmul(Wt, x_real + 1j * torch.zeros_like(x_real)) + torch.matmul(We, covariance_vector)
            s_abs = torch.abs(s)
            if layer < self.num_layers - 1:
                x_real = self.leakly_relu(s_abs - self.theta[layer])
            else:
                x_real = self.relu(s_abs - self.theta[layer])
            x_real = x_real / (torch.norm(x_real, dim=1, keepdim=True) + 1e-20)
            # x_real = x_real / torch.mean(torch.norm(covariance_vector, dim=1, keepdim=True))
            x_layers_virtual[:, layer] = x_real
        return x_real, x_layers_virtual
