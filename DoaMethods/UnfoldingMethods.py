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


class CPSS_LISTA(torch.nn.Module):
    def __init__(self, dictionary: torch.Tensor, **kwargs):
        """
        :param dictionary: **Required**
        :param is_SS: True
        :param is_PP: True
        :param num_layers: 10
        :param p_selection: 1.2
        :param p_max: 9
        :param device: 'cpu
        """
        super().__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        self.num_meshes = dictionary.shape[1]
        self.is_SS = kwargs.get('SS', True)
        self.is_CP = kwargs.get('CP', True)
        self.num_layers = kwargs.get('num_layers', 10)
        self.p_selection = kwargs.get('p_selection', 3.31)
        self.p_max = kwargs.get('p_max', 3.31)
        self.device = kwargs.get('device', torch.device('cpu'))
        self.dictionary = dictionary.to(torch.complex64)

        self.weight = torch.nn.Parameter(
            torch.eye(self.num_sensors ** 2).repeat(self.num_layers, 1, 1) + 1j * torch.zeros(
                [self.num_layers, self.num_sensors ** 2, self.num_sensors ** 2]), requires_grad=True)
        # TODO: Better initialization
        self.theta = torch.nn.Parameter(0.001 * torch.ones(self.num_layers), requires_grad=True)
        self.gamma = torch.nn.Parameter(0.001 * torch.ones(self.num_layers), requires_grad=True)

    def forward(self, covariance_array):
        covariance_array = covariance_array.to(torch.complex64)
        covariance_array = covariance_array / torch.linalg.matrix_norm(covariance_array, ord=np.inf, keepdim=True)
        eta = torch.real(torch.matmul(self.dictionary.transpose(1, 0).conj(), covariance_array))
        # eta /= torch.norm(eta, dim=1, keepdim=True)
        eta_layers = torch.zeros(covariance_array.shape[0], self.num_layers, self.num_meshes, 1).to(self.device)
        for i in range(self.num_layers):
            W = self.gamma[i] * torch.matmul(self.weight[i], self.dictionary).transpose(1, 0).conj()
            w_1 = ((torch.eye(self.num_meshes) + 0j * torch.zeros([self.num_meshes, self.num_meshes])) -
                   torch.matmul(W, self.dictionary))
            w_2 = W
            r = torch.matmul(w_1, (eta + 0j)) + torch.matmul(w_2, covariance_array)
            p = torch.min(torch.tensor([self.p_selection * (i + 1), self.p_max]))
            p = (p / 100).to(self.device)
            if self.is_SS:
                eta = support_selection(r, self.theta[i], p)
            else:
                eta = soft_threshold(r, self.theta[i])
            eta = torch.abs(eta)
            eta = torch.abs(eta) / (
                    torch.sqrt(torch.tensor(2.)) * (torch.norm(torch.abs(eta), dim=1, keepdim=True) + 1e-20))
            eta_layers[:, i, :, :] = eta
        return eta, eta_layers


class ALISTA(torch.nn.Module):
    def __init__(self, dictionary, **kwargs):
        """
        :param dictionary **required**
        :param num_layers: 10
        :param device: 'cpu'
        """
        super(ALISTA, self).__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        M2 = self.num_sensors ** 2
        self.num_meshes = dictionary.shape[1]
        self.M2 = M2
        self.covariance_norm = kwargs.get('covariance_norm', 1)
        self.num_layers = kwargs.get('num_layers', 10)
        self.device = kwargs.get('device', 'cpu')
        self.is_SS = kwargs.get('SS', True)
        if self.is_SS:
            self.p_para = 0.031 * torch.ones(self.num_layers)
        # calculate the F-norm of dictionary matrix (64, 121)
        self.dictionary = dictionary

        # calculate the step size
        self.stepsize_init = 1 / (5 * torch.linalg.eigvals(torch.matmul(dictionary.conj().T, dictionary)).real.max())
        print(f"Step Size Init: {self.stepsize_init}")

        # Trainable Parameters
        self.gamma = torch.nn.Parameter(self.stepsize_init * torch.ones(self.num_layers), requires_grad=True)
        self.theta = torch.nn.Parameter(0.1 * self.stepsize_init * torch.ones(self.num_layers), requires_grad=True)
        self.relu = torch.nn.ReLU()

        ite, epsilon = 0, 1
        W, W_before = dictionary, torch.zeros_like(dictionary)
        while epsilon > 0.00001 and ite < 100000:
            W = self.PGD(W, self.stepsize_init)
            epsilon = torch.norm(W_before - W)
            W_before = W
            ite += 1
            print(ite, epsilon)

        self.W = W

    def PGD(self, W, gamma):
        """
        Projection Gradient Descent
        :param W: Weight matrix
        :param gamma: Step size
        """
        D = self.dictionary.to(torch.complex64)
        I = torch.eye(self.num_meshes) + 1j * torch.zeros([self.num_meshes, self.num_meshes])
        # Apply Projection
        part1 = torch.matmul(D.conj().T, W) - I
        part2 = torch.matmul(D, part1)
        W = W - gamma * part2
        return W

    def forward(self, covariance_vector: torch.Tensor):
        dictionary = self.dictionary.to(torch.complex64)
        covariance_vector = covariance_vector.reshape(-1, self.M2, 1).to(self.device).to(torch.complex64)
        self.covariance_norm = torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        covariance_vector = covariance_vector / self.covariance_norm
        batch_size = covariance_vector.shape[0]
        x0 = torch.matmul(dictionary.conj().T, covariance_vector)
        x0 = x0 / self.covariance_norm
        x_real = x0.real.float()
        x_layers_virtual = torch.zeros(batch_size, self.num_layers, self.num_meshes, 1).to(self.device)
        for layer in range(self.num_layers):
            p1 = torch.matmul(dictionary, x_real + 1j * torch.zeros_like(x_real)) - covariance_vector
            p2 = self.gamma[layer] * torch.matmul(self.W.conj().T, p1)
            p3 = x_real - p2
            x_abs = torch.abs(p3)
            if self.is_SS:
                x_real = support_selection(x_abs, self.theta[layer], self.p_para[layer])
                a = 1
            else:
                # if layer < self.num_layers - 1:
                #     x_real = self.leakly_relu(x_abs - self.theta[layer])
                # else:
                #     x_real = self.relu(x_abs - self.theta[layer])
                x_real = self.relu(x_abs - torch.abs(self.theta[layer]))
            x_real = x_real / (torch.norm(x_real, dim=1, keepdim=True) + 1e-20)
            x_real = x_real / self.covariance_norm
            x_layers_virtual[:, layer] = x_real
        return x_real, x_layers_virtual


class ALISTA_DU(torch.nn.Module):
    def __init__(self, dictionary, **kwargs):
        """
        :param dictionary **required**
        :param num_layers: 10
        :param device: 'cpu'
        """
        super(ALISTA, self).__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        M2 = self.num_sensors ** 2
        self.num_meshes = dictionary.shape[1]
        self.M2 = M2
        self.covariance_norm = kwargs.get('covariance_norm', 1)
        self.num_layers = kwargs.get('num_layers', 10)
        self.device = kwargs.get('device', 'cpu')
        self.is_SS = kwargs.get('SS', True)
        self.PGD_layers = kwargs.get('PGD_layers', 10)
        if self.is_SS:
            self.p_para = torch.nn.Parameter(5 * torch.ones(self.num_layers), requires_grad=True)
        self.is_train = kwargs.get('is_train', False)
        self.dictionary = dictionary
        # calculate the step size
        stepsize_init = 1 / (5 * torch.linalg.eigvals(torch.matmul(dictionary.conj().T, dictionary)).real.max())
        print(f"Step Size Init: {stepsize_init}")
        # Trainable Parameters
        self.stepsize_PGD = torch.nn.Parameter(stepsize_init, requires_grad=True)
        self.weight_PGD = torch.nn.Parameter(
            torch.eye(self.num_meshes) + 1j * torch.zeros([self.num_meshes, self.num_meshes]), requires_grad=True)
        # self.penalty = torch.nn.Parameter(torch.ones(self.num_meshes, self.num_meshes) + torch.eye(self.num_meshes), requires_grad=False)

        self.gamma = torch.nn.Parameter(stepsize_init * torch.ones(self.num_layers), requires_grad=True)
        self.theta = torch.nn.Parameter(0.1 * stepsize_init * torch.ones(self.num_layers), requires_grad=True)
        self.relu = torch.nn.ReLU()

    def PGD(self, weight_PGD, gamma, PGD_layers=10):
        """
        Projection Gradient Descent
        :param penalty: penalty weight
        :param gamma: Step size
        :param PGD_layers: Number of layers
        """
        D = self.dictionary.to(torch.complex64)
        W = D
        D_weighted = torch.matmul(D, weight_PGD)
        I = torch.eye(self.num_meshes) + 1j * torch.zeros([self.num_meshes, self.num_meshes])
        # penalty_ = torch.mul(penalty, penalty)
        for i in range(PGD_layers):
            part1 = torch.matmul(D_weighted.conj().T, W) - I
            part2 = gamma * torch.matmul(D_weighted, part1)
            W = W - part2
        return W

    def forward(self, covariance_vector: torch.Tensor):
        W = self.PGD(self.weight_PGD, self.stepsize_PGD, PGD_layers=self.PGD_layers)
        dictionary = self.dictionary.to(torch.complex64)
        covariance_vector = covariance_vector.reshape(-1, self.M2, 1).to(self.device).to(torch.complex64)
        self.covariance_norm = torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        covariance_vector = covariance_vector / self.covariance_norm
        batch_size = covariance_vector.shape[0]
        x0 = torch.matmul(dictionary.conj().T, covariance_vector)
        x0 = x0 / self.covariance_norm
        x_real = x0.real.float()
        x_layers_virtual = torch.zeros(batch_size, self.num_layers, self.num_meshes, 1).to(self.device)
        for layer in range(self.num_layers):
            p1 = torch.matmul(dictionary, x_real + 1j * torch.zeros_like(x_real)) - covariance_vector
            p2 = self.gamma[layer] * torch.matmul(W.conj().T, p1)
            p3 = x_real - p2
            x_abs = torch.abs(p3)
            if self.is_SS:
                x_real = support_selection(x_abs, self.theta[layer], self.p_para[layer] / 100)
            else:
                # if layer < self.num_layers - 1:
                #     x_real = self.leakly_relu(x_abs - self.theta[layer])
                # else:
                #     x_real = self.relu(x_abs - self.theta[layer])
                x_real = self.relu(x_abs - torch.abs(self.theta[layer]))
            x_real = x_real / (torch.norm(x_real, dim=1, keepdim=True) + 1e-20)
            x_real = x_real / self.covariance_norm
            x_layers_virtual[:, layer] = x_real
        return x_real, x_layers_virtual


class ALISTA_SS(torch.nn.Module):
    def __init__(self, dictionary, **kwargs):
        """
        :param dictionary **required**
        :param num_layers: 10
        :param device: 'cpu'
        """
        super(ALISTA_SS, self).__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        M2 = self.num_sensors ** 2
        self.num_meshes = dictionary.shape[1]
        self.M2 = M2
        self.covariance_norm = kwargs.get('covariance_norm', 1)
        self.num_layers = kwargs.get('num_layers', 10)
        self.device = kwargs.get('device', 'cpu')
        # calculate the F-norm of dictionary matrix (64, 121)
        self.dictionary = dictionary

        # calculate the step size
        self.stepsize_init = 1 / (5 * torch.linalg.eigvals(torch.matmul(dictionary.conj().T, dictionary)).real.max())
        print(f"Step Size Init: {self.stepsize_init}")

        # Trainable Parameters
        self.gamma = torch.nn.Parameter(self.stepsize_init * torch.ones(self.num_layers), requires_grad=True)
        self.theta = torch.nn.Parameter(0.1 * self.stepsize_init * torch.ones(self.num_layers), requires_grad=True)
        self.relu = torch.nn.ReLU()

        ite, epsilon = 0, 1
        W, W_before = dictionary, torch.zeros_like(dictionary)
        while epsilon > 0.00001 and ite < 10000:
            W = self.PGD(W, self.stepsize_init)
            epsilon = torch.norm(W_before - W)
            W_before = W
            ite += 1
            print(ite, epsilon)

        self.W = W

        self.linear1 = torch.nn.Linear(self.num_meshes, self.num_meshes)
        self.linear2 = torch.nn.Linear(self.num_meshes, self.num_meshes)
        self.sigmoid = torch.nn.Sigmoid()

    def PGD(self, W, gamma):
        """
        Projection Gradient Descent
        :param W: Weight matrix
        :param gamma: Step size
        """
        D = self.dictionary.to(torch.complex64)
        I = torch.eye(self.num_meshes) + 1j * torch.zeros([self.num_meshes, self.num_meshes])
        # Apply Projection
        part1 = torch.matmul(D.conj().T, W) - I
        part2 = torch.matmul(D, part1)
        W = W - gamma * part2
        return W

    def cal_threshold_weight(self, x):
        x_input = x.reshape(-1, self.num_meshes)
        x_forward1 = self.linear1(x_input)
        x_forward2 = self.sigmoid(x_forward1)
        x_forward3 = self.linear2(x_forward2)
        x_forward4 = self.sigmoid(x_forward3)
        return x_forward4.reshape(-1, self.num_meshes, 1)

    def forward(self, covariance_vector: torch.Tensor):
        dictionary = self.dictionary.to(torch.complex64)
        covariance_vector = covariance_vector.reshape(-1, self.M2, 1).to(self.device).to(torch.complex64)
        self.covariance_norm = torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        covariance_vector = covariance_vector / self.covariance_norm
        batch_size = covariance_vector.shape[0]
        x0 = torch.matmul(dictionary.conj().T, covariance_vector)
        x0 = x0 / self.covariance_norm
        x_real, x_init = x0.real.float(), x0.real.float()
        x_layers_virtual = torch.zeros(batch_size, self.num_layers, self.num_meshes, 1).to(self.device)
        for layer in range(self.num_layers):
            p1 = torch.matmul(dictionary, x_real + 1j * torch.zeros_like(x_real)) - covariance_vector
            p2 = self.gamma[layer] * torch.matmul(self.W.conj().T, p1)
            p3 = x_real - p2
            x_abs = torch.abs(p3)
            x_threshold_weight = self.cal_threshold_weight(x_abs)
            x_real = self.relu((x_abs - torch.mul(x_threshold_weight, x_real) - self.theta[layer]))
            x_real = x_real / (torch.norm(x_real, dim=1, keepdim=True) + 1e-20)
            x_real = x_real / self.covariance_norm
            x_layers_virtual[:, layer] = x_real
            # To visualize
            # plt.matshow(x_init[-1].detach().numpy(), cmap=plt.cm.Reds)
            # plt.show()
            # plt.matshow(x_abs[-1].detach().numpy(), cmap=plt.cm.Reds)
            # plt.show()
            # plt.matshow(x_threshold_weight[-1].detach().numpy(), cmap=plt.cm.Reds)
            # plt.show()
            # plt.matshow(x_real[-1].detach().numpy(), cmap=plt.cm.Reds)
            # plt.show()
        return x_real, x_layers_virtual
