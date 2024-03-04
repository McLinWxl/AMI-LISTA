import numpy as np

import DoaMethods
import torch.utils.data
from configs import config_test as config
from configs import name, DataMethods, UnfoldingMethods, ModelMethods, is_checkpoint, num_sensors
import matplotlib.pyplot as plt
import numpy
from DoaMethods.functions import ReadRaw
num_sources = 2
num_meshes = 121
DoaMethods.configs.configs(name=name, UnfoldingMethods=UnfoldingMethods, DataMethods=DataMethods, ModelMethods=ModelMethods)

raw, label = ReadRaw(config['data_path'])
dataset = DoaMethods.MakeDataset(raw, label)
print(len(dataset))

loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

dictionary_numpy = dataset.dictionary
dictionary = torch.from_numpy(dataset.dictionary)

if name in UnfoldingMethods or name in DataMethods:
    if is_checkpoint:
        model = DoaMethods.functions.ReadModel(name=name, dictionary=dictionary, num_layers=config['num_layers'], device=config['device']).load_model(f"{config['model_path']}")
    else:
        model = DoaMethods.functions.ReadModel(name=name, dictionary=dictionary, num_layers=config['num_layers'], device=config['device']).load_model(f"{config['model_path']}/best.pth")

    # W1 = model.W
    # Res = W1 - dictionary
    # Res = Res.real
    # plt.matshow(Res.detach().numpy())
    # plt.show()
    # print(f"Step Size: {model.gamma}")
    # print(f"Threshold: {model.theta}")
    # print(f"Theta/gamma: {model.theta/model.gamma}")
    # print(f"CPSS: {model.p_para}")
    # print(f"Strp Size PGD: {model.stepsize_PGD}")


    model.eval()
    mse_val_last = 0
    with torch.no_grad():
        for data, label in loader:
            label = label.to(config['device'])
            data = data.to(config['device'])
            if name in UnfoldingMethods:
                output, layers_output_val = model(data)
            elif name in DataMethods:
                output = model(data)
            else:
                raise ValueError("Wrong name!")

elif name in ModelMethods:
    algorithm = DoaMethods.ModelMethods.ModelMethods(dictionary=dictionary_numpy)
    predict = np.zeros((len(dataset), dataset.num_meshes, 1))
    if name == 'ISTA':
        covariance_vector = dataset.cal_covariance_vector()
        label = dataset.label
        for i in range(covariance_vector.shape[0]):
            predict[i] = algorithm.ISTA(covariance_vector[i])

    elif name == 'MUSIC':
        covariance_matrix = dataset.cal_covariance_matrix_clean()
        label = dataset.label
        for i in range(covariance_matrix.shape[0]):
            predict[i] = algorithm.MUSIC(covariance_matrix[i])

    elif name == 'SBL':
        raw_data = dataset.raw_data
        label = dataset.label
        for i in range(raw_data.shape[0]):
            predict[i] = algorithm.SBL(raw_data[i]).reshape(-1, 1)

    elif name == 'MVDR':
        covariance_matrix = dataset.cal_covariance_matrix_clean()
        label = dataset.label
        for i in range(covariance_matrix.shape[0]):
            predict[i] = algorithm.MVDR(covariance_matrix[i])

    else:
        raise ValueError("Wrong name!")


idxs = [0, 1, 2, 3, 4, 5, 6]
for idx in idxs:
    if name in UnfoldingMethods:
        plt.rcParams['font.family'] = 'Times New Roman'
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x_idx = numpy.arange(121) - 60
        for i in range(config['num_layers']):
            to_plot = layers_output_val[idx, i].detach().numpy().reshape(-1)
            ax.plot(x_idx, numpy.ones(121) * (config['num_layers'] - i), to_plot)
        ax.set_box_aspect((20, 60, 13))
        ax.view_init(27, -28)
        # no meshed and no grid and no axis
        # ax.set_axis_off()
        # ax.grid(False)
        ax.set_xticks([-60, -30, 0, 30, 60])
        # ax.set_yticks([])
        ax.set_yticks([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        # Set ylables backwords
        ax.set_yticklabels([10, 9, 8, 7, 6, 5, 4, 3, 2, 1][::-1])
        ax.set_zticks([])
        # ax.set_xlabel('Angle $^{\circ}$')
        # set x label position
        # ax.xaxis.set_label_coords(0.5, -0.1)
        # ax.set_ylabel('Layer')
        # ax.set_zlabel('Amp.')
        # Delete the white space around the figure
        # plt.subplots_adjust(left=0, right=0.9, bottom=0, top=1)
        # plt.show()
        plt.savefig(f"{config['figure_path']}/layer_{idx}_{num_sensors}.pdf")
        plt.show()
        plt.close()

        with plt.style.context(['science', 'ieee', 'grid']):
            plt.plot(output[idx].detach().numpy(), label=name)
            # Find the non-zero index of the label[idx, :, 0]
            true_angle = np.where(label[idx, :, :] != 0)[0]
            for i in range(int(len(true_angle)/2)):
                plt.axvline(x=true_angle[2*i]+0.5, color='red', linestyle='--')

            # for i in range(output.shape[1]):
            #     if label[idx, i]:
            #         plt.axvline(x=i, color='red', linestyle='--')
            plt.xlabel('Angle $^{\circ}$')
            plt.ylabel('Amp.')
            plt.legend()
            plt.savefig(f"{config['figure_path']}/output_{idx}_{num_sensors}.pdf")
            plt.show()

    elif name in DataMethods:
        with plt.style.context(['science', 'ieee', 'grid']):
        # plt.rcParams['figure.dpi'] = 1000
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.plot(output[idx].detach().numpy(), label=name)
            true_angle = np.where(label[idx, :, :] != 0)[0]
            for i in range(int(len(true_angle) / 2)):
                plt.axvline(x=true_angle[2 * i] + 0.5, color='red', linestyle='--')

            plt.xlabel('Angle $^{\circ}$')
            plt.ylabel('Amp.')
            plt.legend()
            # plt.show()
            plt.savefig(f"{config['figure_path']}/output_{idx}_{num_sensors}.pdf")
            plt.show()

    elif name in ModelMethods:
        with plt.style.context(['science', 'ieee', 'grid']):
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.plot(predict[idx].reshape(-1), label=name)
            true_angle = np.where(label[idx, :, :] != 0)[0]
            for i in range(int(len(true_angle) / 2)):
                plt.axvline(x=true_angle[2 * i] + 0.5, color='red', linestyle='--')
            plt.xlabel('Angle $^{\circ}$')
            plt.ylabel('Amp.')
            plt.legend()
            # plt.show()
            plt.savefig(f"{config['figure_path']}/output_{idx}_{num_sensors}.pdf")
            plt.show()

