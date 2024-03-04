import DoaMethods
from configs import config_test_static as config
from configs import name, is_checkpoint, ModelMethods, DataMethods, UnfoldingMethods, num_sensors, is_insert_superresolution
import matplotlib.pyplot as plt
mode = config['mode']
print(mode)
DoaMethods.configs.configs(name=name, UnfoldingMethods=UnfoldingMethods, DataMethods=DataMethods, ModelMethods=ModelMethods)

TestCurve = DoaMethods.TestCurve(dir_test=config['data_path'])
if name in UnfoldingMethods or name in DataMethods:
    if is_checkpoint:
        predict, _ = TestCurve.test_model(name=name, model_dir=f"{config['model_path']}",
                                          num_layers=config['num_layers'], device=config['device'])
    else:
        predict, _ = TestCurve.test_model(name=name, model_dir=f"{config['model_path']}/best.pth",
                                          num_layers=config['num_layers'], device=config['device'])
    peak = TestCurve.find_peak(predict.detach().numpy(), is_insert_superresolution)

elif name in ModelMethods:
    predict = TestCurve.test_alg(name=name)
    peak = TestCurve.find_peak(predict, is_insert=is_insert_superresolution)
else:
    raise ValueError("Wrong name!")
_, RMSE, NMSE, prob = TestCurve.calculate_error(peak)

plt.style.use(['science', 'ieee', 'grid'])
mode_ranges = {'SNR': range(-12, 13, 1),
               'Separation': range(2, 42, 2),
               'Snapshots': range(10, 410, 20),
               'Sensors': range(2, 21, 2)}
x_tricks = [i for i in mode_ranges[mode]]

mode_labels = {'SNR': 'SNR/dB',
               'Separation': 'Angle Separation/$^{\circ}$',
               'Snapshots': 'Snapshots',
               'Sensors': 'Number of Sensors'}
plt.plot(x_tricks, RMSE, label=name)
plt.xlabel(mode_labels.get(mode, ValueError("Wrong mode!")))
plt.ylabel('RMSE/$^{\circ}$')
plt.yscale('log')
plt.title(f"RMSE vs {mode}")
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/var{mode}_RMSE.pdf")
plt.show()
plt.close()

mode_labels = {'SNR': 'SNR/dB',
               'Separation': 'Angle Separation/$^{\circ}$',
               'Snapshots': 'Snapshots',
               'Sensors': 'Number of Sensors'}
plt.plot(x_tricks, NMSE, label=name)
plt.xlabel(mode_labels.get(mode, ValueError("Wrong mode!")))
plt.ylabel('NMSE/dB')
plt.title(f"NMSE vs {mode}")
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/var{mode}_NMSE.pdf")
plt.show()
plt.close()

if mode == 'SNR' or mode == 'Snapshots':
    with open(f"{config['result_path']}/var{mode}{config['testSNR_interval']}_{num_sensors}.csv", 'w') as f:
        f.write("SNR, RMSE, NMSE, prob\n")
        for i in range(len(x_tricks)):
            f.write(f"{x_tricks[i]}, {RMSE[i]}, {NMSE[i]}, {prob[i]}\n")
elif mode == 'Separation':
    with open(f"{config['result_path']}/var{mode}_{num_sensors}.csv", 'w') as f:
        f.write("Separation, RMSE, NMSE, prob\n")
        for i in range(len(x_tricks)):
            f.write(f"{x_tricks[i]}, {RMSE[i]}, {NMSE[i]}, {prob[i]}\n")
elif mode == 'Sensors':
    with open(f"{config['result_path']}/var{mode}.csv", 'w') as f:
        f.write("Separation, RMSE, NMSE, prob\n")
        for i in range(len(x_tricks)):
            f.write(f"{x_tricks[i]}, {RMSE[i]}, {NMSE[i]}, {prob[i]}\n")
