import DoaMethods
from configs import config_test_static as config
from configs import name, is_checkpoint, ModelMethods, DataMethods, UnfoldingMethods, testSNR_interval
import matplotlib.pyplot as plt


mode = 'Layers'
TestCurve = DoaMethods.TestCurve(dir_test=f'./DataSet/Data/TestData_varLayers_{testSNR_interval}.h5')
assert name in UnfoldingMethods, "Wrong name!"
predict, predict_all = TestCurve.test_model(name=name, model_dir=f"{config['model_path']}",

                                  num_layers=config['num_layers'], device=config['device'])
RMSE_all, NMSE_all ,prob_all = [], [], []
for i in range(10):
    peak = TestCurve.find_peak(predict_all[:, :, i].detach().numpy())
    _, RMSE, NMSE, prob = TestCurve.calculate_error(peak)
    RMSE_all.append(RMSE)
    NMSE_all.append(NMSE)
    prob_all.append(prob)


plt.style.use(['science', 'ieee', 'grid'])

x_tricks = [i for i in range(1, 11, 1)]


plt.plot(x_tricks, RMSE_all, label=name)
plt.xlabel('Layers')
plt.ylim(1e-1, 30)
plt.ylabel('RMSE($^{\circ}$)')
plt.yscale('log')
plt.title("RMSE vs number of layers")
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/var{mode}_RMSE.pdf")
plt.show()
plt.close()

plt.plot(x_tricks, NMSE_all, label=name)
plt.xlabel('Layers')
plt.ylabel('NMSE(dB)')
plt.title("NMSE vs number of layers")
plt.ylim(-35, -5)
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/var{mode}_NMSE.pdf")
plt.show()
plt.close()

plt.plot(x_tricks, prob_all, label=name)
plt.xlabel('Layers')
plt.ylabel('Accuracy')
plt.title("Accuracy vs number of layers")

with open(f"{config['result_path']}/var{mode}{config['testSNR_interval']}.csv", 'w') as f:
    f.write("SNR, RMSE, NMSE, prob\n")
    for i in range(len(x_tricks)):
        f.write(f"{x_tricks[i]}, {RMSE_all[i][0]}, {NMSE_all[i][0]}, {prob_all[i][0]}\n")


