import numpy as np
import matplotlib.pyplot as plt
from configs import testSNR_interval, config_test_static

mode = "Layers"
names = ["AMI-LF10", "AMI-10", "LISTA-LF10", "LISTA-10"]
for name in names:
    varSNR_dir = f"../Result/{name}/varLayers{testSNR_interval}.csv"
    varSNR = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 0]
    if name == "AMI-LF10":
        RMSE_Ami = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_Ami = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_Ami = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    elif name == "AMI-10":
        RMSE_LISTA_AM = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_LISTA_AM = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_LISTA_AM = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    elif name == "LISTA-LF10":
        RMSE_LISTA_LF = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_LISTA_LF = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_LISTA_LF = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    elif name == "LISTA-10":
        RMSE_LISTA = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_LISTA = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_LISTA = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    else:
        raise ValueError("Wrong name!")

plt.style.use(['science', 'ieee', 'grid'])
plt.plot(varSNR, RMSE_Ami, label='AMI-LISTA')
plt.plot(varSNR, RMSE_LISTA_AM, label='LISTA-AM')
# plt.plot(varSNR, RMSE_Cpss, label='CPSS-LISTA')
plt.plot(varSNR, RMSE_LISTA_LF, label='LISTA-LF')
plt.plot(varSNR, RMSE_LISTA, label='LISTA')

plt.xlabel('Layers')

plt.ylabel('RMSE($^{\circ}$)')
# plt.ylim(1e-1, 30)
plt.yscale('log')
plt.title(f"RMSE vs {mode}")
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"../Figure/Static/var{mode}{testSNR_interval}_RMSE.pdf")
plt.show()
plt.close()

plt.plot(varSNR, NMSE_Ami, label='AMI-LISTA')
plt.plot(varSNR, NMSE_LISTA_AM, label='LISTA-AM')
plt.plot(varSNR, NMSE_LISTA_LF, label='LISTA-LF')
plt.plot(varSNR, NMSE_LISTA, label='LISTA')
plt.xlabel('Layers')
plt.ylabel('NMSE(dB)')
plt.title(f"RMSE vs {mode}")
# plt.ylim(-35, -5)
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"../Figure/Static/var{mode}{testSNR_interval}_NMSE.pdf")
plt.show()
plt.close()

plt.plot(varSNR, prob_Ami, label='AMI-LISTA')
plt.plot(varSNR, prob_LISTA_AM, label='LISTA-AM')
# plt.plot(varSNR, prob_Cpss, label='CPSS-LISTA')
plt.plot(varSNR, prob_LISTA_LF, label='LISTA-LF')
plt.plot(varSNR, prob_LISTA, label='LISTA')
plt.xlabel('Layers')
plt.ylabel('Accuracy')
plt.title("Accuracy of Estimation")
plt.legend(loc='lower right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"../Figure/Static/var{mode}{testSNR_interval}_accuracy.pdf")
plt.show()



