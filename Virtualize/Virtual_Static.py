import numpy as np
import matplotlib.pyplot as plt
from configs import testSNR_interval, config_test_static, num_sensors
is_ablition = False
mode = config_test_static['mode']
import numpy as np

def load_data(names, mode, testSNR_interval):
    data = {}
    for name in names:
        if mode in ['SNR', 'Snapshots']:
            varSNR_dir = f"../Result_A/{name}/var{mode}{testSNR_interval}_{num_sensors}.csv"
        elif mode == 'Separation':
            varSNR_dir = f"../Result_A/{name}/var{mode}_{num_sensors}.csv"
        elif mode == 'Sensors':
             varSNR_dir = f"../Result_A/{name}/var{mode}.csv"
        varSNR, RMSE, NMSE, prob = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1).T
        data[name] = {'varSNR': varSNR, 'RMSE': RMSE, 'NMSE': NMSE, 'prob': prob}
    return data

def plot_graph(data, plot_config, names, mode, ylabel, save_as):
    for name, config in plot_config.items():
        if name in names:
            plt.plot(data[name]['varSNR'], data[name][ylabel], **config)
    plt.xlabel('SNR(dB)' if mode == 'SNR' else 'Angle Separation()$^{\circ}$)' if mode == 'Separation' else 'Snapshots' if mode == 'Snapshots' else ValueError("Wrong mode!"))
    plt.ylabel(f'{ylabel}($^\circ$)')
    if ylabel == 'RMSE':
        plt.ylim(1e-1, 30)
    elif ylabel == 'NMSE':
        plt.ylim(-45, -5)
    else:
        plt.ylim(0, 1)
    plt.yscale('log') if ylabel == 'RMSE' else None
    plt.title(f"{ylabel} vs {mode}" if ylabel != 'prob' else "Accuracy of Estimation")
    plt.legend(loc='upper right' if ylabel != 'prob' else 'lower right', prop={'size': 5})
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
    plt.savefig(f"../Figure_A/Static/Ab_var{mode}{testSNR_interval}_{save_as}.pdf" if is_ablition else f"../Figure_A/Static/var{mode}{testSNR_interval}_{save_as}.pdf")
    plt.show()
    plt.close()

if is_ablition:
    names = ["AMI-LF10", "LISTA-LF10", "AMI-10", "LISTA-10"]
    labels = ['AMI-LISTA', 'LISTA-LF', 'AMI-10', 'LISTA-10']
    ylabels = ['RMSE', 'NMSE', 'prob']
    plot_config = {
        "AMI-LF10": {"label": 'AMI-LISTA'},
        "LISTA-LF10": {"label": 'LISTA-LF'},
        "LISTA-10": {"label": 'LISTA'},
        "AMI-10": {"label": 'LISTA-AM'},
    }
    data = load_data(names, mode, testSNR_interval)
    plt.style.use(['science', 'ieee', 'grid'])
    plot_graph(data, plot_config, names, mode, 'RMSE', 'RMSE')
    plot_graph(data, plot_config, names, mode, 'NMSE', 'NMSE')
    plot_graph(data, plot_config, names, mode, 'prob', 'accuracy')
else:
    #     names = ["AMI-LF10", "LISTA-10", "MUSIC", "MVDR", "DCNN", "ALISTA-10"]
    names = ["MUSIC", "MVDR", "DCNN", "ALISTA-SS-10"] #
    plot_config = {
        "AMI-LF10": {"label": 'AMI-LISTA'},
        "LISTA-10": {"label": 'LISTA'},
        "CPSS-10": {"label": 'CPSS-LISTA'},
        "MUSIC": {"label": 'MUSIC'},
        "MVDR": {"label": 'MVDR'},
        "DCNN": {"label": 'DCNN'},
        "ALISTA-10": {"label": 'ALISTA'},
        "ALISTA-20": {"label": 'ALISTA-20'},
        "ALISTA-SS-10": {"label": 'ALISTA-SS', 'color': 'lightcoral'},
    }
    data = load_data(names, mode, testSNR_interval)
    plt.style.use(['science', 'ieee', 'grid'])
    plot_graph(data, plot_config, names, mode, 'RMSE', 'RMSE')
    plot_graph(data, plot_config, names, mode, 'NMSE', 'NMSE')
    plot_graph(data, plot_config, names, mode, 'prob', 'accuracy')