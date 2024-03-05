# AMI-LISTA

---

## Data

All data generator code are uploaded, you can easily set any configurations for dataset.

Or if you want the exactlly same dataset, please refer to [here](https://1drv.ms/f/s!ArYnYw5pONtDgoMnRr3rgJ42_8tthg?e=oMfhnX) (If link failded, please contact me with wangxinlin525@gmail.com). 

## Pre-trained checkpoint

Saved as *.pth files for proposed AMI-LISTA, its ablation nets, and compared nets.

1. AMI-LF10.pth: Proposed;
2. LISTA-LF10: LISTA-LF. The LISTA network with layer-fused strategy into the loss function;
3. AMI-10.pth: LISTA-AM. The array manifold integrated LISTA network;
4. LISTA-10: The basic LISTA framework without training layer-by-layer.

## Models and Neworks

In package DoaMethods:
1. ModelMethods: model-based methods(ISTA, MUSIC, SBL and MVDR);
2. DataMethods: data-based methods(DCNN by [Wuliuli](https://ieeexplore.ieee.org/abstract/document/8854868/));
3. UnfoldingMethods: LISTA by [Wuliuli](https://ieeexplore.ieee.org/abstract/document/9886344), and proposed AMI-LISTA.

## Train

Run trainFrame.py

Notice: Some configurations may need to be applied in configs.py

## Test

Three python files are created for test
1. testFrame.py: for basic test, including predicted power spectrum;
2. testStatic.py: for static performance, including varying SNR, snapshots and separation. (Change the ```mode``` in configs.py)
3. testLayers.py: for ablation and interpretable experiments.


