import DoaMethods
import torch.utils.data
from DoaMethods.configs import config_test as config
from DoaMethods.configs import name
import matplotlib.pyplot as plt
import numpy

epoch_read = config['epoch']


dataset = DoaMethods.MakeDataset(f"{config['data_path']}")
print(len(dataset))

dictionary = torch.from_numpy(dataset.get_dictionary())

model = DoaMethods.functions.ReadModel(name=name, dictionary=dictionary, num_layers=config['num_layers'], device=config['device']).load_model(f"{config['model_path']}/best.pth")

Weight_A = model.state_dict()['W1'].detach().numpy()
Weight_B = model.state_dict()['W2'].detach().numpy()

# Draw subplots
for i in range(config['num_layers']):
    plt.matshow(numpy.abs(Weight_A[i]))
plt.show()
plt.close()






