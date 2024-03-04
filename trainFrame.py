import DoaMethods
import torch.utils.data
from configs import name, config, UnfoldingMethods, DataMethods, ModelMethods
from DoaMethods.functions import ReadRaw
import matplotlib.pyplot as plt

DoaMethods.configs.configs(name=name, UnfoldingMethods=UnfoldingMethods, DataMethods=DataMethods, ModelMethods=ModelMethods)

raw_data, label = ReadRaw(config['data_path'])
dataset = DoaMethods.MakeDataset(raw_data, label)
print(len(dataset))
train_set, valid_set = torch.utils.data.random_split(dataset,
                                                     [int(len(dataset) * 0.8),
                                                      len(dataset) - int(len(dataset) * 0.8)])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False)

dictionary = torch.from_numpy(dataset.dictionary)

model = DoaMethods.functions.ReadModel(name=name, dictionary=dictionary, num_layers=config['num_layers'], device=config['device'], is_train=True).get_model()
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# TODO: Better scheduler?
scheduler_cosine_warmup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

loss_best = 100
for epoch in range(config['epoch']+1):
    if epoch < 15 or epoch % 10 == 0:
        torch.save({'model': model.state_dict()}, f"{config['model_path']}/model_{epoch}.pth")
    model.train()
    mse_train_last = 0
    iii = 0
    for data, label in train_loader:
        mse_loss = 0
        label = label.to(config['device'])
        data = data.to(config['device'])
        if name in UnfoldingMethods:
            output, layers_output = model(data)
            if config['LF']:
                for i in range(config['num_layers']):
                    mse_loss = mse_loss + (loss(layers_output[:, i].to(torch.float32), label.to(torch.float32))) * (
                            torch.log(torch.tensor(i + 2)))
            else:
                mse_loss = loss(output.to(torch.float32), label.to(torch.float32))
        elif name in DataMethods:
            output = model(data)
            mse_loss = loss(output.to(torch.float32), label.to(torch.float32))
        mse_loss.backward()
        # print(f"{model.p_para.is_leaf}:{model.p_para.requires_grad}:{model.p_para.grad}:{model.p_para.grad_fn}")
        optimizer.step()
        # make the gamma and theta positive
        # if name == 'ALISTA':
        #     model.gamma.data = torch.abs(model.gamma.data)
        #     model.theta.data = torch.abs(model.theta.data)

        optimizer.zero_grad()
        mse_train_last += mse_loss.item()
        # if iii == 0:
        #     print(f"Epoch: {epoch}, Iteration: {iii}, Loss: {mse_loss.item()}")
        iii += 1
    mse_train_last /= len(train_loader)

    model.eval()
    mse_val_last = 0
    with torch.no_grad():
        for covariance_array, label in val_loader:
            label = label.to(config['device'])
            # if epoch <= config['warmup_epoch']:
            #     label /= torch.sqrt(torch.tensor(3))
            covariance_array = covariance_array.to(config['device'])
            loss_value = 0
            if name in UnfoldingMethods:
                output, layers_output_val = model(covariance_array)
                if config['LF']:
                    for i in range(config['num_layers']):
                        loss_value = loss_value + (
                            loss(layers_output_val[:, i].to(torch.float32), label.to(torch.float32))) * (
                                             torch.log(torch.tensor(i + 2)))
                else:
                    loss_value = loss(output.to(torch.float32), label.to(torch.float32))
                mse_val_last += loss_value.item()
            elif name in DataMethods:
                output = model(covariance_array)
                mse_val_last += loss(output.to(torch.float32), label.to(torch.float32)).item()
    mse_val_last /= len(val_loader)
    print(f"Epoch: {epoch}, Train Loss: {mse_train_last}, Valid Loss: {mse_val_last}")

    if config['scheduler']:
        lr = []
        if epoch <= config['warmup_epoch']:
            scheduler_cosine_warmup.step(epoch)
        else:
            scheduler.step(mse_val_last)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        lr.append(optimizer.param_groups[0]['lr'])

    with open(f"{config['result_path']}/lr.csv", 'a+') as f:
        f.write(f"{epoch},{optimizer.param_groups[0]['lr']}\n")

    if mse_val_last < loss_best:
        loss_best = mse_val_last
        torch.save({'model': model.state_dict()}, f"{config['model_path']}/best.pth")
        print("Best model saved")

    # Save loss to .csv
    with open(f"{config['result_path']}/loss.csv", 'a+') as f:
        f.write(f"{epoch},{mse_train_last},{mse_val_last}\n")




