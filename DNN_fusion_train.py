import numpy as np
import datetime as dt
import glob as gl
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.preprocessing import StandardScaler, PowerTransformer
import joblib

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

import matplotlib.pyplot as plt
from matplotlib import gridspec
import datetime as dt
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm

device = 'cpu' #can use 'cuda' option if gpu is available
num_cores = 1 #cores per configuration
num_samples = 20 #the number of configuration combination

scheduler = ASHAScheduler(
                          metric="loss", #metric for the training
                          mode="min", #select configuration with minimum metric as the best configuration
                          grace_period= 10, #scheduler won't terminate the configuration until grace period
                          reduction_factor=2, #only 50% of all trials are kept each time they are reduced
                          max_t = 30 #maximum epochs
                          )

#define the value range of the configuration
config = {
        'l1':tune.choice([i for i in range(9,19)]),
        'l2':tune.choice([i for i in range(5,16)]),
        'l3':tune.choice([i for i in range(3,12)]),
        'lr':tune.loguniform(1e-4, 1e-1),
        'batch_size':tune.choice([128,256,512,1024])
        }

def scale_gaussian(data):
    if len(data.shape) == 1:
        data = data.reshape((-1,1))
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return scaler, data

def scale_aod(data):
    if len(data.shape) == 1:
        data = data.reshape((-1,1))
    scaler = PowerTransformer(method='box-cox', standardize=True)
    scaler.fit(data)
    data = scaler.transform(data)
    return scaler, data

for flag in range(7):
    with open(f'/home/minseok/Fusion/paper/train_dataset_flag={flag:01}.pkl', 'wb') as fr:
        inputs = pk.load(fr)
        output = pk.load(fr)

    # #########################################################################
    # Data pre-processing
    # #########################################################################
    if flag==0:
    # box-cox doesn't work with negative values #
        inputs[:,0] += 0.05
        inputs[:,1] += 0.05
        inputs[:,2] += 0.05
        inputs[:,3] += 0.05

        scale_in_aod, temp = scale_aod(inputs[:,:4]); inputs[:,:4] = temp.reshape((-1,4))
        scale_in_other, temp = scale_gaussian(inputs[:,4:]); inputs[:,4:] = temp.reshape((-1,6))

        scaler_out, output = scale_aod(output)
    if flag in [1,3]:
    # box-cox doesn't work with negative values #
        inputs[:,0] += 0.05
        inputs[:,1] += 0.05
        inputs[:,2] += 0.05

        scale_in_aod, temp = scale_aod(inputs[:,:3]); inputs[:,:3] = temp.reshape((-1,3))
        scale_in_other, temp = scale_gaussian(inputs[:,3:]); inputs[:,3:] = temp.reshape((-1,5))

        scaler_out, output = scale_aod(output)
    elif flag in [2,5]:
    # box-cox doesn't work with negative values #
        inputs[:,0] += 0.05
        inputs[:,1] += 0.05

        scale_in_aod, temp = scale_aod(inputs[:,:2]); inputs[:,:2] = temp.reshape((-1,2))
        scale_in_other, temp = scale_gaussian(inputs[:,2:]); inputs[:,2:] = temp.reshape((-1,4))

        scaler_out, output = scale_aod(output)
    elif flag in [4,6]:
    # box-cox doesn't work with negative values #
        inputs[:,0] += 0.05
        scale_in_aod, temp = scale_aod(inputs[:,0]); inputs[:,0] = temp.reshape((-1))
        scale_in_other, temp = scale_gaussian(inputs[:,1:]); inputs[:,1:] = temp.reshape((-1,3))

        scaler_out, output = scale_aod(output)

    joblib.dump(scale_in_aod, f'flag={flag:1d}_scaler_in_aod.sav') 
    joblib.dump(scale_in_other, f'flag={flag:1d}_scaler_in_other.sav') 
    joblib.dump(scaler_out, f'flag={flag:1d}_scaler_out.sav') 
    # #########################################################################
    # Numpy array >> Tensor
    # #########################################################################
    class CustomDataset(Dataset):
        def __init__(self):
            self.labels = output
            self.features = inputs

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            x = torch.FloatTensor(self.features[idx])
            y = torch.FloatTensor(self.labels[idx])
            return x, y

    dataset = CustomDataset()

    # split datas into train, validation, test
    train_size = int(n * 0.7)
    validation_size = int(n * 0.3)
    while (train_size+validation_size) != n:
        train_size+=1
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    # #########################################################################
    # Define Neural Network
    # #########################################################################
    class NeuralNetwork(nn.Module):
        def __init__(self, l1, l2, l3):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                    nn.Linear(input_cols, l1),
                    nn.BatchNorm1d(l1),
                    nn.ReLU(),
                    nn.Linear(l1, l2),
                    nn.BatchNorm1d(l2),
                    nn.ReLU(),
                    nn.Linear(l2, l3),
                    nn.BatchNorm1d(l3),
                    nn.ReLU(),
                    nn.Linear(l3, 1)
            )
        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    # #########################################################################
    # Define training
    # #########################################################################
    def train_fusion(config):
        model = NeuralNetwork(config['l1'], config['l2'], config['l3']).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        for epoch in range(20):
            running_loss = 0.
            epoch_steps = 0
            model.train()
            for i, data in enumerate(train_dataloader):
                X, y = data
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()

                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                epoch_steps += 1

            val_loss = 0.
            val_steps = 0
            total = 0
            for i, data in enumerate(validation_dataloader, 0):
                with torch.no_grad():
                    model.eval()
                    X, y = data
                    X, y = X.to(device), y.to(device)

                    pred = model(X)
                    # _, predicted = torch.max(pred.data, 1)
                    total += y.size(0)

                    val_loss += loss_fn(pred, y).cpu().numpy()
                    val_steps += 1

            torch.save(
                    (model.state_dict(), optimizer.state_dict()), './checkpoint.pt'
                )
            checkpoint = Checkpoint.from_directory('./')
            train.report(
                {'loss':(val_loss / val_steps)},
                checkpoint=checkpoint,
            )
    # #########################################################################
    # Hyper-parameter Tuning
    # #########################################################################
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train_fusion)
                            , resources={'cpu':num_cores})
                            , param_space=config
                            , tune_config = tune.TuneConfig(num_samples=num_samples, scheduler=scheduler)
                        )

    result = tuner.fit()
    best_result = result.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))

    fig = plt.figure(figsize=(8.6,7))
    ax = fig.add_subplot(1,1,1)

    for i in range(len(result)):
        ax.plot(result[i].metrics_dataframe.loss)
    plt.savefig(f'flag={flag:1d}_loss.png')
    plt.close()

    # #########################################################################
    # Best model test
    # #########################################################################
    model = NeuralNetwork(best_result.config['l1'], best_result.config['l2'], best_result.config['l3'])

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)

    # model.load_state_dict(best_result.checkpoint.to_dict()['model_weights'])
    model.to(device)

    # Saving model
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(f'flag={flag:1d}_fusion_model.pth') # Save
