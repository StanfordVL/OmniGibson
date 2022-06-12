import copy

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
from typing import Optional, Any
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import ModuleList
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time

class OmniDataset(Dataset):
    def __init__(self, mode):
        self.data = h5py.File('dataset_sim2real_raw.hdf5', 'r', driver='core')
        self.test_size = 10
        self.train_size = len(self.data) - self.test_size

        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return self.train_size
        else:
            return self.test_size

    def switch_mode(self):
        if self.mode == 'train':
            self.mode = 'test'
        else:
            self.mode = 'train'

    def __getitem__(self, idx):
        if self.mode == 'test':
            idx += self.train_size

        random_idx = random.randint(0, len(self.data['trial_{0}'.format(idx)]['act'])-1)

        obs = torch.from_numpy(np.array(self.data['trial_{0}'.format(idx)]['rgb'][random_idx]).astype(np.float32))
        gri = torch.from_numpy(np.array([self.data['trial_{0}'.format(idx)]['gri'][random_idx]]).astype(np.float32))
        im_x, im_y = obs.size()
        print(obs.size())
        gri = gri.view(1, 1, 1).repeat(im_x, im_y, 1)
        obs = torch.cat((obs, gri), dim=2)

        act = torch.from_numpy(np.array([self.data['trial_{0}'.format(idx)]['act'][random_idx]]).astype(np.int16))

        return obs, act


class PrimModel(nn.Module):
    def __init__(self, num_class):
        super(PrimModel, self).__init__()
        self.num_class = num_class
        self.cnn = models.resnet18(num_classes=self.num_class)

    def forward(self, x):

        return self.cnn(x)



def train(dataloader, model):
    for batch, data in enumerate(dataloader, 0):
        obs, gt = data
        obs, gt = Variable(obs).cuda(), Variable(gt).cuda()

        output_act = model(obs)

        loss = criterion(output_act.view(-1, 2), gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        print("Train step ", batch, loss.item())


def eval(dataloader, model):
    total_loss = 0.
    count = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader, 0):
            obs, gt = data
            obs, gt = Variable(obs).cuda(), Variable(gt).cuda()

            output_act = model(obs)

            loss = criterion(output_act.view(-1, 2), gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            print("Test step ", batch, loss.item())

            total_loss += loss.item()
            count += 1

    return total_loss / count



batch_size = 16
lr = 0.0001
epochs = 30

model = PrimModel(6)

dataset_train = OmniDataset('train')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = OmniDataset('test')
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = float('inf')
best_model = None

for epoch in range(epochs):
    train(dataloader_train, model)
    val_loss = eval(dataloader_test, model)

    if (val_loss < best_val_loss) or (epoch % 10 == 0):
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'trained_models/best_model_{0}_{1}.pth'.format(epoch, val_loss))
        print("Best model saved!!!!!!!!!! score: {0}".format(val_loss))