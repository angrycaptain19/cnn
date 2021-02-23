import torch
import os
import torch.nn as nn
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader  
import torchvision.datasets
import torchvision.transforms

batch_size = 270
transfrom = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
dataset = torchvision.datasets.ImageFolder('../picture', transform=transfrom)
train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size)

print(train_loader.sampler)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
define model
"""
#first data = 32 * 640 * 480 (depth, width, height)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(3, 16, 3, 1, padding=1)) # out = 32 * 640 * 480
        layer1.add_module('relu1', nn.ReLU())
        layer1.add_module('pool1',nn.MaxPool2d(2, 2)) # out = 32 * 320 * 240
        self.layer1 = layer1

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(16 * 320 * 240, 512))
        layer4.add_module('fc_relu1', nn.ReLU())
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU())
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4
        
    def forward(self, x):
        conv1 = self.layer1(x)
        fc_input = conv1.view(-1, conv1.size(0))
        return self.layer4(fc_input)

model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)

#study
num_epochs = 5
for _ in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        print('images.size: ', images.size())
        # print('images:',images)
        #print(labels)
        print('labels.size: ', labels.size())
        preds = model(images)
        print('preds:', preds)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

