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
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1',nn.MaxPool2d(2, stride= 2)) # out = 32 * 320 * 240
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(16, 32, 3, 1, padding=1)) # out = 64 * 320 * 240 
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, stride= 2)) # out = 64 * 160 * 120
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(32, 64, 3, 1, padding=1)) # out = 64 * 160 * 120
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, stride= 2)) # out = 64 * 80 * 60
        self.layer3 = layer3
        
        add_layer1 = nn.Sequential()
        add_layer1.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1)) # out = 128 * 80 * 60
        add_layer1.add_module('relu3', nn.ReLU(True))
        add_layer1.add_module('pool3', nn.MaxPool2d(2, stride= 2)) # out = 128 * 40 * 30
        self.add_layer1 = add_layer1

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(128 * 40 * 30, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4
        
    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        add_conv1 = self.add_layer1(conv3)
        fc_input = add_conv1.view(-1, add_conv1.size(0))
        return self.layer4(fc_input)

model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)

#study
num_epochs = 5
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        print('images.size: ', images.size())
        # print('images:',images)
        #print(labels)
        print('labels.size: ', labels.size())
        preds = model(images)
        #print('preds:', preds)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print('epoch {}, batch{}, loss = {:g}'.format(epoch, idx, loss.item()))

