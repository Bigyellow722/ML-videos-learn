import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.optim as optim
from tqdm import *
import matplotlib.pyplot as plt
import time

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def default_loader(path):
    return Image.open(path).convert('L')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        img_entrys = []
        train_file_entrys =open(txt, 'r')

        for line in train_file_entrys:
            line = line.strip('\n')
            line = line.rstrip('\n')

            entrys = line.split()

            img_entrys.append((entrys[0], int(entrys[1])))

        self.imgs = img_entrys
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img_path, index = self.imgs[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, index

train_transforms = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor()]
)

test_transforms = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor()]
)

train_data = MyDataset(txt='train.txt', transform=train_transforms)
test_data = MyDataset(txt='test.txt', transform=test_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=1)

print(train_data.__getitem__(1)[0].shape)


#Defining the convolutional neural network
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


model = LeNet(num_classes).to(device)

#Setting the loss function
cost = nn.CrossEntropyLoss()

#Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#this is defined to print how many steps are remaining when training
total_step = len(train_loader)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)
        	
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        		
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

