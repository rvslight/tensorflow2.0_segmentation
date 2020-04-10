import numpy as np

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import onnx
# from onnx_tf.backend import prepare
# import tensorflow as tf

# Generate data
train_size = 8000
test_size = 2000

input_size = 20
hidden_sizes = [50,50]
output_size = 1
num_classes = 2

x_train = np.random.randn(train_size, input_size).astype(np.float32)
x_test = np.random.randn(test_size, input_size).astype(np.float32)
y_train = np.random.randint(num_classes, size=train_size)
y_test = np.random.randint(num_classes, size=test_size)

#Define data set
class SimpleDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#Create dataloader
train_loader = DataLoader(dataset=SimpleDataset(x_train,y_train),batch_size=8,shuffle=True)
test_loader = DataLoader(dataset=SimpleDataset(x_test,y_test),batch_size=8,shuffle=False)

##Build model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleModel,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fcs = [] # List of fcn layers
        next_in_size = input_size

        for i, next_out_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_features=next_in_size, out_features=next_out_size)
            next_in_size = next_out_size
            self.__setattr__('fc{}'.format(i), fc) # set name each layer.
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_features=next_in_size, out_features=output_size)

    def forward(self, x):
        for i,fc in enumerate(self.fcs):
            x = fc(x)
            x = nn.ReLU()(x)
        out = self.last_fc(x)
        return nn.Sigmoid()(out)

# set device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
model_pytorch = SimpleModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
model_pytorch = model_pytorch.to(device)

# set loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model_pytorch.parameters(), lr=1e-3)

# train 20 epoch
num_epochs = 5

# train model
time_start = time.time()

for epoch in range(num_epochs):
    model_pytorch.train()

    train_loss_total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model_pytorch(data).squeeze()
        target = target.float()
        train_loss = criterion(output,target)
        train_loss.backward()
        optimizer.step()
        train_loss_total += train_loss.item() * data.size(0)

    print(f'Epoch{epoch+1} completed. Train loss is {train_loss_total/train_size}')

#Eval model
model_pytorch.eval()

test_loss_total = 0
total_num_corrects = 0
threshold = 0.5
time_start = time.time()

for data,target in test_loader:
    data,target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model_pytorch(data).squeeze()
    target = target.float()
    test_loss = criterion(output,target)
    test_loss.backward()
    optimizer.step()
    test_loss_total += test_loss.item() * data.size(0)

    pred = (output >= threshold).view_as(target) # make same shape pred and target
    num_correct = torch.sum(pred == target.byte()).item()
    total_num_corrects += num_correct

print(f'Eval completed. Test loss is {test_loss_total/test_size}')
print(f'Test accuracy is {total_num_corrects/test_size}')
print(f'Time {time.time()-time_start}/60')

## save model
if not os.path.exists('./model/'):
    os.mkdir('./model/')
torch.save(model_pytorch.state_dict(), './model/model_simple.pt')

