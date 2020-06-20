# coding: utf-8

import torch
import pandas as pd
import numpy as np
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module, CrossEntropyLoss
from torch.optim import Adam


# 搭建网络
class Cnn(Module):
    
    def __init__(self, out_1=13, out_2=32):
        super(Cnn, self).__init__()
        self.cnn1 = Conv2d(in_channels=3, out_channels=out_1, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.cnn2 = Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=0)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.fc1 = Linear(out_2 * 23 * 23, 2)
    
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class CnnTorch:
    
    # 读取数据
    def init(self, data_path):
        self.d = pd.read_csv(data_path, header=None)
        self.train_loader = torch.tensor(np.array(self.d[0:100]), dtype=torch.float32)
        self.validation_loader = torch.tensor(np.array(self.d[100:110]), dtype=torch.float32)
        
        self.loss_list = []
        self.accuracy_list = []
        self.N_test = 10
        
    # 训练模型 
    def train(self):
        net = Cnn()
        optimizer = Adam(net.parameters(), lr=0.07)
        criterion = CrossEntropyLoss()
        
        for epoch in range(100):
            for x, y in self.train_loader:
                optimizer.zero_grad()
                z = net(x.float())
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()

            correct=0
            for x_test, y_test in self.validation_loader:
                z = net(x_test)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y_test).sum().item()
            accuracy = correct / self.N_test
            self.accuracy_list.append(accuracy)
            self.loss_list.append(loss.data)
            
    def run(self):
        self.train()


if __name__ == '__main__':
    a = CnnTorch()
    a.init(r'\test.csv')
    a.run()



