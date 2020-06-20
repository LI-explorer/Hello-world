# coding: utf-8

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置网络
class Ae(nn.Module): 
     
    def __init__(self):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=1440, out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=1440)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        
        reconstructed = torch.relu(activation)
        return reconstructed


class AutoEncoder:
    def init(self, data_path):
        
        # 读取文件
        self.d = pd.read_csv(data_path, header=None)
        self.x_train = torch.tensor(np.array(self.d[0:100]), dtype=torch.float32)
        self.x_test = torch.tensor(np.array(self.d[100:110]), dtype=torch.float32)
        
        # 模型参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 10
        self.epochs = 50
        self.l1 = []
        self.l2 = []
        
    # 训练网络   
    def train(self):
        model = Ae()
        
        model = Ae().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
        for epoch in range(self.epochs):
            loss = 0
            for x in range(0, len(self.x_train), self.batch_size):
                batch = torch.tensor(self.x_train[x : x + self.batch_size], dtype=torch.float32)
                self.optimizer.zero_grad()
                outputs = model(batch)
                train_loss = self.criterion(outputs, batch)
                train_loss.backward()
                self.optimizer.step()
                loss += train_loss.item()

            loss = loss / (len(self.x_train) / self.batch_size)
            
            o = model(self.x_test)
            l = self.criterion(o, self.x_test)
            self.l1.append(loss)
            self.l2.append(l/10)
            
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, self.epochs, loss))
            
    # 输出结果        
    def show(self):
        n = range(50)
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(n, self.l1, color='orange', label='train')
        plt.grid()
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(n, self.l2, label='test')
        plt.legend()
        plt.grid()
        
        plt.show()
      
    def run(self):
        self.train()
        self.show()


if __name__ == '__main__':
    ae = AutoEncoder()
    ae.init(r'\test.csv')
    ae.run() 
    
    

