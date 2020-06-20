# coding: utf-8

import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# 构建网络，设置参数
class FullyConnectedAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(FullyConnectedAutoEncoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.bottleneck = tf.keras.layers.Dense(16, activation=tf.nn.relu)

        self.dense4 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(64, activation=tf.nn.relu)

        self.dense_final = tf.keras.layers.Dense(1440)

    def call(self, inp):
        x_reshaped = inp
        x = self.dense1(x_reshaped)
        x = self.dense2(x)
                
        x = self.bottleneck(x)
                
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense_final(x)
        return x, x_reshaped


class AutoEncoder:
    
    def init(self, data_path):
        # 读取文件
        self.d = pd.read_csv(data_path, header=None)
        self.x_train = np.array(self.d[0:100])
        self.x_test = np.array(self.d[100:110])

        # 模型参数
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.global_step = 0
        self.num_epochs = 50
        self.batch_size = 10
        self.mse_train = []
        self.mse_test = []

    # 定义损失函数
    def loss(self, x, x_bar):
        return tf.losses.mean_squared_error(x, x_bar)
    
    # 更新网络参数
    def grad(self, model, inputs):
        
        with tf.GradientTape() as tape:
            reconstruction, inputs_reshaped = model(inputs)
            loss_value = self.loss(inputs_reshaped, reconstruction)
                
        return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction
    
    # 训练网络
    def train(self):
        model = FullyConnectedAutoEncoder()
        for epoch in range(1, self.num_epochs+1):
            lst = []
            
            for x in range(0, len(self.x_train), self.batch_size):
                x_inp = tf.Variable(self.x_train[x: x + self.batch_size], dtype=tf.dtypes.float32)
                
                loss_value, grads, inputs_reshaped, reconstruction = self.grad(model, x_inp)
                
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

                mse = self.loss(inputs_reshaped, reconstruction).numpy().mean()
                lst.append(mse)
                
            pre, ori = model.call(self.x_test)    
            mse1 = self.loss(pre, ori).numpy().mean()
            print("Epoch: ", epoch)
            print('train loss {}'.format(mse), 'test loss {}'.format(mse1))
            self.mse_test.append(mse1)
            self.mse_train.append(np.mean(lst))
            
        model._set_inputs(x_inp)
        model.save('my_model')
        
    # 输出结果
    def show(self):
        x = range(len(self.mse_test))
        plt.figure()
        plt.plot(x, self.mse_test, label='mse_test')
        plt.plot(x, self.mse_train, label='mse_train')
        plt.grid()
        plt.legend()
        plt.show()
    
    # 进行预测
    def test(self, data):
        new_model = keras.models.load_model('my_model')
        return np.mean((data - new_model(data))**2)
    
    def run(self):
        self.train()
        self.show()


if __name__ == '__main__':
    ae = AutoEncoder()
    ae.init(r'.\test.csv')
    ae.run()
