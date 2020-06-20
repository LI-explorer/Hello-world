# coding: utf-8

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


# 构建网络，设置参数
class Baseline(Model):
    
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same') 
        self.b1 = BatchNormalization()  
        self.a1 = Activation('relu')  
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  
        self.d1 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


class CnnTrainer:
    
    # 读取文件
    def init(self):
        self.fashion = tf.keras.datasets.fashion_mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.fashion.load_data()
        
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
    
    # 训练数据
    def train(self):
        self.model = Baseline()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['sparse_categorical_accuracy'])

        self.history = self.model.fit(self.x_train, self.y_train,  
                                      batch_size=32,epochs=5, 
                                      validation_data=(self.x_test, self.y_test),
                                      validation_freq=1)
        self.model.summary()
        
        self.model.save('my_model_cnn')
        
    # 输出结果
    def show(self):
        acc = self.history.history['sparse_categorical_accuracy']
        val_acc = self.history.history['val_sparse_categorical_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure()
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.grid()
        plt.legend()

        plt.figure()
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()

    # 保持模型
    def save(self):
        pass

    def run(self):
        self.train()
        self.show()


# Inference
class CnnInfer:

    # 读取文件
    def init(self):
        self.fashion = tf.keras.datasets.fashion_mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.fashion.load_data()

        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)

    def load_model(self):
        pass

    # 进行预测，只对一个样本
    def predict(self, datasets):
        (d, t), (d1, t1) = datasets

        new_model = tf.keras.models.load_model('my_model_cnn')
        new_model.fit(d, t, epochs=1)
        new_model.summary()


def _train():
    t = CnnTrainer()
    t.init()
    t.run()


def _infer():
    i = CnnInfer()
    i.init()
    model_path = './model'
    i.load(model_path)
    i.predict(one)


if __name__ == '__main__':
    _train()
    _infer()
