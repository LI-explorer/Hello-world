# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from matplotlib import pyplot as plt


class Iris1:
    
    # 读取文件
    def init(self):
        self.x_train = datasets.load_iris().data  
        self.y_train = datasets.load_iris().target
        
    # 训练网络
    def train(self):
        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())])
        
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['sparse_categorical_accuracy'])

        self.history = self.model.fit(self.x_train,
                                      self.y_train,
                                      batch_size=32,
                                      epochs=50,
                                      validation_split=0.2,
                                      validation_freq=1)
        self.model.summary()
        
        self.model.save('my_model_iris1')
        
    # 输出结果   
    def show(self):
        acc = self.history.history['sparse_categorical_accuracy']
        loss = self.history.history['loss']

        plt.figure()
        plt.plot(acc, label='Training Accuracy')
        plt.title('Training Accuracy')
        plt.grid()
        plt.legend()

        plt.figure()
        plt.plot(loss, label='Training Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid()
        plt.show()
    
    # 进行预测
    def test(self, database):
        d,t = (database.data, database.target)
        
        new_model = keras.models.load_model('my_model_iris1')
        new_model.fit(d, t, epochs=1)
        new_model.summary()
        
    def run(self):
        self.train()
        self.show()


if __name__ == '__main__':
    a = Iris1()
    a.init()
    a.run()

        

