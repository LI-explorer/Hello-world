# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn import datasets
import numpy as np

class Iris2:
    
    def init(self):
        
        # 读取数据
        self.x_data = datasets.load_iris().data
        self.y_data = datasets.load_iris().target

        np.random.seed(116)
        np.random.shuffle(self.x_data)
        np.random.seed(116)
        np.random.shuffle(self.y_data)
        tf.random.set_seed(116)

        self.x_train = self.x_data[:-30]
        self.y_train = self.y_data[:-30]
        self.x_test = self.x_data[-30:]
        self.y_test = self.y_data[-30:]

        self.x_train = tf.cast(self.x_train, tf.float32)
        self.x_test = tf.cast(self.x_test, tf.float32)

        self.train_db = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(32)
        self.test_db = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(32)
        
        # 模型参数
        self.w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
        self.b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

        self.lr = 0.1
        self.train_loss_results = []
        self.test_acc = []
        self.epoch = 50
        self.loss_all = 0
        
    # 训练数据
    def train(self):
        for epoch in range(self.epoch):
            for step, (x_train, y_train) in enumerate(self.train_db):
                with tf.GradientTape() as tape:
                    
                    y = tf.matmul(self.x_train, self.w1) + self.b1
                    y = tf.nn.softmax(y)
                    y_ = tf.one_hot(self.y_train, depth=3)
                    loss = tf.reduce_mean(tf.square(y-y_))
                    self.loss_all += loss.numpy()
                    
                grads = tape.gradient(loss, [self.w1, self.b1])

                self.w1.assign_sub(self.lr*grads[0])
                self.b1.assign_sub(self.lr*grads[1])

            print('epoch {}, loss:{}'.format(epoch, self.loss_all / 4))
            
            self.train_loss_results.append(self.loss_all/4)
            self.loss_all=0

            total_correct, total_number = 0, 0
            
            for x_test, y_test in self.test_db:
                y = tf.matmul(x_test, self.w1) + self.b1
                y = tf.nn.softmax(y)
                
                pred = tf.argmax(y, axis=1)
                pred = tf.cast(pred, dtype=y_test.dtype)
                
                correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
                correct = tf.reduce_sum(correct)
                
                total_correct += int(correct)
                total_number += x_test.shape[0]
                
            acc = total_correct / total_number
            self.test_acc.append(acc)
            
            print('test_acc:', acc)
            print('-------------')

        np.save('my_model_iris2_w',self.w1)
        np.save('my_model_iris2_b',self.b1)

    # 验证模型
    def test(self, database):
        
        x_data = database.data
        y_data = database.target
        
        w = np.load('my_model_iris2_w.npy')
        b = np.load('my_model_iris2_b.npy')
        
        y1 = tf.matmul(x_data, w) + b
        y1 = tf.nn.softmax(y1)
                
        pred1 = tf.argmax(y1, axis=1)
        pred1 = tf.cast(pred1, dtype=y_data.dtype)
                
        correct1 = tf.cast(tf.equal(pred1, y_data), dtype=tf.int32)
        correct1 = tf.reduce_sum(correct1)
        
        total_correct1, total_number1 = 0, 0

        total_correct1 += int(correct1)
        total_number1 += x_data.shape[0]
                
        acc1 = total_correct1 / total_number1
        
        return acc1
    
    def main(self):
        self.init()
        self.train()


if __name__ == '__main__':
    b = Iris2()
    b.main()
