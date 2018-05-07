'''
File Name: architecture   
Author:    Shiming Luo
Date:      2018.05.06
'''

import numpy as np

class OneHiddenModel():
    def __init__(self,in_features = 784, hid_features = 64, out_features = 10,\
                 batch_size = 128, bias = 1, lr = 0.01):
        self.lr = lr
        self.batch_size = batch_size
        self.bias = bias
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.in_layer = np.zeros((batch_size, self.in_features + self.bias))
        self.hid_layer = np.zeros((batch_size, self.hid_features + self.bias))
        self.out_layer = np.zeros((batch_size, self.out_features))
        self.in2hid_weight = np.random.normal(0,0.05,[self.in_features + self.bias,self.hid_features])
        self.hid2out_weight = np.random.normal(0,0.05,[self.hid_features + self.bias, self.out_features])
        self.y = np.zeros(self.batch_size)
        self.error = 0
        self.delta2 = np.zeros((self.batch_size, self.out_features))
        self.delta1 = np.zeros((self.batch_size, self.hid_features))
        
    def sigmoid(self, a):
        return 1/(1+np.exp(-a))

    def softmax(self, a): 
        return np.exp(a)/(sum(np.exp(a).T).reshape(-1,1)) 

    def calE(self, target, y):
        error = -sum(sum(np.array(target)*np.array(np.log(y))).T)/len(target)
        return error
    
    def forward(self, data):
        self.in_layer = data
        temp_hid_layer = np.dot(self.in_layer, self.in2hid_weight)
        temp_hid_layer = self.sigmoid(temp_hid_layer)
        self.hid_layer = np.insert(temp_hid_layer, self.hid_features, 1, axis = 1)
        self.out_layer = np.dot(self.hid_layer, self.hid2out_weight)
        self.y = self.softmax(self.out_layer)
    
    def backward(self, data, target):
        self.delta2 = target - self.y
        self.delta1 = self.hid_layer[:,:-1] * \
                        (1-self.hid_layer[:,:-1]) * \
                        np.dot(self.delta2,self.hid2out_weight.T[:,:-1])
        self.in2hid_weight += self.lr * np.dot(data.T, self.delta1) / self.batch_size
        self.hid2out_weight += self.lr * np.dot(self.hid_layer.T, self.delta2) / self.batch_size
        
    def predict(self, data):
        in_layer = data
        hid_layer = self.sigmoid(np.dot(data,self.in2hid_weight))
        hid_layer = np.insert(hid_layer, len(data), 1, axis = 1)
        out_layer = np.dot(hid_layer, self.hid2out_weight)
        y = self.softmax(out_layer)
        return y
        
    def evaluation(self, data, target):
        self.forward(data)
        error = self.calE(target, self.y)
        count = 0
        for i in range(len(self.y)):
            temp1 = self.y[i].tolist()
            temp2 = target[i].tolist()
            if temp2.index(max(temp2)) == temp1.index(max(temp1)):
                count += 1
        acc = count/len(self.y)
        return error, acc
    
    def train(self, data, target):
        self.forward(data)
        self.backward(data, target)
        self.error,_ = self.evaluation(data, target)
        
class TwoHiddenModel():
    def __init__(self,in_features = 784, hid1_features = 64, hid2_features = 128, out_features = 10,\
                 batch_size = 128, bias = 1, lr = 0.01):
        self.lr = lr
        self.batch_size = batch_size
        self.bias = bias
        self.in_features = in_features
        self.hid1_features = hid1_features
        self.hid2_features = hid2_features
        self.out_features = out_features
        self.in_layer = np.zeros((batch_size, self.in_features + self.bias))
        self.hid1_layer = np.zeros((batch_size, self.hid1_features + self.bias))
        self.hid1_layer = np.zeros((batch_size, self.hid2_features + self.bias))
        self.out_layer = np.zeros((batch_size, self.out_features))
        self.in2hid1_weight = np.random.normal(0,0.05,[self.in_features + self.bias, self.hid1_features])
        self.hid12hid2_weight = np.random.normal(0,0.05,[self.hid1_features + self.bias, self.hid2_features])
        self.hid22out_weight = np.random.normal(0,0.05,[self.hid2_features + self.bias, self.out_features])
        self.y = np.zeros(self.batch_size)
        self.error = 0
        self.delta3 = np.zeros((self.batch_size, self.out_features))
        self.delta2 = np.zeros((self.batch_size, self.hid2_features))
        self.delta1 = np.zeros((self.batch_size, self.hid1_features))
        
        
    def sigmoid(self, a):
        return 1/(1+np.exp(-a))

    def softmax(self, a): 
        return np.exp(a)/(sum(np.exp(a).T).reshape(-1,1)) 

    def calE(self, target, y):
        error = -sum(sum(np.array(target)*np.array(np.log(y))).T)/len(target)
        return error
    
    def forward(self, data):
        self.in_layer = data
        
        temp_hid1_layer = np.dot(self.in_layer, self.in2hid1_weight)
        temp_hid1_layer = self.sigmoid(temp_hid1_layer)
        self.hid1_layer = np.insert(temp_hid1_layer, self.hid1_features, 1, axis = 1)
        
        temp_hid2_layer = np.dot(self.hid1_layer, self.hid12hid2_weight)
        temp_hid2_layer = self.sigmoid(temp_hid2_layer)
        self.hid2_layer = np.insert(temp_hid2_layer, self.hid2_features, 1, axis = 1)
        
        self.out_layer = np.dot(self.hid2_layer, self.hid22out_weight)
        self.y = self.softmax(self.out_layer)
    
    def backward(self, data, target):
        self.delta3 = target - self.y
        self.delta2 = self.hid2_layer[:,:-1] * \
                        (1-self.hid2_layer[:,:-1]) * \
                        np.dot(self.delta3,self.hid22out_weight.T[:,:-1])
        self.delta1 = self.hid1_layer[:,:-1] * \
                        (1-self.hid1_layer[:,:-1]) * \
                        np.dot(self.delta2,self.hid12hid2_weight.T[:,:-1])
        self.in2hid1_weight += self.lr * np.dot(data.T, self.delta1) / self.batch_size
        self.hid12hid2_weight += self.lr * np.dot(self.hid1_layer.T, self.delta2) / self.batch_size
        self.hid22out_weight += self.lr * np.dot(self.hid2_layer.T, self.delta3) / self.batch_size
        
        
    def predict(self, data):
        in_layer = data
        hid1_layer = self.sigmoid(np.dot(data,self.in2hid1_weight))
        hid1_layer = np.insert(hid1_layer, len(data), 1, axis = 1)
        hid2_layer = self.sigmoid(np.dot(hid1_layer,self.hid12hid2_weight))
        hid2_layer = np.insert(hid2_layer, len(hid1_layer), 1, axis = 1)
        out_layer = np.dot(hid2_layer, self.hid22out_weight)
        y = self.softmax(out_layer)
        return y
        
    def evaluation(self, data, target):
        self.forward(data)
        error = self.calE(target, self.y)
        count = 0
        for i in range(len(self.y)):
            temp1 = self.y[i].tolist()
            temp2 = target[i].tolist()
            if temp2.index(max(temp2)) == temp1.index(max(temp1)):
                count += 1
        acc = count/len(self.y)
        return error, acc
    
    def train(self, data, target):
        self.forward(data)
        self.backward(data, target)
        self.error,_ = self.evaluation(data, target)
        
        
        