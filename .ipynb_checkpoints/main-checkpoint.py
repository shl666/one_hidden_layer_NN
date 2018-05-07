'''
File Name: main   
Author:    Shiming Luo
Date:      2018.05.06
'''

import numpy as np
import loader
from preprocessing import *
from architecture import *

if __name__ == '__main__':
    
    ### load data
    mnist = loader.MNIST('/MNIST/raw/')
    train_images, train_labels = mnist.load_training()
    test_images, test_labels = mnist.load_testing()


    train_images,train_labels = preprocess(train_images,train_labels,60000)

    #### set up train, validation, test sets
    x_train = np.matrix(train_images[:50000]) ## N1*785
    x_val = np.matrix(train_images[50000:])  ## N2*785
    t_train = np.matrix(train_labels[:50000]).T  ## N1*1
    t_val = np.matrix(train_labels[50000:]).T  ## N2*1

    #### pre-processing
    x_train = z_score(x_train) ## N1*785
    x_val = z_score(x_val) ## N2*785

    t_train_original = t_train
    t_val_original = t_val

    t_train = one_hot(t_train) ## N1*10
    t_val = one_hot(t_val) ## N2*10

    ### constrct a one hidden layer model
    OneHM = OneHiddenModel()

    ### convert train set and validation set to array
    x_train = np.array(x_train)
    t_train = np.array(t_train)
    x_val = np.array(x_val)
    t_val = np.array(t_val)

    ### train 30 epoches
    epoches = 30
    for epoch in range(epoches):
        for m in range(len(x_train)//OneHM.batch_size):
            X = x_train[m*OneHM.batch_size : (m+1)*OneHM.batch_size]
            t = t_train[m*OneHM.batch_size : (m+1)*OneHM.batch_size]
            OneHM.train(X, t)
        if (epoch+1)%3 == 0:
            print('epoch: {}/{}'.format(epoch+1,epoches))
            error,acc = OneHM.evaluation(x_train,t_train)
            print('train error: {:.4f}, train accuracy: {:.2f}%'.format(error,100*acc))
            error,acc = OneHM.evaluation(x_val,t_val)
            print('valid error: {:.4f}, valid accuracy: {:.2f}%'.format(error,100*acc))


