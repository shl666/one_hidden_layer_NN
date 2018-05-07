'''
File Name: preprocessing  
Author:    Shiming Luo
Date:      2018.05.06
'''
import numpy as np

def preprocess(images,labels,k):
    images,labels = images[:k],labels[:k] #select first k
    for i in range(len(images)):
        images[i] = [1]+images[i] # add bias
    return(images,labels)

def z_score(x):
    return(x/127.5 - 1)

def one_hot(t):
    temp = np.matrix(np.zeros((len(t),10)))
    for n in range(len(t)):
        temp[n,int(t[n])] = 1.0
    return(temp)