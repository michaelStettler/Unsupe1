# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:57:09 2016

@author: jb & michael
"""

import matplotlib.pyplot as plt
import numpy as np
import kohonen as ko

def visualizeSample(sample, size_k=None,labels=None):
  
    if size_k:
        plt.figure(figsize=(2.5 * int(size_k), 2.5 * int(size_k)))
        for i, sample_ in enumerate(sample):    
            plt.subplot(size_k, size_k, i+1)
            s = np.reshape(sample_, (28,28))
            s = s[::-1, :]  #reverse lines for visualisation purpose
        
            #heatmap
            plt.pcolor(s, cmap=plt.get_cmap('jet'), vmin=0, vmax=255)
            plt.axis('off')
            plt.xlim([0, 27])
            plt.ylim([0, 27])
            if labels != None:
                plt.xlabel(labels[i])
                plt.title(labels[i])
           
        plt.show()
    
    elif np.shape(sample) == (784,):
        sample = np.reshape(sample, (28,28))
        sample = sample[::-1, :]  #reverse lines for visualisation purpose
        
        #heatmap
        plt.figure(figsize=(5,5))
        plt.pcolor(sample, cmap=plt.get_cmap('jet'), vmin=0, vmax=255)
        plt.axis('off')
        plt.xlim([0, 27])
        plt.ylim([0, 27])
        plt.show()
    
    else:
        for d in sample:
            visualizeSample(d)
    return None
    

def filterData(data, labels, name):
    # select 4 digits    
    targetdigits = ko.name2digits(name) # assign the four digits that should be used
    print("target digits = {}".format(targetdigits)) # output the digits that were selected

    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
    labels = labels[np.logical_or.reduce([labels==x for x in targetdigits])]

    return data, labels
    

def averageSample(data,labels,targetDigits):
    
    dt_mean = []
    for i in targetDigits :
        dt = data[np.where(labels == i),:][0]
        dt_mean.append(np.mean(dt,axis=0))
        
    print(np.shape(dt_mean))
    
    return dt_mean


def distance_btw_Sample(sample1,sample2):
    return np.sum(np.power(sample1-sample2,2))


if __name__ == "__main__":
    print('Test visualize')
    data = np.loadtxt('data.txt')
    labels = np.loadtxt('labels.txt')
    sample = data[0]
    visualizeSample(sample)
    
    sample = data[0:3]
    visualizeSample(sample)
    