# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:57:09 2016

@author: jb
"""

import matplotlib.pyplot as plt
import numpy as np
import kohonen as ko

def visualizeSample(sample):
    
    if np.shape(sample) == (784,):
        sample = np.reshape(sample, (28,28))
        sample = sample[::-1, :]  #reverse lines for visualisation purpose
        
        #heatmap
        plt.figure(figsize=(5,5))
        plt.pcolor(sample, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
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
    
    
if __name__ == "__main__":
    print('Test visualize')
    data = np.loadtxt('data.txt')
    labels = np.loadtxt('labels.txt')
    sample = data[0]
    visualizeSample(sample)
    
    sample = data[0:3]
    visualizeSample(sample)
    