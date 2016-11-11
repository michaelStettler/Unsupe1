# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:57:09 2016

@author: jb
"""

import matplotlib.pyplot as plt
import numpy as np

def VisualizeSample(sample):
    
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
            VisualizeSample(d)
    return None
    
    
    
if __name__ == "__main__":
    print('Test visualize')
    data = np.loadtxt('data.txt')
    sample = data[0]
    VisualizeSample(sample)
    
    sample = data[0:3]
    VisualizeSample(sample)
    