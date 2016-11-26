# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:57:09 2016

@author: jb & michael
"""

import matplotlib.pyplot as plt
import numpy as np
import kohonen as ko

def visualizeSample(sample, size_k=None, labels=None, title=None):
    if not labels:
        if np.shape(sample) == (784,):
            labels = ''
        else:
            labels = [''] * len(sample)
#    else:
#        labels = ['Digit {}'.format(l) for l in labels]
  
    if size_k:
        plt.figure(figsize=(2.5 * int(size_k), 2.5 * int(size_k)))
        for i, (sample_, label) in enumerate(zip(sample, labels)):    
            plt.subplot(size_k, size_k, i+1)
            s = np.reshape(sample_, (28,28))
            s = s[::-1, :]  #reverse lines for visualisation purpose
        
            #heatmap
            plt.pcolor(s, cmap=plt.get_cmap('jet'), vmin=0, vmax=255)
            plt.axis('off')
            plt.xlim([0, 27])
            plt.ylim([0, 27])
            plt.text(1.5, 20, label, fontsize=30, color='white')
        
        if title:
            plt.suptitle(title, fontsize=30)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
        plt.draw()
    
    elif np.shape(sample) == (784,):
        sample = np.reshape(sample, (28,28))
        sample = sample[::-1, :]  #reverse lines for visualisation purpose
        
        #heatmap
        plt.figure(figsize=(5,5))
        plt.pcolor(sample, cmap=plt.get_cmap('jet'), vmin=0, vmax=255)
        plt.axis('off')
        plt.xlim([0, 27])
        plt.ylim([0, 27])
        plt.text(1.5, 23, labels, fontsize=40, color='white')
        if title:
            plt.title(title)
        plt.draw()
    
    else:
        for d, l in zip(sample, labels):
            visualizeSample(d, labels=l)
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
    return np.sqrt(np.sum(np.power(sample1-sample2,2)))
    

def assignDigit(centers, average, nameDigit):
    digits = []
    for center in centers:
        rmses = []
        for digit in average:
            rmses.append(distance_btw_Sample(center,digit))
        
        digits.append(nameDigit[np.argmin(rmses)])
    return digits

    
def visualizeFun(fun, xmax=None):    
    plt.figure()
    x = np.arange(xmax)
    plt.plot(x, fun(x))
    plt.draw()
    return None


def visualizeError(error, opt=1, legend=None):
    
    plt.figure()
    if opt == 3: #several parameters tested several times
        cmap = plt.get_cmap('RdBu')
        color = cmap(np.linspace(0, 1, len(error)))
        for err, c in zip(error, color):
            SEM = np.std(err, axis=0)/float(np.sqrt(np.shape(err)[0]))
            EM = np.mean(err, axis=0)
            plt.loglog(EM, color=c, linewidth=2)
            plt.fill_between(np.arange(len(SEM)), EM + SEM, EM - SEM, color=c, alpha=0.5)
        if legend:
            plt.legend(legend)
            
    elif opt == 2: #one parameter tested several times
        SEM = np.std(error, axis=0)/float(np.sqrt(np.shape(error)[0]))
        EM = np.mean(error, axis=0)
        plt.plot(EM)
        plt.fill_between(np.arange(len(SEM)), EM + SEM, EM - SEM, alpha=0.5)
        
    else: #one parameter tested one time.
        plt.plot(error)
        
    plt.xlabel('iterations', fontsize=15)
    plt.ylabel(r'$||\Delta\omega_j||$', fontsize=15)
    plt.draw()
    return None

if __name__ == "__main__":
    print('Test visualize')
    data = np.loadtxt('data.txt')
    labels = np.loadtxt('labels.txt')
    sample = data[0]
    visualizeSample(sample, labels='5')
    
    sample = data[0:3]
    visualizeSample(sample)
    