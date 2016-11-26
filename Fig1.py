# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:21:12 2016

@author: JB
"""

import numpy as np
from Supplementary import *
from kohonen import *


#First preprocessing steps
name = "Stettler"
nameDigit = name2digits(name)
data = np.loadtxt("data.txt")
labels = np.loadtxt("labels.txt")
data_filtered, labels_flitered = filterData(data,labels,name)
average_digit = averageSample(data_filtered, labels_flitered, nameDigit)

#parameters
nb_exp = 30
it_max = 100000
eta_range = [1E-4, 1E-3, 1E-2, 1E-1, 0.3,  1] #

#save value
all_error = np.zeros((len(eta_range), nb_exp, it_max))
legend = []


for j, eta in enumerate(eta_range[::-1]):
    
    save = np.load('save_eta={}.npy'.format(eta)).item()
    all_error[j, :, :] = save['error']
    digit = assignDigit(save['centers'][0,:,:], average_digit, nameDigit)
    str_ = r'$\eta$ = {}'.format(eta)
    visualizeSample(save['centers'][0,:,:], 6, digit, str_)
    legend.append(str_)

visualizeError(all_error, opt=3, legend=legend)