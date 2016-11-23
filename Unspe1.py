
# coding: utf-8

# # Unspervised learning project 1

# In[6]:

from kohonen import *
from Supplementary import *
import numpy as np

name = "Stettler"
nameDigit = name2digits(name)
print(nameDigit)

# ### import data
data = np.loadtxt("data.txt")
labels = np.loadtxt("labels.txt")

data_filtered, labels_flitered = filterData(data,labels,name)
print(np.shape(data_filtered))
#visualizeSample(data[0])

centers = run_kohonen(data_filtered, size_k=6, sigma=3.0, eta=0.1, tmax=30000)
visualizeSample(centers, size_k=6)

average_digit = averageSample(data_filtered, labels_flitered, nameDigit)
visualizeSample(average_digit)

digits = []
for center in centers:
    rmses = []
    for digit in average_digit:
        rmses.append(distance_btw_Sample(center,digit))
    
    digits.append(nameDigit[np.argmin(rmses)])
    
print(digits)





