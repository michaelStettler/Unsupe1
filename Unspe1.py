
from kohonen import *
from Supplementary import *
import numpy as np

name = "Stettler"
nameDigit = name2digits(name)
print(nameDigit)

# ### import data
data = np.loadtxt("data.txt")
labels = np.loadtxt("labels.txt")

#Obtain set of digit
name = "Stettler"
nameDigit = name2digits(name)
#print('Digits : {}'.format(nameDigit))

#Load Data
data = np.loadtxt("data.txt")
labels = np.loadtxt("labels.txt")

#keep data corresponding to the digit
data_filtered, labels_flitered = filterData(data,labels,name)
#print(np.shape(data_filtered))
#visualizeSample(data[0])


#%%
average_digit = averageSample(data_filtered, labels_flitered, nameDigit)

#centers, error = run_kohonen(data_filtered, size_k=6, sigma=3.0, eta=0.1, tmax=100, convergence=2)
#digits = assignDigit(centers, average_digit, nameDigit)
#visualizeSample(centers, size_k=6, labels=digits)
#visualizeError(error, opt=1)

#%%
"""
Second Point:
    - constant learning rate
    - sigma = 3
    - 6x6 Kohonen map
    - TODO: Implement convergence function    
"""
nb_exp = 100
it_max = 1000
for eta in [1E-1]:#, 1E-3, 1E-2, 1E-1, 1]:
    print('eta={}'.format(eta))
    centers = np.zeros((nb_exp, 36, 28*28))
    error = np.zeros((nb_exp, it_max))
    for i in range(nb_exp):
        cent, err = run_kohonen(data_filtered, size_k=6, sigma=3.0, eta=eta, tmax=it_max, convergence=2)
        centers[i, : , :] = cent
        error[i, :] = err 
    np.save('save', centers, error)
#average_digit = averageSample(data_filtered, labels_flitered, nameDigit)
digits = assignDigit(centers[0, : ,:], average_digit, nameDigit)
visualizeSample(centers[0, : ,:], size_k=6, labels=digits)
visualizeError(error, opt=2)


#==============================================================================
# #%%
# """
# Third point:
#     - Visualize prototypes
# """
# visualizeSample(centers, size_k=6)
# 
# #%%
# """
# Fourth point:
#     - automatically assign one digit to each prototypes that best represent it
# """
# average_digit = averageSample(data_filtered, labels_flitered, nameDigit)
# digits = assignDigit(centers, average_digit, nameDigit)
# visualizeSample(average_digit)
# print('assigned digit : {}'.format(digits))
# visualizeSample(centers, size_k=6, labels=digits)
# 
# #%%
# """
# Fifth point:
#     - Explore various number of kohonen/neighborhood function
# """
# sigma = [1,3,5]
# size = [6,7,8]
# for sig in sigma:
#     for siz in size:
#         print('Sigma = {}, size = {}'.format(sig, siz))
#         centers = run_kohonen(data_filtered, size_k=siz, sigma=sig, eta=0.1, tmax=100000)
#         digits = assignDigit(centers, average_digit, nameDigit)
#         visualizeSample(centers, size_k=siz, labels=digits)
# 
# #%%
# """
# Sixth point:
#     - dynamic width that is decreasing over time.
# """
# tmax = 10000
# sigma_init = 2
# #fun = lambda x: -sigma_init/tmax*x + sigma_init #linear
# fun = lambda x: np.exp(-np.power(x,2)/(tmax*1e6))/1.8 #gaussian
# centers = run_kohonen_dynamicLearningRate(data_filtered,fun,size_k=6, eta=0.1, tmax=tmax)
# visualizeFun(fun, tmax)
# visualizeSample(centers, size_k=6)
#==============================================================================

