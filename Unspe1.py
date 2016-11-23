
from kohonen import *
from Supplementary import *
import numpy as np

<<<<<<< HEAD
name = "Stettler"
nameDigit = name2digits(name)
print(nameDigit)

# ### import data
data = np.loadtxt("data.txt")
labels = np.loadtxt("labels.txt")

=======
#Obtain set of digit
name = "Stettler"
nameDigit = name2digits(name)
#print('Digits : {}'.format(nameDigit))

#Load Data
data = np.loadtxt("data.txt")
labels = np.loadtxt("labels.txt")

#keep data corresponding to the digit
>>>>>>> origin/master
data_filtered, labels_flitered = filterData(data,labels,name)
#print(np.shape(data_filtered))
#visualizeSample(data[0])

<<<<<<< HEAD
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
=======
#%%
#average_digit = averageSample(data_filtered, labels_flitered, nameDigit)
#
#for e in np.arange(0.1, 1, 0.1):
#    centers = run_kohonen(data_filtered, size_k=6, sigma=3.0, eta=e, tmax=1E6, convergence=0)
#    digits = assignDigit(centers, average_digit, nameDigit)
#    visualizeSample(centers, size_k=6, labels=digits)

#%%
"""
Second Point:
    - constant learning rate
    - sigma = 3
    - 6x6 Kohonen map
    - TODO: Implement convergence function    
"""
for eta in [1E-4, 1E-3, 1E-2, 1E-1, 1]:
    print('eta={}'.format(eta))
    centers = np.zeros((10, 36, 28*28))
    error = np.zeros((10, int(1E6 -1)))
    for i in range(10):
        cent, err = run_kohonen(data_filtered, size_k=6, sigma=3.0, eta=eta, tmax=1E6, convergence=0)
        centers[i, : , :] = cent
        error[i, :] = error 
    np.save('save_eta={}'.format(eta), (centers, error))
#average_digit = averageSample(data_filtered, labels_flitered, nameDigit)
#digits = assignDigit(centers, average_digit, nameDigit)
#visualizeSample(centers, size_k=6, labels=digits)
#visualizeError(error, opt=1)


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

>>>>>>> origin/master





