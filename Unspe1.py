
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
#nb_exp = 1
#it_max = 1000000
#eta_range = [1E-4, 1E-3, 1E-2, 1E-1, 0.3,  1] #
#all_error = np.zeros((len(eta_range), nb_exp, it_max))
#for j, eta in enumerate(eta_range):
#    print('eta={}'.format(eta))
#    centers = np.zeros((nb_exp, 36, 28*28))
#    error = np.zeros((nb_exp, it_max))
#    for i in range(nb_exp):
#        cent, err = run_kohonen(data_filtered, size_k=6, sigma=3.0, eta=eta, tmax=it_max, convergence=2)
#        centers[i, :, :] = cent
#        error[i, :] = err 
#    save = {'centers' : centers, 'error': error}
#    np.save('save_eta={}_nbexp=1'.format(eta), save) #(centers, error))
#    all_error[j, :, :] = error
##average_digit = averageSample(data_filtered, labels_flitered, nameDigit)
#digits = assignDigit(centers[0, : ,:], average_digit, nameDigit)
#visualizeSample(centers[0, : ,:], size_k=6, labels=digits)
#visualizeError(all_error, opt=3, legend=eta_range)


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
#==============================================================================

#%%
"""
Fifth point:
    - Explore various number of kohonen/neighborhood function
"""
nb_exp = 30
it_max = 100000
sigma = [1,3,5]
size = [6,7,8]
for sig in sigma:
    for siz in size:
        centers = np.zeros((nb_exp, siz*siz, 28*28))
        error = np.zeros((nb_exp, it_max))
        for i in range(nb_exp):
            print('Sigma = {}, size = {}'.format(sig, siz))
            cent, err = run_kohonen(data_filtered, size_k=siz, sigma=sig, eta=1E-1, tmax=it_max, convergence=2)
            centers[i, : , :] = cent
            error[i, :] = err 
        save = {'centers' : centers, 'error': error}
        np.save('save_sig={}_size={}'.format(sig, siz), save)

#digits = assignDigit(centers, average_digit, nameDigit)

#%%
"""
Sixth point:
    - dynamic width that is decreasing over time.
"""
#==============================================================================
# nb_exp = 30
# it_max = 100000
# sigma_init = 3
# all_error = np.zeros((2, nb_exp, it_max))
# fun1 = lambda x: -sigma_init/it_max*x + sigma_init #linear
# fun2 = lambda x: np.exp(-np.power(x,2)/(it_max*1e6))/1.8 #gaussian
# for j, (fun, fun_str) in enumerate(zip([fun2], ['gaussian'])):
#     centers = np.zeros((nb_exp, 36, 28*28))
#     error = np.zeros((nb_exp, it_max))
#     for i in range(nb_exp):
#         cent, err = run_kohonen_dynamicLearningRate(data_filtered,fun,size_k=6, eta=0.1, tmax=it_max, convergence=2)
#         centers[i, :, :] = cent
#         error[i, :] = err 
#     save = {'centers' : centers, 'error': error}
#     np.save('save_fun={}'.format(fun_str), save) #save just in case there is a crash
#     all_error[j, :, :] = error
#     visualizeFun(fun, tmax)
#     digits = assignDigit(centers, average_digit, nameDigit)
#     visualizeSample(centers[0, :, :], 6, digits, fun_str)
# visualizeError(all_error, opt=3, legend=['linear', 'gaussian'])
#==============================================================================

