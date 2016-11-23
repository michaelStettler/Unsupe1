"""Python script for Exercise set 6 of the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
from Supplementary import *
import matplotlib.pyplot as plb
from copy import copy

def kohonen():
    """Example for using create_data, plot_data and som_step.
    """
#    plb.close('all')
    
    dim = 28*28
    data_range = 255.0
    
    # load in data and labels    
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')

    # select 4 digits    
    name = "Stettler"
    targetdigits = name2digits(name) # assign the four digits that should be used
    print(targetdigits) # output the digits that were selected

    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
    
    dy, dx = data.shape
    
    #set the size of the Kohonen map. In this case it will be 6 X 6
    size_k = 6
    
    #set the width of the neighborhood via the width of the gaussian that
    #describes it
    sigma = 2.0
    
    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range
    
    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))

    #set the learning rate
    eta = 0.9 # HERE YOU HAVE TO SET YOUR OWN LEARNING RATE
    
    #set the maximal iteration count
    tmax = 5000 # this might or might not work; use your own convergence criterion
    
    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)
    
    for t, i in enumerate(i_random):
        som_step(centers, data[i,:],neighbor,eta,sigma)

    # for visualization, you can use this:
    for i in range(size_k**2):
        plb.subplot(size_k,size_k,i)
        
        plb.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')
        plb.axis('off')
        
    # leave the window open at the end of the loop
    plb.show()
    plb.draw()

    
def run_kohonen(data, size_k: int=6, sigma: float=2.0, eta: int=0.9, 
                tmax: int=5000, convergence=0):
    """
    data = data
    size_k = size of the kohonen map
    sigma = width of Gaussian
    eta =  learning rate
    tmax = maximal iteration count
    """
    dim = 28*28
    data_range = 255.0
    dy, dx = data.shape
    
    #convergence criteria
    eps = 1E-6
    eps_2 = 0.1
    
    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range
    
    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))
    
    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)
    
    #error for convergence criterion
    error = [np.inf]
    
    print('start iteration')
    for t, i in enumerate(i_random):
        old_centers = copy(centers)
        som_step(centers, data[int(i),:],neighbor,eta,sigma)
            
        if t % 1E4 == 0:
            print('iteration {}'.format(t))
<<<<<<< HEAD
            
        if convergence == 1:
            #convergence: distance between samples and best matching prototypes 
            error.append(calculate_error(centers,data))
#            if np.abs((error[-2]-error[-1])/error[1]) < eps :
#                break
            
        elif convergence == 2:
=======
        
        error.append(calculate_error(centers,data))
        if convergence == 1:
            #convergence: distance between samples and best matching prototypes 
            if np.abs((error[-2]-error[-1])/error[1]) < eps :
                break
                
        if convergence == 2:
>>>>>>> origin/master
            #convergence: non significant weight update
            err = np.linalg.norm(centers-old_centers)
            error.append(err)
#            if err < eps_2:
#                break
            
    """ # for visualization, you can use this:
    for i in range(size_k**2):
        plb.subplot(size_k,size_k,i)
        
        plb.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')
        plb.axis('off')
        
    # leave the window open at the end of the loop
    plb.show()
    plb.draw() """
    
    print('Total iteration : {}'.format(t))
    return centers, error[1:]


def run_kohonen_dynamicLearningRate(data,fun,size_k: int=6, eta: float=0.1, tmax: int=5000):
    """
    data = data
    size_k = size of the kohonen map
    sigma = width of Gaussian
    tmax = maximal iteration count
    fun = function of t that will be use to compute the learning rate at each time step.
    """
    dim = 28*28
    data_range = 255.0
    dy, dx = data.shape
    
    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range
    
    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))
    
    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)
    
    for t, i in enumerate(i_random):
        sigma = fun(t)
        som_step(centers, data[i,:],neighbor,eta,sigma)

    """ # for visualization, you can use this:
    for i in range(size_k**2):
        plb.subplot(size_k,size_k,i)
        
        plb.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')
        plb.axis('off')
        
    # leave the window open at the end of the loop
    plb.show()
    plb.draw() """
    return centers
    

def som_step(centers,data,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a 
    self-organized map (SOM).
    
      centers = som_step(centers,data,neighbor,eta,sigma)
    
      Input and output arguments: 
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """
    size_k = int(np.sqrt(len(centers)))
    
    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))

    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)
        
    # update all units
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights        
        centers[j,:] += disc * eta * (data - centers[j,:])
        

def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))


def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.
     
     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """
    
    name = name.lower()
    
    if len(name)>25:
        name = name[0:25]
        
    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    
    n = len(name)
    
    s = 0.0
    
    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = int(np.mod(s,x.shape[0]))

    return np.sort(x[t,:])


def calculate_error(center,samples):
    
    error_samples = []
    
    for s in samples:
        error_temp = []
        for c in center:
            error_temp.append(distance_btw_Sample(s,c))
        error_samples.append(np.min(error_temp))

    return np.mean(error_samples)
    
    
#if __name__ == "__main__":
#    print(name2digits('Michael Stettler')) #[2 5 7 9]
#    kohonen()

