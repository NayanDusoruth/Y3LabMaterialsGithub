#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:16:39 2026

@author: nayandusoruth
"""

from numpy.random import rand
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


from numba import jit
from numba import njit
import numba as nb

import copy
import time

import NayanGeneralUtils.plotting as plotter

# ==========================================================================================================================================
# code utilities
# ==========================================================================================================================================

def beta(T):
    k_b = 1 # 1.38 * 10**(-23)
    return 1 / (k_b * T)

# utility function - returns the adjacent indicies to "index" - note modulo operation over size to handle edge cases - </function verified/>
# utility function - returns the adjacent indicies to "index" - note modulo operation over size to handle edge cases - </function verified/>
@njit()
def getNeighbourIndices(index, size):
    # get dim and setup returnable - note numba doesn't like appending to lists, so assigned full array to assign values instead
    dim = len(index)
    indices = np.empty((2*dim, dim), dtype=np.int64)

    # add +1 adjacent tuples to first half of indices array
    for i in range(0, dim, 1):
        add = index.copy()
        add[i] = (add[i]+1)%size
        indices[i]=add   
        
    # add -1 adjacent tuples to second half of indices array
    for i in range(dim, 2*dim, 1):
        sub = index.copy()
        sub[i-dim] = (sub[i-dim]-1)%size
        indices[i]=sub

    #print("index :", index, " adjacents: ", indices)
    # return
    return indices


# gets value in multidimensional array given 1D array of index positions - note works by flattening array, and converting multidimensional indices to 1D index; done like this to satisfy numba
@njit()
def getArrayVal(array, indices, size): # array is N dimensional np.array, indices is 1D array with length N
    dimensions = len(indices)
    flatArray = array.copy().flatten()
    flattenedIndex = 0
    
    for i in range(0, dimensions, 1):
        flattenedIndex = flattenedIndex + indices[i] * ((size) ** (i))
    
    return flatArray[flattenedIndex]


@njit()
def getNeighbours(config, size, index):
    # get indices for neighbours, and setup returnable
    indices = getNeighbourIndices(index, size).T
    neighbourValues = np.empty(indices.shape[1])
    
    # iterate through all indices and assign to returnable
    for i in range(0, indices.shape[1]):
        currentIndex = np.flip(indices[:,i])#.tolist()
        #print(currentIndex)
        neighbourValues[i] = getArrayVal(config, currentIndex, size)
        
    # return
    return neighbourValues
    



# ==========================================================================================================================================
# parent config class
# ==========================================================================================================================================


# config object - do for each dimension - do as inheritance structure
class config():
    # ------------------------------------------------
	# constructor - </method verified/>
    # ------------------------------------------------
    def __init__(self, size):
		# variables:
        self.size = size
			# simulationParameters (i.e, dimensionality, size L)
            # currentConfiguration
			# observableArraysOverSimulation
	
    # ------------------------------------------------
    # utility methods
    # ------------------------------------------------
    
	# saveToFile - saves self as pickle file at directory/filename - </method verified/>
    def saveToFile(self, directory, fileName):
        # set current directory
        curDir = os.getcwd()
        os.chdir(directory)
        
        # write directory
        with open(fileName, 'ab') as dbfile:
            pickle.dump(self, dbfile)
            
        # rest directory
        os.chdir(curDir)
        
	# readFromFile - reads and returns config from pickle from directory/filename - </method verified/>
    def readFromFile(directory, fileName):
        # set current directory
        curDir = os.getcwd()
        os.chdir(directory)
        
        # open file
        with open(fileName, 'rb') as dbfile:
            try:
                while True:
                    config = pickle.load(dbfile)
                    
            except EOFError:
                pass
        
        # reset directory
        os.chdir(curDir)
        
        # return
        return config
    
    # getConfig - config class should save ALL states in time evolution - getConfig returns config at given time step t
    def getConfig(self, t, returnCopy=False):
        pass
    
    # appendConfig - appends a new configuration to the state
    def appendConfig(self, config):
        pass
    # ------------------------------------------------
    # simulation methods
    # ------------------------------------------------
    
	# simulationStep
    def mcMove(config, beta, size):
        pass

	# runSimulation
    
    # ------------------------------------------------
    # Accessor methods
    # ------------------------------------------------

	# observable method - calculate current energy - copy/pasted from provided code - </method verified/>
    def calcEnergy(self):
        pass
    
    # observable method - calculate current magnetisation - copy/pasted from provided code - </method verified/>
    def calcMag(self, t=-1):
        '''Magnetization of a given configuration'''
        mag = np.sum(self.getConfig(t))
        return mag

  
    

# testing
    
testDirectory = "/Users/nayandusoruth/Desktop/Y3physics/Lab module/labProject/testFolder"
#testConfig = config()

#testConfig.saveToFile(testDirectory, "testName")

#readConfig = config.readFromFile(testDirectory, "testName")

#print(testConfig.temp2)
#print(readConfig.temp2)
# ==========================================================================================================================================
# child class config 2D
# ==========================================================================================================================================


class config2D(config):
    # ------------------------------------------------
	# constructor - </method verified/>
    # ------------------------------------------------
    def __init__(self, size):
        super().__init__(size) 
        # main state
        initialState = 2*np.random.randint(2, size=(size,size))-1 # the main ising model state 
        self.state =  initialState.reshape((initialState.shape[0], initialState.shape[1], 1))
        #print("shape ", self.state.shape, " - ", self.getConfig(0))
        # observables
        self.energies = np.array([])
        self.magentisation = np.array([])
        
    
    # ------------------------------------------------
    # utility methods
    # ------------------------------------------------    
    
    # getConfig - config class should save ALL states in time evolution - getConfig returns config at given time step t - </method verified/>
    def getConfig(self, t, returnCopy=False):
        if(returnCopy):
            return copy.deepcopy(self.state[:,:,t])
        else:
            return self.state[:,:,t]
    
    # appendConfig - appends a new configuration to the state - </method verified/>
    def appendConfig(self, config):
        self.state = np.dstack((self.state, config)) 
    
    # ------------------------------------------------
    # simulation methods
    # ------------------------------------------------
    
   
    
	# simulationStep - main MC move step - copy/pasted from provided code - need to refactor for legibility asp - </method verified/>
    # TODO: refactor for legibility and efficiency
    @njit()
    def mcMove(config, beta, size):        
        for i in range(size): # TODO: rework loop
            for j in range(size):
                    a = np.random.randint(0, size)
                    b = np.random.randint(0, size) # TODO: rework random cell chosing
                    
                    s =  config[a, b]
                    index = np.array([a,b]) # TODO: replace with more general system
                    neighbours = getNeighbours(config, size, index)
                    nb = np.sum(neighbours)
                    cost = 2*s*nb
                    if cost < 0:
                        s *= -1
                    elif rand() < np.exp(-cost*beta):
                        s *= -1
                    config[a, b] = s
        return config
        

	# runSimulation
    def runSimulation(self, steps, beta, plotConfigs=False, saveObservables=False, printProgress=False):
        
        for i in range(0, steps,1):
            #if(printProgress):
            #    print("simulation progress: ", i, " / ", steps)
            
            # simulate step
            config = self.getConfig(-1, returnCopy=True)
            #print(config, " - ", self.size)
            newConfig = config2D.mcMove(config, beta, self.size)
            self.appendConfig(newConfig)
            
            #if(plotConfigs):
            #    self.plotConfig(t=-1)
            
            # compute and save observables
            #if(saveObservables):
            #    self.energies = np.append(self.energies, self.calcEnergy())
            #    self.magentisation = np.append(self.magentisation, self.calcMag())
    
    # ------------------------------------------------
    # Accessor methods
    # inc observables
    # ------------------------------------------------

	# observable method - calculate current energy - copy/pasted from provided code - </method verified/>
    # TODO: refactor for legibility
    def calcEnergy(self, t=-1):
        '''Energy of a given configuration'''
        energy = 0
        config = self.getConfig(t)
        for i in range(self.size):
            for j in range(self.size):
                S = config[i,j]
                index = np.array([i,j]) # TODO: replace with more general system
                neighbours = getNeighbours(config, self.size, index)
                nb = np.sum(neighbours)
                energy += -nb*S
        return energy/4.  
    
    # plot the state as a 2D image - </method verified/>
    # TODO: make better
    def plotConfig(self, t=-1):
        plt.imshow(self.getConfig(t), cmap='Greys')
        plt.show()
        
    # TODO: compute more observables of interest
        
        

# testing
T = 300
k_b = 1.38*10**(-23)
#beta = 1 / (k_b * T)

testConfig2D = config2D(100)


#newConfig = np.array([[2,2],[2,2]])
#newConfig2 = np.array([[3,3],[3,3]])
#testConfig2D.appendConfig(newConfig)
#testConfig2D.appendConfig(newConfig2)


#print(testConfig2D.state)
#testConfig2D.plotConfig()
startTime = time.time()
testConfig2D.runSimulation(500, beta(1.5), plotConfigs=False, printProgress=False)
endTime = time.time()

timeDiff = startTime - endTime
print("timeElapsed: ", timeDiff)


testConfig2D.saveToFile(testDirectory, "testName")

readConfig = config.readFromFile(testDirectory, "testName")

print("here")
t = -1
testConfig2D.plotConfig(t=t)
readConfig.plotConfig(t=t)

print(testConfig2D.calcEnergy(t=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

"""
samples = 20
testTemps= np.linspace(.53, 3.28, samples)

testBetas = beta(testTemps)
#print(testBetas)

size = 16
testConfigs = np.array([config2D(size)])
for i in range(0, samples-1):
    testConfigs = np.append(testConfigs, config2D(size))

Energies = np.empty(samples)
magnetisations = np.empty(samples)
for i in range(0, len(testConfigs), 1):
    testConfigs[i].plotConfig()
    testConfigs[i].runSimulation(2000, testBetas[i])
    Energies[i] = np.mean(testConfigs[i].energies[-1000:-1])
    #print(testConfigs[i].energies[-1000:-1])
    magnetisations[i] = np.mean(testConfigs[i].magentisation[-1000:-1])
    testConfigs[i].plotConfig()
    
#print(Energies)
plotE = plotter.plotter()
plotE.gridline()
plotE.scatter(testTemps, Energies, marker = 'x', color='k', label="")
plotE.display()

plotM = plotter.plotter()
plotM.gridline()
plotM.scatter(testTemps, magnetisations, marker = 'x', color='k', label="")
plotM.display()
#print(testConfig2D.state)
#testConfig2D.plotConfig()
#testConfig2D.mcMove(1)
#print(testConfig2D.state)
#print(testConfig2D.calcMag())
#print(testConfig2D.calcEnergy())
#testConfig2D.saveToFile(testDirectory, "testName2")
"""
#readConfig2D = config.readFromFile(testDirectory, "testName2")

#print(testConfig2D.temp3)

#
#print("----")

#testConfig.mcMove(1)
#print(readConfig2D.temp3)
#print(type(readConfig2D))

