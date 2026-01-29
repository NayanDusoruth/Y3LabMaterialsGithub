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

#import NayanGeneralUtils.plotting as plotter

# ==========================================================================================================================================
# code utilities
# ==========================================================================================================================================

def beta(T):
    k_b = 1 # 1.38 * 10**(-23)
    return 1 / (k_b * T)

# utility method - given 1D array of Nd array sizes; returns array to power of index in array - is used for coord flattening reasons - </function verified/>
@njit()
def getPowers(dim, size):
    sizes = np.full(dim, size)
    dimensions = np.arange(0, dim, 1)
    return np.power(sizes, dimensions)

# computes the array "adjacencies" given a dimension
@njit()
def getAdjacencies(dimension):
    indices = np.empty((2*dimension, dimension), dtype=np.int64)
    
    # add +1 adjacent tuples to first half of indices array
    for i in range(0, dimension, 1):
        add =  np.empty(dimension, dtype=np.int64)
        add[i] = add[i]+1
        indices[i]=add   
        
    # add -1 adjacent tuples to second half of indices array
    for i in range(dimension, 2*dimension, 1):
        sub = np.empty(dimension, dtype=np.int64)
        sub[i-dimension] = sub[i-dimension]-1
        indices[i]=sub

    return indices

# computes the nheigbour indices for an index given the adjacencyIndices, size and dimension
@njit()
def getNeighbourIndices(index, adjacencyIndices, size, dimension):

    indices =  adjacencyIndices.copy()

    for i in range(0, 2*dimension, 1):
        indices[i] = (indices[i] + index)%size

    # return
    return indices


# gets value in multidimensional array given 1D array of index positions - note works by flattening array, and converting multidimensional indices to 1D index; done like this to satisfy numba
@njit()
def getArrayVal(array, indices, size, basis): # array is N dimensional np.array, indices is 1D array with length N
    dimensions = len(indices)
    flatArray = array.copy().flatten()
    
    flattenedIndex = np.sum(np.flip(indices) * basis)
    
    return flatArray[flattenedIndex]

# edits values in multidimensional array given 1D array of index positions - </function verified/>
@njit()
def editArrayVal(array, indices, sizesTuple, basis, newVal):
    dimensions = len(indices)
    flatArray = array.copy().flatten()

    flattenedIndex = np.sum(np.flip(indices) * basis)
    
    flatArray[flattenedIndex] = newVal

    return flatArray.reshape(sizesTuple) # this may stop numba functionality


@njit()
def getNeighbours(configB, size, index, basis, adjacencyIndices, dimension):
    # get indices for neighbours, and setup returnable
    indices = getNeighbourIndices(index,adjacencyIndices, size, dimension).T
    neighbourValues = np.empty(indices.shape[1])
    flatArray = configB.copy().flatten()
    # iterate through all indices and assign to returnable
    for i in range(0, indices.shape[1]):
        
        currentIndex = indices[:,i]#.tolist()
        flattenedIndex = np.sum(np.flip(currentIndex) * basis)
        neighbourValues[i] = flatArray[flattenedIndex]

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
    def __init__(self, size, dimension):
		# variables:
        self.size = size
        self.dimension = dimension
        self.sizes = np.full(dimension, size)
        self.sizesTuple = tuple(self.sizes)
        
        self.adjacencyIndices = getAdjacencies(self.dimension)
        
        self.basis = getPowers(self.dimension , self.size) # basis for Nd array a
        
        initialState = 2*np.random.randint(2, size=tuple(self.sizes))-1 # the main ising model state 
        self.state =  [initialState]
                
        # observables
        self.energies = np.array([])
        self.magentisation = np.array([])

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
    
    
    # getConfig - config class should save ALL states in time evolution - getConfig returns config at given time step t - </method verified/>
    def getConfig(self, t, returnCopy=True):
        stateSlice = self.state[t]
        if(returnCopy):
            return copy.deepcopy(stateSlice)
        else:
            return stateSlice
    
    
    # appendConfig - appends a new configuration to the state - </method verified/>
    def appendConfig(self, config):
        self.state.append(config)
        
        
    # ------------------------------------------------
    # simulation methods
    # ------------------------------------------------
    
	# simulationStep - main MC move step - copy/pasted from provided code - need to refactor for legibility asp - </method verified/>
    @njit()
    def mcMove(config, beta, size, dimension, basis, sizes, sizesTuple, totalSize, adjacencyIndices):    
        for cell in range(0, totalSize, 1): 
            randomIndex = np.random.randint(0,size, size=dimension) # get random cell position

            # get value of config at randomIndex - note code written like this to please numba
            flatArray = config.copy().flatten()
            flattenedIndex = np.sum(np.flip(randomIndex) * basis)
            s = flatArray[flattenedIndex]
            
    
            nb = 0
            # get neighbour indices using adjacencyIndices and use that to add neighbour spin values to nb
            for i in range(0, 2*dimension, 1):
                currentIndex = (adjacencyIndices[i] + randomIndex)%size
                flattenedIndex = np.sum(np.flip(currentIndex) * basis)
                nb +=flatArray[flattenedIndex]
           
            cost = 2*s*nb # compute cost
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost*beta):
                s *= -1
                
            #config = editArrayVal(config, randomIndex, sizesTuple, basis, s) #edit value
            flattenedIndex = np.sum(np.flip(randomIndex) * basis)
            flatArray[flattenedIndex] = s
            config = flatArray.reshape(sizesTuple)
            
            
        return config

	# runSimulation
    def runSimulation(self, steps, beta, saveConfigs = True, plotConfigs=False, saveObservables=False, printProgress=False):
        
        curConfig = self.getConfig(-1, returnCopy=True)
        configSize = self.size ** self.dimension
        
        for i in range(0, steps,1):
            if(printProgress):
                print("simulation progress: ", i, " / ", steps)
            
            
            # simulate step
            curConfig = config.mcMove(curConfig, beta, self.size, self.dimension, self.basis, self.sizes, self.sizesTuple, configSize, self.adjacencyIndices)

            if(saveConfigs): 
               self.appendConfig(curConfig)

            
            if(plotConfigs):
                self.plotConfig(t=-1)
            
            # compute and save observables
            if(saveObservables):
                self.energies = np.append(self.energies, self.calcEnergy())
                self.magentisation = np.append(self.magentisation, self.calcMag())
            
        #if(not saveConfigs):
        #    self.appendConfig(curConfig) # append last config to state
        
    
    
    # ------------------------------------------------
    # Accessor methods
    # ------------------------------------------------

	# observable method - calculate current energy - copy/pasted from provided code - </method verified/>
    # TODO: refactor for legibility
    #TODO: ADJUST FOR MULTIPLE DIMENSIONS
    def calcEnergy(self, t=-1):
        '''Energy of a given configuration'''
        energy = 0
        config = self.getConfig(t)
        for i in range(self.size):
            for j in range(self.size):
                S = config[i,j]
                index = np.array([i,j]) # TODO: replace with more general system
                neighbours = getNeighbours(config, self.size, index, self.basis, self.adjacencyIndices, self.dimension)
                nb = np.sum(neighbours)
                energy += -nb*S
        return energy/4.  
    
    # observable method - calculate current magnetisation - copy/pasted from provided code - </method verified/>
    def calcMag(self, t=-1):
        '''Magnetization of a given configuration'''
        mag = np.sum(self.getConfig(t))
        return mag
    
    
    
    # plot the state as a 2D image - </method verified/>
    # TODO: make better
    #TODO: lock out for D != 1, 2 = otherwise will break
    def plotConfig(self, t=-1):
        
        if(self.dimension == 1 or self.dimension == 2):
            self.plotConfig1_2D(t)
        elif(self.dimension == 3):
            self.plotConfig3D(t)
       
        
      # graphing method - basic 1-2D plotting method 
    def plotConfig1_2D(self, t):
        plt.imshow(self.getConfig(t), cmap='Greys')
        plt.show()
        
    def plotConfig3D(self, t):
        boolArray = self.getConfig(t) >0
        
        #boolArray = np.array([[[True, False, True], [False, False, False], [True, False, True]], [[False, False, False], [False, False, False], [False, False, False]], [[True, False, True], [False, False, False], [True, False, True]]])
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal')

        ax.voxels(boolArray, edgecolor="k")

        plt.show()

# testing
    
testDirectory = "/Users/nayandusoruth/Desktop/Y3physics/LabModule/labProject/testFolder"
#testConfig = config()

#testConfig.saveToFile(testDirectory, "testName")

#readConfig = config.readFromFile(testDirectory, "testName")

#print(testConfig.temp2)
#print(readConfig.temp2)
        



