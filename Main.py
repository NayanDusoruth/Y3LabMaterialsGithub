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
import scipy

from numba import jit
from numba import njit
import numba as nb

import copy
import time
import json

#import NayanGeneralUtils.plotting as plotter

# ==========================================================================================================================================
# code utilities
# ==========================================================================================================================================

def strLinesConcatenate(strList, current=""):
    """utility function - concatenates and returns strList with line breaks""" # - </Verified/>
    if(len(strList)==1):
        return '\n'.join([current, str(strList[0])])
    else:
        return '\n'.join([current, strLinesConcatenate(strList[1:],current=str(strList[0]))])
    

# utility function - saves dataframe as csv to filepath/filename.csv - </verified/> - code borrowed from nayanGeneralUtils
def saveCSV(filepath, filename, dataFrame):
    """File handling function - saves dataframe as csv to filepath/filename.csv"""
    curDir = os.getcwd() # get the current directory
    os.chdir(filepath)
    path = filepath + "/" + filename + ".csv"
    dataFrame.to_csv(path)
    os.chdir(curDir)


# utility function - saves python dictionary as a json file  - </verified/> - code borrowed from nayanGeneralUtils
def saveDictAsJson(directory, fileName, dictionary, indent=3):
    """File handling function - saves dictionary to directory with filename as Json with indent=indent"""
    
    # go to desired directory
    curDir = os.getcwd() # get the current directory
    os.chdir(directory)
    
    # save file as json
    with open(fileName+".json", "w") as file:
        json.dump(dictionary, file, indent=indent)
        
    # go back to original directory
    os.chdir(curDir)

# utility function - loads python dictionary from json file - </verified/> - code borrowed from nayanGeneralUtils
def loadDictFromJson(directory, fileName):
    """File handling function - loads dictionary from directory/fileName.json"""
    # go to desired directory
    curDir = os.getcwd() # get the current directory
    os.chdir(directory)
    
    # access json file
    with open(directory + "/" + fileName + '.json') as json_file:
        dictionary = json.load(json_file) 
        
    # go back to original directory
    os.chdir(curDir)
    
    return dictionary

# utility function - saves fig to directory with filename - </verified/> - code borrowed from nayanGeneralUtils
def saveFigure(directory, fileName, fig):
    """File handling function - saves fig to directory with filename"""
    # go to desired directory
    curDir = os.getcwd() # get the current directory
    os.chdir(directory)

    # save figure to folder
    fig.savefig(fileName, dpi=300, bbox_inches='tight')
    
    # go back to original directory
    os.chdir(curDir)
    
    
# create new folder - </verified/> -  code borrowed from nayanGeneralUtils
def createFolder(directory, folderName):
    """File handling function - creates new folder at 'directory/folderName'"""
    newPath = str(directory + "/"+folderName)
    if (not os.path.exists(newPath)):
        os.makedirs(newPath)
    return newPath

# utility function - code borrowed from https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
def weighted_avg_and_std(values, weights, absolute=False):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    if(absolute):
        return abs(average), abs(np.sqrt(variance))
    return average, np.sqrt(variance)


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
        
        self.beta = 0
        
        self.adjacencyIndices = getAdjacencies(self.dimension)
        
        self.basis = getPowers(self.dimension , self.size) # basis for Nd array a
        
        initialState = 2*np.random.randint(2, size=tuple(self.sizes))-1 # the main ising model state 
        self.state =  [initialState]
                
        # observables
        self.energies = np.array([])
        self.energiesSquared = np.array([])
        self.magentisation = np.array([])
        self.magentisationSquared = np.array([])
        self.magentisation4 = np.array([])
        self.partitions = np.array([])
        self.entropies = np.array([])
        self.helmholtzEnergies = np.array([])
       

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
            flatArray = config.flatten() # ravel
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
            elif rand() < np.exp(-cost*beta): #TODO: consider precomputing this
                s *= -1
                
            #config = editArrayVal(config, randomIndex, sizesTuple, basis, s) #edit value
            flattenedIndex = np.sum(np.flip(randomIndex) * basis)
            flatArray[flattenedIndex] = s
            config = flatArray.reshape(sizesTuple)
            
            
        return config

    @njit()
    def mcMove2D(config, beta, size):
        #print("uses this")
        curConfig = config.copy()
        for i in range(0, size,1):
            for j in range(0, size,1):
                a = np.random.randint(0, size)
                b = np.random.randint(0, size)
                s =  curConfig[a, b]
                nb = curConfig[(a+1)%size,b] + curConfig[a,(b+1)%size] + curConfig[(a-1)%size,b] + curConfig[a,(b-1)%size]
                cost = 2*s*nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost*beta):
                    s *= -1
                curConfig[a, b] = s
        return curConfig
    
    @njit()
    def mcMove3D(config, beta, size):
        #print("uses this")
        curConfig = config.copy()
        for i in range(0, size,1):
            for j in range(0, size,1):
                for k in range(0, size,1):
                    a = np.random.randint(0, size)
                    b = np.random.randint(0, size)
                    c = np.random.randint(0, size)
                    s =  curConfig[a, b, c]
                    nb = curConfig[(a+1)%size,b, c] + curConfig[a,(b+1)%size, c] + curConfig[(a-1)%size,b, c] + curConfig[a,(b-1)%size, c] + curConfig[a,b, (c-1)%size] + curConfig[a,b, (c+1)%size]
                    cost = 2*s*nb
                    if cost < 0:
                        s *= -1
                    elif rand() < np.exp(-cost*beta):
                        s *= -1
                        curConfig[a, b] = s
        return curConfig
    
    @njit()
    def mcMove4D(config, beta, size):
        #print("uses this")
        curConfig = config.copy()
        for i in range(0, size,1):
            for j in range(0, size,1):
                for k in range(0, size,1):
                    for z in range(0, size,1):
                        a = np.random.randint(0, size)
                        b = np.random.randint(0, size)
                        c = np.random.randint(0, size)
                        d = np.random.randint(0, size)
                        s =  curConfig[a, b, c, d]
                        nb = curConfig[(a+1)%size,b, c, d] + curConfig[a,(b+1)%size, c, d] + curConfig[(a-1)%size,b, c, d] + curConfig[a,(b-1)%size, c, d] + curConfig[a,b, (c-1)%size, d] + curConfig[a,b, (c+1)%size, d]+ curConfig[a,b, c, (d+1)%size]+ curConfig[a,b, c, (d-1)%size]
                        cost = 2*s*nb
                        if cost < 0:
                            s *= -1
                        elif rand() < np.exp(-cost*beta):
                            s *= -1
                            curConfig[a, b] = s
        return curConfig

	# runSimulation
    def runSimulation(self, steps, beta, saveConfigs = True, plotConfigs=False, saveObservables=True, printProgress=False, saveFigs=False, saveDirectory=""):
        
        curConfig = self.getConfig(-1, returnCopy=True)
        configSize = self.size ** self.dimension
        self.beta = beta
        for i in range(0, steps,1):
            if(printProgress):
                print("simulation progress: ", i, " / ", steps)
            
            
            # simulate step
            if(self.dimension == 2):
                curConfig = config.mcMove2D(curConfig, beta, self.size)
            elif(self.dimension == 3):
                curConfig = config.mcMove3D(curConfig, beta, self.size)
            elif(self.dimension == 4):
                curConfig = config.mcMove4D(curConfig, beta, self.size)
            else:
                curConfig = config.mcMove(curConfig, beta, self.size, self.dimension, self.basis, self.sizes, self.sizesTuple, configSize, self.adjacencyIndices)

            if(saveConfigs): 
               self.appendConfig(curConfig)
            else:
                self.state[-1] = curConfig

            if(saveFigs):
                saveFigure(saveDirectory, str(i), self.plotConfig( t=-1, returnFig=True))

            if(plotConfigs):
                self.plotConfig(t=-1)
            
            # compute and save observables
            if(saveObservables):
                self.energies = np.append(self.energies, self.calcEnergy())
                self.energiesSquared = np.append(self.energiesSquared, self.calcEnergy()**2)
                self.magentisation = np.append(self.magentisation, self.calcMag())
                self.magentisationSquared = np.append(self.magentisationSquared, self.calcMag()**2)
                self.magentisation4 = np.append(self.magentisation4, self.calcMag()**4)
                self.partitions = np.append(self.partitions, self.calcPartitionFunction(beta))
                self.entropies = np.append(self.entropies, self.calcEntropy(beta))
                self.helmholtzEnergies = np.append(self.helmholtzEnergies, self.calcHelmholtz(beta))
                #print(self.calcMag())
        
        #if(not saveConfigs):
        #    self.appendConfig(curConfig) # append last config to state
        
    
    
    # ------------------------------------------------
    # Observable methods
    # ------------------------------------------------
    
    # recompute observables in a range - breaks if the configs don't exist
    def recomputeObservables(self):
        beta = self.beta
        
        for i in range(0, len(self.state),1):
            self.energies[i] = self.calcEnergy(t=i)
            self.energiesSquared[i] =  self.calcEnergy(t=i)**2
            self.magentisation[i] =self.calcMag(t=i)
            self.magentisationSquared[i] = self.calcMag(t=i)**2
            self.magentisation4[i] = self.calcMag(t=i)**4
            self.partitions[i] = self.calcPartitionFunction(beta, t=i)
            self.entropies[i] = self.calcEntropy(beta, t=i)
            self.helmholtzEnergies[i] = self.calcHelmholtz(beta, t=i)
            #print(i)
    
    # recompute mag^4 specifically
    def recomputeMag4(self):
        beta = self.beta
        mag = self.magentisation.copy()
        self.magentisation4 = mag**4

	# observable method - calculate current energy - copy/pasted from provided code - </method verified/>
    # TODO: refactor for legibility
    #TODO: ADJUST FOR MULTIPLE DIMENSIONS
    def calcEnergy(self, t=-1):
        '''Energy of a given configuration'''
        energy = 0
        config = self.getConfig(t)
        flatArray = config.copy().flatten()
        totalSize = self.size ** self.dimension
        
        
        for cell in range(0, totalSize, 1): 
            S = flatArray[cell]

            nb = 0
            # get neighbour indices using adjacencyIndices and use that to add neighbour spin values to nb
            for i in range(0, 2*self.dimension, 1):
                currentIndex = (self.adjacencyIndices[i] + cell)%self.size
                flattenedIndex = np.sum(np.flip(currentIndex) * self.basis)
                nb +=flatArray[flattenedIndex]
            
            energy += -nb*S
        return energy/len(self.basis) 
    
    # observable method - calculate current magnetisation - copy/pasted from provided code - </method verified/>
    def calcMag(self, t=-1):
        '''Magnetization of a given configuration'''
        mag = np.sum(self.getConfig(t))
        return mag
    
    
    def calcPartitionFunction(self, beta, t=-1):
        partition = 0
        config = self.getConfig(t)
        flatArray = config.copy().flatten()
        totalSize = self.size ** self.dimension
        
        for cell in range(0, totalSize, 1): 
            S = flatArray[cell]

            nb = 0
            # get neighbour indices using adjacencyIndices and use that to add neighbour spin values to nb
            for i in range(0, 2*self.dimension, 1):
                currentIndex = (self.adjacencyIndices[i] + cell)%self.size
                flattenedIndex = np.sum(np.flip(currentIndex) * self.basis)
                nb +=flatArray[flattenedIndex]
            
            energy = -nb*S
            
            exponent = np.exp(beta * -energy)
            partition += exponent
    
        return partition/len(self.basis)  
    
    def calcEntropy(self, beta, t=-1, k_b = 1):
        return k_b * (np.log(self.partitions[t]) + self.energies[t] * beta)
    
    def calcHelmholtz(self,beta, t=-1):
        return -beta * np.log(self.partitions[t])
      
    # note, average observables should be weighted by boltzman distribution as \frac{1}{Z}\sum{e^{-H\beta} \cdot A_{i}} for observable A_{i} - this method computes the weights for weighted averaging
    def boltzmanWeights(self, startIndex, finalIndex):
        partitions = self.partitions[startIndex: finalIndex]
        energies = self.energies[startIndex: finalIndex]
        
        exp = -energies * self.beta
        exp = exp.astype(np.float128)
        exponents = np.exp(exp)
        return exponents / partitions
    
    # ------------------------------------------------
    # accessor methods
    # more or less average out observables for certain number of "recent" configs
    # ------------------------------------------------
    
    def averageEnergy(self, startIndex, finalIndex, perSpin=False, absolute=False):
        dataRange = self.energies[startIndex: finalIndex]
        weights = np.full(len(dataRange),1)#self.boltzmanWeights(startIndex, finalIndex)
        if(perSpin):
            numCells = self.size ** self.dimension
            ave, dev = weighted_avg_and_std(dataRange, weights, absolute=absolute)
            ave = ave / numCells
            dev = dev / numCells
            
            return ave, dev
        return weighted_avg_and_std(dataRange, weights, absolute=absolute)
        
    
    def averageEnergySquared(self, startIndex, finalIndex,perSpin=False, absolute=False):
        dataRange = self.energiesSquared[startIndex: finalIndex]
        weights = np.full(len(dataRange),1)#self.boltzmanWeights(startIndex, finalIndex)
        if(perSpin):
            numCells = self.size ** self.dimension
            ave, dev = weighted_avg_and_std(dataRange, weights, absolute=absolute)
            ave = ave / numCells
            dev = dev / numCells
            
            return ave, dev
        return weighted_avg_and_std(dataRange, weights, absolute=absolute)
        
    def averageMag(self, startIndex, finalIndex, perSpin=True, absolute=False):
        dataRange = self.magentisation[startIndex: finalIndex]
        weights = np.full(len(dataRange),1)#self.boltzmanWeights(startIndex, finalIndex)
        if(perSpin):
            numCells = self.size ** self.dimension
            #print(numCells)
            ave, dev = weighted_avg_and_std(dataRange, weights, absolute=absolute)
            ave = ave / numCells
            dev = dev / numCells
            
            return ave, dev
            
        
        return weighted_avg_and_std(dataRange, weights, absolute=absolute)
    
    def averageMagSquared(self, startIndex, finalIndex,perSpin=True, absolute=False):
        dataRange = self.magentisation[startIndex: finalIndex]**2
        weights = np.full(len(dataRange),1)#self.boltzmanWeights(startIndex, finalIndex)
        if(perSpin):
            numCells = (self.size ** self.dimension)**2
            ave, dev = weighted_avg_and_std(dataRange, weights, absolute=absolute)
            ave = ave / numCells
            dev = dev / numCells
            
            return ave, dev
        return weighted_avg_and_std(dataRange, weights, absolute=absolute)
    
    def averageMag4(self, startIndex, finalIndex,perSpin=True, absolute=False):
        dataRange = self.magentisation4[startIndex: finalIndex]
        weights = np.full(len(dataRange),1)#self.boltzmanWeights(startIndex, finalIndex)
        if(perSpin):
            numCells = self.size ** self.dimension
            ave, dev = weighted_avg_and_std(dataRange, weights, absolute=absolute)
            ave = ave / numCells**4
            dev = dev / numCells**4
            
            return ave, dev
        return weighted_avg_and_std(dataRange, weights, absolute=absolute)
    
    def averagePartition(self, startIndex, finalIndex,perSpin=False, absolute=False):
        dataRange = self.partitions[startIndex: finalIndex]
        weights = np.full(len(dataRange),1)#self.boltzmanWeights(startIndex, finalIndex)
        if(perSpin):
            numCells = self.size ** self.dimension
            ave, dev = weighted_avg_and_std(dataRange, weights, absolute=absolute)
            ave = ave / numCells
            dev = dev / numCells
            
            return ave, dev
        return weighted_avg_and_std(dataRange, weights, absolute=absolute)
    
    def averageEntropy(self, startIndex, finalIndex,perSpin=False, absolute=False):
        dataRange = self.entropies[startIndex: finalIndex]
        weights = np.full(len(dataRange),1)#self.boltzmanWeights(startIndex, finalIndex)
        if(perSpin):
            numCells = self.size ** self.dimension
            ave, dev = weighted_avg_and_std(dataRange, weights, absolute=absolute)
            ave = ave / numCells
            dev = dev / numCells
            
            return ave, dev
        return weighted_avg_and_std(dataRange, weights, absolute=absolute)
    
    def averageHelmholtz(self, startIndex, finalIndex,perSpin=False, absolute=False):
        dataRange = self.helmholtzEnergies[startIndex: finalIndex]
        weights = np.full(len(dataRange),1)#self.boltzmanWeights(startIndex, finalIndex)
        if(perSpin):
            numCells = self.size ** self.dimension
            ave, dev = weighted_avg_and_std(dataRange, weights, absolute=absolute)
            ave = ave / numCells
            dev = dev / numCells
            
            return ave, dev
        return weighted_avg_and_std(dataRange, weights, absolute=absolute)
    
    # TODO: fix
    def susceptibility(self, startIndex, finalIndex, perSpin=False, absolute=False):
        magnetisations = self.magentisation[startIndex: finalIndex: 20]
        magnetisationsSquared = magnetisations * magnetisations
        
        # regular average
        mag, magDev = np.mean(magnetisations), np.std(magnetisationsSquared)#self.averageMag(startIndex, finalIndex)
        mag2, magDev2 = np.mean(magnetisationsSquared), np.std(magnetisationsSquared)#self.averageEnergySquared(startIndex, finalIndex)
        
        # weighted average
        #mag, magDev = self.averageMag(startIndex, finalIndex,perSpin=perSpin, absolute=absolute)
        #mag2, magDev2 = self.averageMagSquared(startIndex, finalIndex,perSpin=perSpin, absolute=absolute)
        
        numSpins = self.size ** self.dimension
        
        susceptibility = self.beta * (mag**2 - mag2) /numSpins
        susceptibilityErr = 0#np.sqrt((self.beta / numSpins)**2 * (4 * mag**2 * magDev**2 + magDev2**2))
        
        if(absolute):
            return abs(susceptibility), abs(susceptibilityErr)
        return susceptibility, susceptibilityErr
    
    #TODO: fix
    def heatCapacity(self, startIndex, finalIndex, perSpin=False, absolute=False):
        energies = self.energies[startIndex: finalIndex]
        energiesSquared = energies ** 2
        # regular average
        #E, EDev = np.mean(energies), np.std(energies)#self.averageEnergy(startIndex, finalIndex)
        #E2, EDev2 =  np.mean(energiesSquared), np.std(energiesSquared)#self.averageEnergySquared(startIndex, finalIndex)
        
        # weighted average
        E, EDev = self.averageEnergy(startIndex, finalIndex,perSpin=perSpin, absolute=absolute)
        E2, EDev2 =  self.averageEnergySquared(startIndex, finalIndex,perSpin=perSpin, absolute=absolute)
        
        numSpins = self.size ** self.dimension
        
        heatCapacity = self.beta**2 * (E**2 - E2) /numSpins
        heatCapacityErr = 0#np.sqrt((self.beta**2 / numSpins)**2 * (4 * E**2 * EDev**2 + EDev2**2))
        
        if(absolute):
            return abs(heatCapacity), abs(heatCapacityErr)
        return heatCapacity, heatCapacityErr
    
    def binderCumulant(self, startIndex, finalIndex, absolute=False):
        s4, s4Dev = self.averageMag4(startIndex, finalIndex,perSpin=True, absolute=False)
        s2, s2Dev = self.averageMagSquared(startIndex, finalIndex,perSpin=True, absolute=False)
        cumulant = 1 - s4 / (3 * s2)
        cumulantErr = 0
        
        
        return cumulant, cumulantErr
    
    # ------------------------------------------------
    # plotting methods
    # ------------------------------------------------
    
    # plot the state as a 2D image - </method verified/>
    # TODO: make better
    #TODO: lock out for D != 1, 2 = otherwise will break
    def plotConfig(self, t=-1, returnFig=False, title=""):
        
        if(self.dimension == 1 or self.dimension == 2):
            fig = self.plotConfig1_2D(t, title=title)
        elif(self.dimension == 3):
            fig = self.plotConfig3D(t)
        
        if(returnFig):
            return fig
       
        
      # graphing method - basic 1-2D plotting method 
    def plotConfig1_2D(self, t, title=""):
        fig, ax = plt.subplots()
        ax.imshow(self.getConfig(t), cmap='Greys')
        
        # quick debugging w/plot title
        ax.set_title(title)
            
        plt.show()
        
        return fig
        
    def plotConfig3D(self, t):
        boolArray = self.getConfig(t) >0
        
        #boolArray = np.array([[[True, False, True], [False, False, False], [True, False, True]], [[False, False, False], [False, False, False], [False, False, False]], [[True, False, True], [False, False, False], [True, False, True]]])
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal')

        ax.voxels(boolArray, edgecolor="k")

        plt.show()
        return fig

# testing
    
#testDirectory = "/Users/nayandusoruth/Desktop/Y3physics/LabModule/labProject/testFolder"
#testConfig = config(size=20, dimension=2)
#testConfig.runSimulation(2000, 1.2, saveConfigs = False, plotConfigs=True, saveObservables=True, printProgress=False, saveFigs=False, saveDirectory="")
#testConfig.saveToFile(testDirectory, "testName")

#readConfig = config.readFromFile(testDirectory, "testName")

#print(testConfig.temp2)
#print(readConfig.temp2)
        



