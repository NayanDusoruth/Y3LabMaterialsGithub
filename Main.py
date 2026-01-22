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

import NayanGeneralUtils.plotting as plotter

# ==========================================================================================================================================
# code utilities
# ==========================================================================================================================================

def beta(T):
    k_b = 1.38 * 10**(-23)
    return 1 / (k_b * T)


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
       
    # ------------------------------------------------
    # simulation methods
    # ------------------------------------------------
    
	# simulationStep
    def mcMove(self, beta):
        pass

	# runSimulation
    
    # ------------------------------------------------
    # Accessor methods
    # ------------------------------------------------

	# observable method - calculate current energy - copy/pasted from provided code - </method verified/>
    def calcEnergy(self):
        pass
    
    # observable method - calculate current magnetisation - copy/pasted from provided code - </method verified/>
    def calcMag(self):
        '''Magnetization of a given configuration'''
        mag = np.sum(self.state)
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
        self.state =  2*np.random.randint(2, size=(size,size))-1 # the main ising model state 
        
        # observables
        self.energies = np.array([])
        self.magentisation = np.array([])
    # ------------------------------------------------
    # simulation methods
    # ------------------------------------------------
    
	# simulationStep - main MC move step - copy/pasted from provided code - need to refactor for legibility asp - </method verified/>
    # TODO: refactor for legibility
    def mcMove(self, beta):
        for i in range(self.size):
            for j in range(self.size):
                    a = np.random.randint(0, self.size)
                    b = np.random.randint(0, self.size)
                    s =  self.state[a, b]
                    nb = self.state[(a+1)%self.size,b] + self.state[a,(b+1)%self.size] + self.state[(a-1)%self.size,b] + self.state[a,(b-1)%self.size]
                    cost = 2*s*nb
                    if cost < 0:
                        s *= -1
                    elif rand() < np.exp(-cost*beta):
                        s *= -1
                    self.state[a, b] = s
        

	# runSimulation
    def runSimulation(self, steps, beta):
        
        for i in range(0, steps,1):
            #print(i)
            # simulate step
            self.mcMove(beta)
            
            # compute and save observables
            self.energies = np.append(self.energies, self.calcEnergy())
            self.magentisation = np.append(self.magentisation, self.calcMag())
    
    # ------------------------------------------------
    # Accessor methods
    # inc observables
    # ------------------------------------------------

	# observable method - calculate current energy - copy/pasted from provided code - </method verified/>
    # TODO: refactor for legibility
    def calcEnergy(self):
        '''Energy of a given configuration'''
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                S = self.state[i,j]
                nb = self.state[(i+1)%self.size, j] + self.state[i,(j+1)%self.size] + self.state[(i-1)%self.size, j] + self.state[i,(j-1)%self.size]
                energy += -nb*S
        return energy/4.  
    
    # plot the state as a 2D image - </method verified/>
    # TODO: make better
    def plotConfig(self):
        
        plt.imshow(self.state, cmap='Greys')
        plt.show()
        
        
        

# testing
T = 300
k_b = 1.38*10**(-23)
#beta = 1 / (k_b * T)

testConfig2D = config2D(100)
#print(testConfig2D.state)
#testConfig2D.plotConfig()
#testConfig2D.runSimulation(100, beta)
samples = 10
testTemps= np.linspace(1, 300, samples)

testBetas = beta(testTemps)
print(testBetas)

size = 100
testConfigs = np.full(samples, config2D(size))

Energies = np.empty(samples)
magnetisations = np.empty(samples)
for i in range(0, len(testConfigs), 1):
    testConfigs[i].plotConfig()
    testConfigs[i].runSimulation(100, testBetas[i])
    Energies[i] = testConfigs[i].calcEnergy()
    magnetisations[i] = testConfigs[i].calcMag()
    testConfigs[i].plotConfig()
    
print(Energies)
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

#readConfig2D = config.readFromFile(testDirectory, "testName2")

#print(testConfig2D.temp3)

#
#print("----")

#testConfig.mcMove(1)
#print(readConfig2D.temp3)
#print(type(readConfig2D))

