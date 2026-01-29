#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:45:30 2026

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

from Main import *
from clusterResources import *


class dataset():
    # ------------------------------------------------
    # constructor
    # ------------------------------------------------
    # main constructor - </method verified/>
    def __init__(self, constructorType, configList=[], equilibrationSteps = 10, simulationSteps = 10, dimension=2, size=10, betaList=[], folderPath=""):
        self.betas = []
        self.configs = []
        self.dimension = dimension
        self.size = size
        self.equilibrationSteps = equilibrationSteps
        self.simulationSteps = simulationSteps
        
        if(constructorType=="fromList"):
            self.fromList(configList, betaList)
        elif(constructorType=="fromBetas"):
            self.fromBetas(betaList, equilibrationSteps, simulationSteps)
        elif(constructorType=="fromFile"):
            self.fromFile(folderPath)
        
    # constructor from list of configs - </method verified/>
    def fromList(self, configList, betaList):
        self.betas = betaList
        self.configs = configList
        
    # constructor from array of betas - </method verified/>
    def fromBetas(self, betaList, equilibrationSteps, simulationSteps):
        
        self.betas = betaList
        
        for i in range(0, len(betaList), 1):
            curConfig = config(self.size, self.dimension)
            curConfig.runSimulation(equilibrationSteps, betaList[i], saveConfigs=False) # equilibration steps - do not save these
            curConfig.runSimulation(equilibrationSteps, betaList[i], saveConfigs=True) # main steps - save configs
            self.configs.append(curConfig)
        
    # constructor from folder location - </method verified/>
    def fromFile(self, folderPath):
        print("here")
        configList = []
        betaList = []
        
        folderElements = os.listdir(folderPath)
        print(folderElements)
        
        for file in folderElements:
            betaVal = float(file[5:])
            betaList.append(betaVal)
            configList.append(config.readFromFile(folderPath, file))
            
        self.fromList(configList, betaList)
    
    
    # ------------------------------------------------
    # utility
    # ------------------------------------------------
    
    # utility method - saves dataset object as folder of config object files - </method verified/>
    def saveDataset(self, directory, folderName):
        createFolder(directory, folderName)
        
        for i in range(0, len(self.configs), 1):
            curConfig = self.configs[i]
            curName = str("beta_" + str(self.betas[i]))
            curConfig.saveToFile(directory + "/" + folderName, curName)
            
    
    
    
 
    
# testing
testDirectory = "/Users/nayandusoruth/Desktop/Y3physics/LabModule/labProject/testFolder"    

config1 = config(20,2)
config1.runSimulation(10, 0.5)

config2 = config(20,2)
config2.runSimulation(10, 1)

config3 = config(20,2)
config3.runSimulation(10, 2)

configs = [config1, config2, config3]
betaList = np.array([0.5, 1, 2])

testDataset = dataset("fromBetas", configs, betaList=betaList, equilibrationSteps = 10, simulationSteps = 10, folderPath= testDirectory + "/testDataset")
testDataset.saveDataset(testDirectory, "testDataset")
configs1 = testDataset.configs
betas1 = testDataset.betas

testDataset2 = dataset("fromFile", configs, betaList=betaList, equilibrationSteps = 10, simulationSteps = 10, folderPath= testDirectory + "/testDataset")
configs2 = testDataset2.configs
betas2 = testDataset2.betas

for i in range(0,3):
    configs1[i].plotConfig(-1)
    configs2[i].plotConfig(-1)
    