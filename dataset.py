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
import pandas as pd


import NayanGeneralUtils.plotting as plotting

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
            
        # observables data
        self.energies = np.empty(len(self.configs))
        self.energiesDeviation = np.empty(len(self.configs))
        self.magnetisation = np.empty(len(self.configs))
        self.magnetisationDeviation = np.empty(len(self.configs))
        self.partition = np.empty(len(self.configs))
        self.partitionDeviation = np.empty(len(self.configs))
        self.entropies = np.empty(len(self.configs))
        self.entropiesDeviation = np.empty(len(self.configs))
        self.helmholtzEnergies = np.empty(len(self.configs))
        self.helmholtzEnergiesDeviation = np.empty(len(self.configs))
        
    # constructor from list of configs - </method verified/>
    def fromList(self, configList, betaList):
        self.betas = betaList
        self.configs = configList
        
    # constructor from array of betas - </method verified/>
    def fromBetas(self, betaList, equilibrationSteps, simulationSteps, printProgress=True):
        
        self.betas = betaList
        
        for i in range(0, len(betaList), 1):
            curConfig = config(self.size, self.dimension)
            curConfig.runSimulation(equilibrationSteps, betaList[i], saveConfigs=False) # equilibration steps - do not save these
            curConfig.runSimulation(simulationSteps, betaList[i], saveConfigs=True) # main steps - save configs
            self.configs.append(curConfig)
        
            if(printProgress):
                print("config ", i, "/", len(betaList), " generated")
    # constructor from folder location - </method verified/>
    def fromFile(self, folderPath):
        configList = []
        betaList = []
        
        folderElements = os.listdir(folderPath)
        folderElements.remove("metaData.json")
        
        # get dataset metadata
        metaData = loadDictFromJson(folderPath, "metaData")
        #self.betas = metaData["betas"]
        self.dimension = metaData["dimension"]
        self.size = metaData["size"]
        self.equilibrationSteps = metaData["equilibrationSteps"]
        self.simulationSteps = metaData["simulationSteps"]
        
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
        
        
        # get metadata
        metadata = {
                    "dimension":self.dimension,
                    "size":self.size,
                    "equilibrationSteps":self.equilibrationSteps,
                    "simulationSteps":self.simulationSteps
                    }
        
        # save metadata
        saveDictAsJson(directory + "/" + folderName, "metaData", metadata, indent=3)
        
        for i in range(0, len(self.configs), 1):
            curConfig = self.configs[i]
            curName = str("beta_" + str(self.betas[i]))
            curConfig.saveToFile(directory + "/" + folderName, curName)
            
    # utility method - export observable data as pandas dataframe
    def dataToDataframe(self, returnMetaparameters = False):
            data = {"beta":self.betas,
                    "energies":self.energies,
                    "energiesDeviation":self.energiesDeviation,
                    "magnetisation":self.magnetisation,
                    "magnetisationDeviation":self.magnetisationDeviation,
                    "partition":self.partition,
                    "partitionDeviation":self.partitionDeviation,
                    "entropies":self.entropies,
                    "entropiesDeviation":self.entropiesDeviation,
                    "helmholtzEnergies":self.helmholtzEnergies,
                    "helmholtzEnergiesDeviation":self.helmholtzEnergiesDeviation
                }
            
            
            dataframe = pd.DataFrame(data)
            
            if(returnMetaparameters):
                return self.dimension, self.size, dataframe
            
            return dataframe
            
            
    # ------------------------------------------------
    # data acquisition
    # methods which get the "latest" or average observable values for each config object
    # ------------------------------------------------
    def getAllObservables(self):
        self.getAverageEnergy()
        self.getAverageMagnetisation()
        self.getAveragePartition()
        self.getAverageEntropy()
        self.getAverageHelmholtz()
    
    def getAverageEnergy(self):
        
        self.energyDeviation = np.empty(len(self.configs))
        for i in range(0, len(self.energies),1):
            self.energies[i], self.energiesDeviation[i] = self.configs[i].averageEnergy(self.equilibrationSteps, self.simulationSteps + self.equilibrationSteps)
        
    def getAverageMagnetisation(self):
        
        self.magDeviation = np.empty(len(self.configs))
        for i in range(0, len(self.energies),1):
            self.magnetisation[i], self.magnetisationDeviation[i] = self.configs[i].averageMag(self.equilibrationSteps, self.simulationSteps + self.equilibrationSteps)
        
    def getAveragePartition(self):
        
        self.partitionDeviation = np.empty(len(self.configs))
        for i in range(0, len(self.energies),1):
            self.partition[i], self.partitionDeviation[i] = self.configs[i].averagePartition(self.equilibrationSteps, self.simulationSteps + self.equilibrationSteps)
        
    def getAverageEntropy(self):
        
        self.entropyDeviation = np.empty(len(self.configs))
        for i in range(0, len(self.energies),1):
            self.entropies[i], self.entropiesDeviation[i] = self.configs[i].averageEntropy(self.equilibrationSteps, self.simulationSteps + self.equilibrationSteps)
        
    def getAverageHelmholtz(self):
        
        self.helmholtzDeviation = np.empty(len(self.configs))
        for i in range(0, len(self.energies),1):
            self.helmholtzEnergies[i], self.helmholtzEnergiesDeviation[i] = self.configs[i].averageHelmholtz(self.equilibrationSteps, self.simulationSteps + self.equilibrationSteps)
            
    
    
    
 
    
# testing
testDirectory = "/Users/nayandusoruth/Desktop/Y3physics/LabModule/labProject/testFolder"  
dataSetDirectory = "/Users/nayandusoruth/Desktop/Y3physics/LabModule/labProject/Y3LabMaterialsGithub/Datasets/CodeTestingData/02_02/2026"  

# quick data generation testing
temperatures = np.linspace(0.1, 1, num=20)
betaList = 1/temperatures #np.linspace(0, 3.5, num=50)
#testDataset = dataset("fromBetas", betaList=betaList, equilibrationSteps = 1000, simulationSteps = 1000, dimension=2, size=20, folderPath= dataSetDirectory + "/testDataset20x20_1000_1000_temps")
#testDataset.saveDataset(testDirectory, "testDataset20x20_1000_1000_fromTemp3")


#configs1 = testDataset.configs
#betas1 = testDataset.betas

# quick data reading test
testDataset2 = dataset("fromFile", folderPath= testDirectory + "/testDataset20x20_1000_1000_fromTemp3")
testDataset2.getAllObservables()
data = testDataset2.dataToDataframe()
print(data["magnetisation"])

for i in range(0, len(testDataset2.configs), 1):
    config = testDataset2.configs[i]
    beta = testDataset2.betas[i]
    config.plotConfig(-1, title=str("config" + str(i) + " beta=" + str(beta)))

# quick plotting
def quickPlot(dataHeader, data):
    xData = data["beta"]
    yData = abs(data[dataHeader])
    yDataDeviation = data[dataHeader +"Deviation"]
    magPlot = plotting.plotter()
    magPlot.axisLabels(xLabel="beta", yLabel=dataHeader)
    magPlot.gridline()
    magPlot.scatter(xData, yData, marker = 'x', color='k', label="")
    magPlot.scatterYer(xData, yData, yDataDeviation, color='k', label="")
    magPlot.display()
    return magPlot

quickPlot("magnetisation", data)

#configs2 = testDataset2.configs
#betas2 = testDataset2.betas

#for i in range(0,3):
#    configs1[i].plotConfig(-1)
#    configs2[i].plotConfig(-1)
    
    
    
#testDataset.getAverageEnergy()