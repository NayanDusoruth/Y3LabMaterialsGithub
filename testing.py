#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:58:27 2026

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


# testing
T = 300
k_b = 1.38*10**(-23)
#beta = 1 / (k_b * T)

testConfig = config(20,2)

#blasgg
#newConfig = np.array([[2,2],[2,2]])
#newConfig2 = np.array([[3,3],[3,3]])
#testConfig2D.appendConfig(newConfig)
#testConfig2D.appendConfig(newConfig2)


#print(testConfig2D.state)
#testConfig2D.plotConfig()
startTime = time.time()
testConfig.runSimulation(55, beta(1.5), plotConfigs=True, printProgress=False, saveFigs=False, saveDirectory="/Users/scarlettspiller/MSciLabs/OutputFigs")
endTime = time.time()

timeDiff = startTime - endTime
print("timeElapsed: ", timeDiff)
#print(testConfig.state)
#print("shape",testConfig.getConfig(0).shape)
#print("shape",testConfig.getConfig(0))


cluster = getCluster(testConfig.getConfig(-1), np.array([0,0]), testConfig.adjacencyIndices)
print(cluster)
testConfig.saveToFile(testDirectory, "testName")

readConfig = config.readFromFile(testDirectory, "testName")

t = -1
#testConfig.plotConfig(t=t)
#readConfig.plotConfig(t=t)

#print(testConfig.calcEnergy(t=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

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