#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:54:36 2026

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

 # ------------------------------------------------
 # cluster algorithms
 # ------------------------------------------------
 
def getCluster(config, startingCell, adjacencies):
    returnable =[startingCell]
    initialSpin = config[tuple(startingCell)]
     
    # get neighbours
    neighbours = [*getNeighbourIndices(startingCell, adjacencies, config.shape[0], config.ndim)]
    
     
    # check neighbour validity - remove invalid neighbours
    for i in range(0, len(neighbours), 1):
        neighbour = neighbours[i]
        if(not(config[tuple(neighbour)] == initialSpin)):
            neighbours.pop(i)
             
     
    if(len(neighbours) == 0):
    # base case - no valid neighbours
        pass
    else:
    # recursive case - getCluster on neighbours
        for neighbour in neighbours:
            returnable.append()