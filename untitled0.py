#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 14:27:19 2026

@author: nayandusoruth
"""

import numpy as np
import copy

from numba import jit
from numba import njit

import numba as nb

testArray = np.array([[1, 2, 3], 
                      [4, 5, 6], 
                      [7, 8, 9]])
indices = np.array([0,0]).T
#print(indices)
#print(tuple(3))
#print(testArray[indices])

# utility function - returns the adjacent indicies to "index" - note modulo operation over size to handle edge cases - </function verified/>
@jit()
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


    # return
    return indices



#@jit()
def getArrayVal(array, indices, size): # array is N dimensional np.array, indices is 1D array with length N

    dimensions = len(indices)
    flatArray = array.flatten()
    flattenedIndex = 0
    
    for i in range(0, dimensions, 1):
        #print("i: ",i, " ", indices[i] * ((size-1) ** (i)), " = ", indices[i], " x ", (size-1) , "^", i)
        flattenedIndex = flattenedIndex + indices[i] * ((size) ** (i))
    
    return flatArray[flattenedIndex]


@jit()
def getNeighbours(config, size, index):
    # get indices for neighbours, and setup returnable
    indices = getNeighbourIndices(index, size).T
    neighbourValues = np.empty(indices.shape[1])
    
    # iterate through all indices and assign to returnable
    for i in range(0, indices.shape[1]):
        currentIndex = indices[:,i]#.tolist()
        neighbourValues[i] = getArrayVal(config, currentIndex, size)
        
    # return
    return neighbourValues
    
coord = np.array([0, 0])
coord1 = np.array([1, 0])
coord2 = np.array([2, 0])
coord3 = np.array([0, 1])
coord4 = np.array([1, 1])
coord5 = np.array([2, 1])
coord6 = np.array([0, 2])
coord7 = np.array([1, 2])
coord8 = np.array([2, 2])
"""
#print(nb.typeof(coord))
print(getArrayVal(testArray, coord, 3))
print(getArrayVal(testArray, coord1, 3))
print(getArrayVal(testArray, coord2, 3))
print(getArrayVal(testArray, coord3, 3))
print(getArrayVal(testArray, coord4, 3))
print(getArrayVal(testArray, coord5, 3))
print(getArrayVal(testArray, coord6, 3))
print(getArrayVal(testArray, coord7, 3))
print(getArrayVal(testArray, coord8, 3))
"""


testArray2 = np.array([["00", "01", "02", "03", "04"],
                       ["10", "11", "12", "13", "14"],
                       ["20", "21", "22", "23", "24"],
                       ["30", "31", "32", "33", "34"],
                       ["40", "41", "42", "43", "44"]])



dim = 3
size = 5
# utility method - given 1D array of Nd array sizes; returns array to power of index in array - is used for coord flattening reasons - </function verified/>
@jit()
def getPowers(dim, size):
    sizes = np.full(dim, size)
    dimensions = np.arange(0, dim, 1)
    return np.power(sizes, dimensions)
print(getPowers(3,5))


for i in range(0, 5):
    for j in range(0,5):
        coord00 = np.array([j,i])
        #print(np.flip(coord00), ":", getArrayVal(testArray2, coord00, 5))
#print(nb.typeof(nb.int64[::1]))
#print(getNeighbourIndices(coord, 3))

#print(getNeighbours(testArray, 3, coord))