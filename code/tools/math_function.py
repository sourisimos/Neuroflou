#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:42:03 2023

@author: tsesmat
"""
## Imports
# Libraries
import numpy as np


def sigmoid(weight, bias, input_):
# =============================================
# sigmoid function for membership value
# =============================================
# input :
# weight : float, sigmoid's weight
# bias : float, sigmoid's bias
# input_ : float, score for a single measure
# 
# output:
# sig_value : float, value of the sigmoidal function
    sig_value = 1.0 / (1.0 + np.exp(- (input_ * weight + bias)))
    return sig_value


def squared_error(lx, ly, K=1):
# =============================================
# Cost function: mean squarred error with a 
# regularisation term for normalisation 
# =============================================
# input :
# lx : list of floats in [0, 1], prevision of class membership after scores have passed through the NFN
# ly : list of elements in {0, 1}, real observed one hot vectors
# K : float, regularisation term constant
# 
# output:
# cost : float, cost value 

    cost = sum([(x - y)**2 for x, y in zip(lx, ly)]) + K*(sum([x for x in lx]) -1)**2
    return cost

def cross_entropy(lx, ly, K=1):
# =============================================
# Cost function: cross entropy with a 
# regularisation term for normalisation 
# =============================================
# input :
# lx : list of floats in [0, 1], prevision of class membership after scores have passed through the NFN
# ly : list of elements in {0, 1}, real observed one hot vectors
# K : float, regularisation term constant
# 
# output:
# cost : float, cost value 

    cost = -sum([y * np.log(x) for x, y in zip(lx, ly)]) + K*(sum([x for x in lx]) -1)**2
    return cost