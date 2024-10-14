#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:53:41 2023

@author: tsesmat
"""
## Imports
# Libraries
import numpy as np

def trad_linked_b(w,x,p):
# =============================================
# conversion function from x absissa value and p
# value a x abssissa to biais b for linked cells
# =============================================
# input :
# w : float, sigmoid's weight
# x : float, value of the abscissa where the two sigmoids cross (on p)
# p : float, crossing value for the two sigmoids representing the large and small descriptors for a score 
# 
# output:
# b : float, corresponding bias

    b = np.zeros(np.shape(w))

    for i in range(len(w)):
        b[i] = - np.log(1/p - 1) - w[i,:]* x[i]

    return b
