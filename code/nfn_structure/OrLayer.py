#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:39:06 2023

@author: tsesmat
"""
## Imports
# Libraries
import numpy as np
import itertools

# Scripts
from nfn_structure.OrCell import OrCell


class OrLayer:
# =============================================
# Disjunction layer class
# =============================================

    def __init__(self, ll3_weight):
    # =============================================
    # Init method
    # =============================================
    # input :
    # ll3_weight : list of list of bool, list of cells' weights (wich are also store into lists)
        
        self.ll3_weight = ll3_weight
        self.ors = [OrCell(l_weight) for l_weight in self.ll3_weight]


    def activate(self, out_temp2, train=False):
    # =============================================
    # Activation function method
    # =============================================
    # input :
    # out_temp2 : list of int, previous layer's results
    # train: bool, activate the training phase or not
    #
    # output :
    # out_temp3 : list of int, results of this layer after neuron activation
        
        # Initialisation
        self.out_temp3 = [0.0] * len(self.ors)

        # Activation of each layer's cell
        for i in range(len(self.ors)):
            self.ors[i].activate(out_temp2)
            if self.ors[i].output <= 1:
                self.out_temp3[i] = self.ors[i].output
        
        # Learning phase
        # The sensitivity of this layer is defined as 1 if the value is the pseudo maximum, and 0 otherwise
        if train:
            len_out2 = len(out_temp2)
            len_out3 = len(self.out_temp3)
            self.sensib = np.zeros((len_out3, len_out2))
            for i, j in itertools.product(range(len_out3), range(len_out2)):
                if self.ll3_weight[i][j] and self.out_temp3[i] == out_temp2[j]:
                    self.sensib[i, j] = 1

        return self.out_temp3


    def back_propagation(self, back_prop2):
    # =============================================
    # Backpropagation method
    # =============================================
    # input :
    # back_prop2 : np array matrix, next layer's gradient relative to output
    # output :
    # back_prop3 : np array matrix, gradient of layer weights relative to output
    
        back_prop3 = back_prop2 @ self.sensib 
        return back_prop3