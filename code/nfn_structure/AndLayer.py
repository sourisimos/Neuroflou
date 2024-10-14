#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:41:49 2023

@author: tsesmat
"""
## Import
# Libraries
import numpy as np
import itertools

# Scripts
from nfn_structure.AndCell import AndCell


class AndLayer:
# =============================================
# Conjunction layer class
# =============================================

    def __init__(self, ll2_weight):
    # =============================================
    # Init method
    # =============================================
    # input :
    # ll2_weight : list of list of bool, list of cells' weights (wich are also store into lists)

        self.ll2_weight = ll2_weight
        self.ands = [AndCell(l_weight) for l_weight in self.ll2_weight] # list of the cells in initial config
        
    
    def activate(self, out_temp1, train=False):
    # =============================================
    # Activation function method
    # =============================================
    # input :
    # out_temp1 : list of int, previous layer's results
    # train: bool, activate the training phase or not
    #
    # output :
    # out_temp2 : list of int, results of this layer after neuron activation

        # Initialisation
        self.out_temp2 = [0.0] * len(self.ands)
        
        # Activation of each layer's cell
        for i in range(len(self.ands)):
            self.ands[i].activate(out_temp1)
            self.out_temp2[i] = self.ands[i].output
            
        # Learning phase
        # The sensitivity of this layer is defined as 1 if the value is the pseudo minimum, and 0 otherwise
        if train:
            len_out1 = len(out_temp1)
            len_out2 = len(self.out_temp2)
            self.sensib = np.zeros((len_out2, len_out1))

            for i, j in itertools.product(range(len_out2), range(len_out1)):
                if self.ll2_weight[i][j] and self.out_temp2[i] == out_temp1[j]:
                        self.sensib[i, j] = 1

        return self.out_temp2


    def back_propagation(self, back_prop3):
    # =============================================
    # Backpropagation method
    # =============================================
    # input :
    # back_prop3 : np array matrix, next layer's gradient relative to output
    # output :
    # back_prop4 : np array matrix, gradient of layer weights relative to output

        back_prop4 = back_prop3 @ self.sensib 
        return back_prop4