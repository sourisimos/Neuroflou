#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 6 09:40:22 2023

@author: tsesmat
"""
## Imports
# Libraries
import numpy as np

# Scripts
from nfn_structure.SingleInputLinkedSigmoidBlock import SingleInputLinkedSigmoidBlock
from tools.math_function import sigmoid

class FuzzyficationLayer:
# =============================================
# Fuzzyfication layer class
# =============================================

    def __init__(self, ll1_weight, lx_star, p):
    # =============================================
    # init method
    # =============================================
    # input :
    # ll1_weight : numpy array of float, steepness of the slope of the sigmoid functions 
    # lx_star: numpy array of float, Value of the absis where the sigmoid cross "p"
    # p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 

        self.ll1_weight = ll1_weight
        
        self.lx_star = lx_star
        self.p = p
        
        self.size_w = np.size(ll1_weight)
        self.size_x = np.size(lx_star)

        self.blocks = ([SingleInputLinkedSigmoidBlock(l_w, x_star, self.p)\
                        for l_w, x_star in zip(self.ll1_weight, self.lx_star)])



    def activate(self, l_input_, train=False):
    # =============================================
    # Activation function method
    # =============================================
    # input :
    # l_input_ : numpy array of float, list of scores for a single measure
    # train: bool, activate the training phase or not
    #
    # output :
    # out_temp1 : list of int, results of this layer after neuron activation
        
        # Initilisation
        self.out_temp1 = [0.0]*np.size(self.ll1_weight)
        out_index = 0

        # Activation of each layer's cell
        for multi_sigm, input_ in zip(self.blocks, l_input_):
            multi_sigm.activate(input_)
            for sigm in multi_sigm.sigmoids:
                self.out_temp1[out_index] = sigm.output
                out_index += 1 # Have to do like this due to nested lists 
        
        # Learning phase
        if train:

            # Initialisation 
            self.s_weight = np.zeros((self.size_w, self.size_w))
            self.s_x_star = np.zeros((self.size_w, self.size_x))
            
            ind = 0
            for i in range(len(self.ll1_weight)): # Going through all the input
                for j in range(len(self.ll1_weight[i])): # Goind through all the descriptors for a considered input
                    
                    b= - np.log(1/self.p -1) - self.ll1_weight[i][j] * self.lx_star[i]
                    self.s_weight[ind, ind] = (l_input_[i] - self.lx_star[i]) * \
                        sigmoid(self.ll1_weight[i][j], b, l_input_[i]) * \
                        (1 - sigmoid(self.ll1_weight[i][j], b, l_input_[i]))

                    ind += 1 # Have to do like this due to nested lists 

            ind = 0
            for i in range(len(self.lx_star)):
                for j in range(len(self.ll1_weight[i])):
    
                    b= -np.log(1/self.p -1) - self.ll1_weight[i][j]*self.lx_star[i]
                    self.s_x_star[ind, i] = - self.ll1_weight[i][j] * \
                        sigmoid(self.ll1_weight[i][j], b, l_input_[i]) * \
                        (1 - sigmoid(self.ll1_weight[i][j], b, l_input_[i]))

                    ind += 1 # Have to do like this due to nested lists 

        return self.out_temp1

    def back_propagation(self, back_prop4):
        # =============================================
        # Backpropagation method
        # =============================================
        # input :
        # back_prop4 : np array matrix, next layer's gradient relative to output
        # output :
        # back_prop_w : np array matrix, gradient of layer weights relative to output
        # back_prop_b : np array matrix, gradient of layer biais relative to output

        back_prop_w, back_prop_b = back_prop4 @ self.s_weight, back_prop4 @ self.s_x_star
        return back_prop_w, back_prop_b
