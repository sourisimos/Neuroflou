#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:26:44 2023

@author: tsesmat
"""
## Import
# Scripts
from tools.math_function import sigmoid


class SingleInputSigmoid:
# =============================================
# Sigmoid neuron class
# =============================================
    def __init__(self, weight, bias):
    # =============================================
    # Init method
    # =============================================
    # input :
    # weight : float, neuron weight
    # bias : float, neuron bias
    
        self.weight = weight
        self.bias = bias

    def activate(self, input_):
    # =============================================
    # Activation function method
    # =============================================
    # input :
    # input_ : float, score for a single measure

        self.output = sigmoid(self.weight, self.bias, input_) # Corresponds to the membership rate of the descriptor considered

