#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:39:50 2023

@author: tsesmat
"""
################################### 
# UNUSED BUT WORK IF ADDED IN NFN
###################################

import numpy as np

class NormalizationLayer:

    # constructor to initialize the dimension
    def __init__(self, dimension):
        self.dimension = dimension

    # activate function to calculate the output
    def activate(self, out_temp3, train=False):
        self.out_temp4 = [0.0] * self.dimension
        sum_ = sum(out_temp3)
        for i in range(self.dimension):
            self.out_temp4[i] = out_temp3[i]/sum_
        if train:
            
            self.sensib = np.array([[out_temp3[j]/(sum_**2) 
                                    for i in range(self.dimension)] 
                                        for j in range(self.dimension)])
            self.sensib = np.eye(self.dimension) / sum_ - self.sensib


        return self.out_temp4

    def back_propagation(self, back_prop1):
        # print(back_prop1)
        return back_prop1 @ self.sensib
