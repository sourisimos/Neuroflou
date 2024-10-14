#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:40:57 2023

@author: tsesmat
"""

class AndCell:
# =============================================
# NFN's conjunction cell class            
# =============================================

    def __init__(self, l2_weight):    
    # =============================================
    # Init method
    # =============================================
    # input :
    # l2_weight : list of bool, list of cell's weights

        self.l_weight = l2_weight

    def activate(self, out_temp1):
    # =============================================
    # Activation function method
    # =============================================
    # input :
    # out_temp : list of int, output of the previous layer

        self.output = min([i for i,w in zip(out_temp1, self.l_weight) if w ], default=0)
