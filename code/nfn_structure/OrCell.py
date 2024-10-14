#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:37:38 2023

@author: tsesmat
"""

class OrCell:
# =============================================
# NFN's disjunction cell class            
# =============================================

    def __init__(self, l3_weight):
    # =============================================
    # Init method
    # =============================================
    # input :
    # l3_weight : list of bool, list of cell's weights

        self.l_weight = l3_weight

    def activate(self, out_temp2):
    # =============================================
    # Activation function method
    # =============================================
    # input :
    # out_temp : list of int, output of the previous layer

        self.output = max((i for w, i in zip(self.l_weight, out_temp2) if w), default=0)
