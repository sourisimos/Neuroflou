#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:34:56 2023

@author: tsesmat
"""
## Imports
# Libraries
import numpy as np

# Scripts
from nfn_structure.SingleInputSigmoid import SingleInputSigmoid

class SingleInputLinkedSigmoidBlock:
# =============================================
# Cell of linked sigmoids class
# =============================================

    def __init__(self, l1_weight, x_star, p):
    # =============================================
    # Init method
    # =============================================
    # input :
    # l1_weight : list of float, list of cells' weights (wich are also store into lists)
    # x_star : float, value of the abscissa where the two sigmoids cross
    # p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 
        
        self.sigmoids = [SingleInputSigmoid(w, - np.log(1/p - 1) - w*x_star)
                         for w in l1_weight]
        self.nb_descript = len(l1_weight)

    def activate(self, input_):
    # =============================================
    # Activation function method
    # =============================================
    # input :
    # input_ : float, score for a single measure

        for sigmoid in self.sigmoids:
            sigmoid.activate(input_)
