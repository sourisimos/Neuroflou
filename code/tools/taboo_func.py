#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:54:47 2023

@author: tsesmat
"""

def init_taboo_seniority(bool_layers, seniority):
# =============================================================================
# Function that initialize seniority list to manage taboo algorithm
# =============================================================================
# input:
# bool_layers : double nested list of bool, pattern of the boolean configuration
# seniority : int, seniority criterion 
# 
# output: 
# bool_seniority : double nested list of int, initial senioirty of each bool of the bool config (set to max value)
    
    # Because every bool can be change once they are equal or above the seniority criterion, 
    # on initialisisation, we set all of them to seniority criterion

    bool_seniority = [
        [[seniority] * len(layer[0]) for _ in range(len(layer))] 
        for layer in bool_layers
    ]
    return bool_seniority

def comp_seniority(parent_seniority, i_co, i_ce, i_w):
# =============================================================================
# Function that compute the seniority of a network according to its parent 
# seniority and the boolean that has been changed 
# =============================================================================
# parent_seniority : double nested list of int, senoirity of parent bool config
#                    from wich we switched the boolean (i_co, i_ce, i_w)
# i_co : int, layer of the switched bool
# i_ce : int, cell of the switched bool
# i_w : int, position in cell of the swited bool
# 
# output :
# bool_seniority : double nested list of int, real time senioirty of each bool of the bool config

    # Increment each boolean seniority
    bool_seniority = [
        [
            [seniority + 1 for seniority in cell]
            for cell in layer
        ]
        for layer in parent_seniority
    ]
    
    # Set to 0 the switched bool 
    bool_seniority[i_co][i_ce][i_w] = 0

    return bool_seniority

