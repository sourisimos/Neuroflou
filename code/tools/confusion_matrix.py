#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:02:47 2023

@author: tsesmat
"""
## Imports
# Libraries
import numpy as np

def confusion_matrix(y_true, y_pred):
# =============================================
# Function that give a pseudo confusion matrix
# It's the a mean membership values for every 
# single measure organized by reall class 
# =============================================
# y_pred : nested list of floats in [0, 1], list of scores for every measure
# y_true : nested list of elements in {0, 1}, one hot vectors of the corresponding measures
#
# output:
# str_conf_mat : str, confusion/membreship matrix
    
    
    str_conf_mat = ''
    n_classes = [0] * (len(y_true[0]) + 1)
    # y_pred_classes = [np.zeros_like(y_pred[0])] * (len(n_classes) - 1)
    
    # Get the number of prevision in each class
    for prev in y_true:
        ok = np.argmax(prev)
        n_classes[ok + 1] += 1
    
    # Get starting index of each class (they are stored by classes)
    ind_class = np.cumsum(n_classes)

    # mean of membership vectors grouped by class
    for i, (ind, ind_p) in enumerate(zip(ind_class[:-1], ind_class[1:])):
        y_pred_class = np.mean(y_pred[ind:ind_p], axis=0, dtype=np.float16)
        # y_pred_classes[i] = y_pred_class
        str_conf_mat += f"  {y_pred_class}\n"


    return str_conf_mat