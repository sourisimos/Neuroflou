#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:48:30 2023

@author: tsesmat
"""
## Imports
# Scripts
from tools.confusion_matrix import confusion_matrix
from nfn_structure.NeuroFuzzyNetwork import NeuroFuzzyNetwork


def classif(opt_netw_var, ini_netw_var, p, cost_function, K ,data, target=None, print_res=True):
# =============================================
# Function of the NFN classification          #
# =============================================
# input :
# opt_netw_var : dict, dict of optimal network's variables (see INIT.py for def each variables)
# ini_netw_var : dict, dict of initial network's variables (see INIT.py for def each variables)
# p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 
# K : float, factor of the normalisation criterion in the CE
# data : nested lists of float in [0, 1], data's scores
# target : nested lists of {0, 1}, data's one hot vector
# print_res : bool, print the results in the console or not
# output:
# None

    # Initialisation
    printer = ''
    
    nfn_unopt = NeuroFuzzyNetwork(ini_netw_var, p, K, cost_function)
    nfn_opt = NeuroFuzzyNetwork(opt_netw_var, p, K, cost_function)
    
    ech_size = len(data)
    ce_opt = 0
    ce_unopt = 0  
    y_pred_opt = []
    y_pred_unopt = []
    
    # detecting and converting unique sample into the correct format (i.e. [[float, ...], ...])
    if any(isinstance(d, float) for d in data): 
        data = [data]
        target = [target]
        
    # Case of learning case
    if target:
        for d, t in zip(data, target):
            output_opt = nfn_opt.activate(d, l_target_=t, train=False)
            output_unopt = nfn_unopt.activate(d, l_target_=t, train=False)
            
            y_pred_opt.append(output_opt)
            y_pred_unopt.append(output_unopt)
            
            ce_opt_inc = nfn_opt.cost(output_opt, t)
            ce_unopt_inc = nfn_unopt.cost(output_unopt, t)
            
            ce_opt += ce_opt_inc
            ce_unopt += ce_unopt_inc
            
            print_res and print('\nData : ', d)
            print_res and print('Target : ', t)
            print_res and print('Output :', output_opt)
        
        ce_opt = ce_opt/ech_size
        ce_unopt = ce_unopt/ech_size
        
        conf_unopt = confusion_matrix(target, y_pred_unopt)
        conf_opt = confusion_matrix(target, y_pred_opt)

        # Printer
        printer = f"\n CE with initial variables: {ce_unopt}"
        printer += f"\n CE with optimized variables: {ce_opt}"
        printer += f"\n\n Confusions matrix with initial variables:\n{conf_unopt}"
        printer += f"\n Confusions matrix with optimized variables:\n{conf_opt}"
    
    # Case of exploitation phase
    else:
        for d in data:
            output_opt = nfn_opt.activate(d, l_target_=None, train=False)
            print_res and print('\nData : ', d)
            print_res and print('Output :', output_opt)

    return output_opt, ce_opt, printer
