#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:56:05 2023

@author: tsesmat
"""
from nfn_structure.NeuroFuzzyNetwork import NeuroFuzzyNetwork


def optim_num(netw_var, p,
              NBREP, EPSILON, data_learning, ce_evol, CE_FOR_SLOPE, K, cost_function):

    # =============================================================================
    # Function that compute a steepest descent algo to optimise the C-E with update of the variables
    # =============================================================================
    # input : 
    # netw_var : dict, dict of optimal network's variables (see INIT.py for def each variables)
    # p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 
    # NBREP : int, Nb of gradient descent iterations
    # EPSILON : float, step of the gradient descent algorithm
    # data_learning : dict, dict of the datas divided in "scores" and "output"
    # ce_evol : list of float, Evolution of the C-E NBREP by NBREP
    # CE_FOR_SLOPE : int, Calibrates the buffer length used to estimate (linearly) the CE slope.
    # K : float, factor of the noramlisation criterion in the CE
    # 
    # output :
    # w_num: numpy array of float, steepness of the slope of the sigmoid functions 
    # x_star: numpy array of float, Value of the absis where the sigmoid cross "p"
    # ce_evol: list of float, Evolution of the C-E NBREP by NBREP, actualised with the new iterations
    # slope: float, CE steep for the current network based

    # Initialisation
    ech_size = len(data_learning["scores"])
    weights_num, x_num, w2, w3 = netw_var.values()
    is_firsts_iter = True
    
    for i in range(NBREP):
        ce = 0
        nfn = NeuroFuzzyNetwork(netw_var, p, K, cost_function)
        
        for j in range(ech_size):
            
            l_input_ = data_learning["scores"][j]
            l_target_ = data_learning["output"][j]
            
            nfn.activate(l_input_, l_target_, train=True, compute_grad_num=False)
            ce_inc = nfn.cost(nfn.output, l_target_)
            ce += ce_inc
        
        # Updating NFN num variables with steepest descent alg
        nfn.desc_grad(EPSILON, ech_size, compute_grad_num=False)
        weights_num = nfn.ll1_weight
        x_num = nfn.lx_star
        netw_var['w_num'] = nfn.ll1_weight
        netw_var['x_star'] = nfn.lx_star

        ce = ce/ech_size
        
        # If the NFN has never been optimized, we add the first occurrence to the CE list to set a first value.
        if len(ce_evol) == 0: 
            ce_evol.append(ce)
    
    ce_evol.append(ce)
    
    # Updating a sliding buffer to compute the current slope
    if len(ce_evol) > CE_FOR_SLOPE+1:
        ce_evol.pop(0)
        is_firsts_iter = False
    
    
    if is_firsts_iter:
        slope = (ce_evol[-1] - ce_evol[0]) / ((len(ce_evol)-1)*NBREP-1) # on the firsts optim num of a network there is one less value of diff 
        
    else:
        slope = (ce_evol[-1] - ce_evol[0]) / ((len(ce_evol)-1)*NBREP)

    
    
    return weights_num, x_num, ce_evol, slope

