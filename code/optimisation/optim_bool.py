#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:56:27 2023

@author: tsesmat
"""


def optim_bool(rzo_dict, rem_iter):
    # =============================================================================
    # Function that give the optimal network from a dictionnary (rzo_dict) according to the
    # best linear estimated C-E for a finit nbr of iteration (rem_iter)
    # Is the estimated value of the CE is negativ, we consider the optimal one as the one
    # which cross the abs line in first
    # =============================================================================
    # input: 
    # rzo_dict : dict of dict, dict with each dict characteristic of networks
    # rem_iter : int, remaining iteration until the maximal iteration of steepest descent is reached
    #
    # output :
    # optim_id : str(int), key dict of the most promising network at the moment 

    #Initialisation
    min_exp_val = float('inf')
    min_exp_abs = float('inf')
    only_pos_val = True

    # Research of the optimal network according to the nbr of iteration available
    for id_rzo in rzo_dict:

        # Updtating the linear prevision based on the nbr of remaining iter
        cur_exp_val = rzo_dict[id_rzo]['ce_evol'][-1] + rzo_dict[id_rzo]['slope'] * max(rem_iter,0) 

        # As long as there is a neg value, every positive prevision are worst
        if cur_exp_val < 0:
            only_pos_val = False
            cur_exp_abs = - rzo_dict[id_rzo]['ce_evol'][-1]/rzo_dict[id_rzo]['slope']
            if (cur_exp_abs < min_exp_abs):
                optim_id = id_rzo
                min_exp_abs = cur_exp_abs

        # Compute the optimal network if every values atm are positives
        elif only_pos_val and cur_exp_val < min_exp_val:
                optim_id = id_rzo
                min_exp_val = cur_exp_val

    return optim_id
