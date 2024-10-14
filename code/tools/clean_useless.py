#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:57:37 2023

@author: tsesmat
"""
## Imports
# Scripts
from tools.deep_copy import deep_copy_dbl_nested_list
from nfn_structure.NeuroFuzzyNetwork import NeuroFuzzyNetwork


def are_better_netw(netw_var, desc_copy, p, K, data, target, cost_function):
# =============================================
# Function that compare the performance of two
# bool configuration for a given network
# =============================================
# input :
# netw_var : dict, dict of network's variables (see INIT.py for def each variables)
# desc_copy : double nested list of bools, Full configuration of a netw, correspond to netw_var config with 
#             1 bool switch to False
# p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 
# K : float, factor of the normalisation criterion in the CE
# data : nested lists of float in [0, 1], data's scores
# target : nested lists of {0, 1}, data's one hot vector
#
# output:
# is_better : bool, True if netw_var is the worst

    # Initialisation
    ce_1, ce_2 = 0, 0
    netw_mod = netw_var.copy()
    netw_mod['w_bool1'], netw_mod['w_bool1'] = desc_copy[0], desc_copy[1] 
    nfn_ini = NeuroFuzzyNetwork(netw_var, p, K, cost_function)
    nfn_simplif = NeuroFuzzyNetwork(netw_mod, p, K, cost_function)

    # Compute of the CE on all of the datas for both the new bool network and the initial one
    for d, t in zip(data, target):
        output_ini = nfn_ini.activate(d, l_target_=t, train=False)
        output_simplif = nfn_simplif.activate(d, l_target_=t, train=False)

        ce_1 += nfn_ini.cost(output_ini, t)
        ce_2 += nfn_simplif.cost(output_simplif, t)

    is_better = ce_1 >= ce_2
    return is_better

def are_same_netw(netw_var, desc_copy, p, K, data, target, cost_function):
# =============================================
# Function that compare the performance of two
# bool configuration for a given network
# =============================================
# input :
# netw_var : dict, dict of network's variables (see INIT.py for def each variables)
# desc_copy : double nested list of bools, Full configuration of a netw, correspond to netw_var config with 
#             1 bool switch to False
# p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 
# K : float, factor of the normalisation criterion in the CE
# data : nested lists of float in [0, 1], data's scores
# target : nested lists of {0, 1}, data's one hot vector
#
# output:
# are_equal : bool, True if the two netw have same performances

    # Initialisation
    co_1, co_2 = 0, 0
    netw_mod = netw_var.copy()
    netw_mod['w_bool1'], netw_mod['w_bool1'] = desc_copy[0], desc_copy[1] 
    nfn_ini = NeuroFuzzyNetwork(netw_var, p, K, cost_function)
    nfn_simplif = NeuroFuzzyNetwork(netw_mod, p, K, cost_function)

    # Compute of the CE on all of the datas for both the new bool network and the initial one
    for d, t in zip(data, target):
        output_ini = nfn_ini.activate(d, l_target_=t, train=False)
        output_simplif = nfn_simplif.activate(d, l_target_=t, train=False)

        co_1 += nfn_ini.cost(output_ini, t)
        co_2 += nfn_simplif.cost(output_simplif, t)
    are_equal = (co_1 == co_2)
    return are_equal


def clean_useless(netw_var, p, K, DATA, comparison, cost_function, nul=2):
# =============================================
# Function that try to set successively every 
# boolean to False in order to simplifie the
# netw
# =============================================
# input :netw_var
# netw_var : dict, dict of network's variables (see INIT.py for def each variables)
# p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 
# K : float, factor of the normalisation criterion in the CE
# data : nested lists of float in [0, 1], data's scores
# comparison : {'equal', 'better'}, type of comparison 
# nul : int, number of occurence of the simplification loop (only one lead to non idempotence issue)
#
# output:
# desc_bool[0] : list of list of bool, conjunction layer simplest config
# desc_bool[1] : list of list of bool, disjunction layer simplest config
    
    # Initialisation
    data, target = DATA["scores"], DATA["output"]
    _, __, w2, w3 = netw_var.values()
    desc_bool = deep_copy_dbl_nested_list([w2, w3])

    # comparison selection
    if comparison == "equal":
        func_comp = are_same_netw
    elif comparison == "better":
        func_comp = are_better_netw

    # Compute each neighbors network by switching 1 by 1 each bool
    for _ in range(nul):
        for i_co in range(len(desc_bool) - 1, -1, -1):
            nb_cell = len(desc_bool[i_co])
            nb_weight = len(desc_bool[i_co][0])

            for i_ce in range(nb_cell):
                for i_w in range(nb_weight):
                    desc_copy = deep_copy_dbl_nested_list(desc_bool)

                    # Simplify last layer first because it's most influent
                    if (i_co != len(desc_bool) - 1 or desc_bool[i_co][i_ce].count(True) != 1) and desc_bool[i_co][i_ce][i_w]:
                        desc_copy[i_co][i_ce][i_w] = not desc_bool[i_co][i_ce][i_w]

                        # Simplifying the initial network if the two nfn give the same CE
                        if func_comp(netw_var, desc_copy, p, K, data, target, cost_function):
                            desc_bool = deep_copy_dbl_nested_list(desc_copy)

    return desc_bool[0], desc_bool[1]


