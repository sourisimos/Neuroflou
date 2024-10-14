#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:39:40 2023

@author: tsesmat
"""
## Import
# Libraries
import random as rd

def rd_weights(ini_netw_var, rd_type):
# =============================================
# sigmoid function for membership value
# =============================================
# input :
# ini_netw_var : dict, dict of initial network variables (see INIT.py for def each variables)
# rd_type : {"bool", "num"}, type of the rdness 
# 
# output:
# rd_netw_var : dict, dict of network with randomized variables (see INIT.py for def each variables)
    
    # Intialisation
	ll1_weight, ll1_bias, ll2_weight, ll3_weight = ini_netw_var.values()
    
    # Boolean config randommization 
	if rd_type == "bool":
		w = [ll2_weight, ll3_weight]

		for ind_ll_var in range(len(w)): 
			for ind_l_var in range(len(w[ind_ll_var])):
				for ind_var in range(len(w[ind_ll_var][ind_l_var])):

					rd_bool = rd.randint(0, 1)
					w[ind_ll_var][ind_l_var][ind_var] = True if rd_bool == 1 else False 
		ans = [ll1_weight, ll1_bias, w[0], w[1]]

    # Numeric varaibles randomization
	elif rd_type == "num":
		w = [ll1_weight, ll1_bias]

		for ind_ll_var in range(len(w)): 
			for ind_l_var in range(len(w[ind_ll_var])):
				for ind_var in range(len(w[ind_ll_var][ind_l_var])):

					sgn = -1 if w[ind_ll_var][ind_l_var][ind_var]<0 else 1 # sign conservation with the initial variables
					w[ind_ll_var][ind_l_var][ind_var] = sgn*rd.uniform(0, 10)

		ans = [w[0], w[1], ll2_weight, ll3_weight]
    
    # In case of bool rd, we need at least one full path to ensure classif prediction
    # we control this condition manually after randommization
	for ind, l_var in enumerate(ans[3]): 
		if l_var.count(True)==0:
			i_rd_true = rd.randint(0,3)
			ans[3][ind][i_rd_true] = True

	rd_netw_var = {'w_num': ans[0],'x_star': ans[1],'w_bool1': ans[2],'w_bool2': ans[3]}
	return  rd_netw_var