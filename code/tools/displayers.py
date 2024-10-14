#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:25:38 2023

@author: tsesmat
"""
##Imports
#Librarys
import os
import numpy as np
import matplotlib.pyplot as plt

#Scripts
from tools.math_function import sigmoid
from tools.trad_b import trad_linked_b


def display_text(title, printer_perf, printer_lear, printer_vali, printer_interp,  opt_netw_var, ini_netw_var, const_num_dict, categ, folder, cost_function):
# =============================================
# Console_log displayer
# =============================================
# input :
# title : str, Run title, used to label all saved outputs
# printer_perf : str, algorithme performance printer
# printer_lear : str, learning sample results printer
# printer_interp : validation sample results printer
# opt_netw_var : dict, dict of optimal network's variables (see INIT.py for def each variables)
# ini_netw_var : dict, dict of initial network's variables (see INIT.py for def each variables)
# 
# output:
# None

	ITER_GLOB = const_num_dict['ITER_GLOB'] 
	NBREP_r = const_num_dict['NBREP_r']
	NBREP_t = const_num_dict['NBREP_t']
	EPSILON = const_num_dict['EPSILON']
	SENIORITY = const_num_dict['SENIORITY']
	CE_FOR_SLOPE = const_num_dict['CE_FOR_SLOPE']
	p = const_num_dict['p']
	K = const_num_dict['K']

	w1, x1, w2, w3 = opt_netw_var.values()
	iw1, ix1, iw2, iw3 = ini_netw_var.values()
	path = os.getcwd()+'/output_temp/console_log/'
	os.makedirs(path, exist_ok=True)
	with open(path+str(title)+'.txt', 'a+') as file_console:
		file_console.write(printer_perf)

		file_console.write("\n\nGlobal parameter: ")
		file_console.write("\n Datas: " + folder + '_' + categ)
		file_console.write("\n NBREP exploration: " + str(NBREP_r))
		file_console.write("\n NBREP exploitation: " + str(NBREP_t))
		file_console.write("\n EPSILON: "+ str(EPSILON))
		file_console.write("\n ITER_GLOB: " + str(ITER_GLOB))
		file_console.write("\n SENIORITY: " + str(SENIORITY))
		file_console.write("\n CE_FOR_SLOPE: " + str(CE_FOR_SLOPE))
		file_console.write("\n p: " + str(p))
		file_console.write("\n K: " + str(K))
		file_console.write("\n cost_function: " + str(cost_function))

    	# Variables interpretations
		file_console.write(printer_interp)
        
    	# Results on learning data
		printer_lear and file_console.write("\n\nLearning datas:")
		file_console.write(printer_lear)

		# Results on validation datas
		printer_lear and file_console.write("\n\nValidation datas:")
		file_console.write(printer_vali)

    	# Initial variables
		file_console.write("\n\nInitial variables: ")
		file_console.write("\n w_num:  "+ str(iw1))
		file_console.write("\n x_star: "+ str(ix1))
		file_console.write("\n w_bool1:"+ str(np.array(iw2)))
		file_console.write("\n w_bool2:"+ str(np.array(iw3)))
	 
	    # Founded variables after learning phase
		file_console.write("\n\nOptimized variables: ")
		file_console.write("\n w_num:  "+ str(w1))
		file_console.write("\n x_star: "+ str(x1))
		file_console.write("\n w_bool1:"+ str(np.array(w2)))
		file_console.write("\n w_bool2:"+ str(np.array(w3)))
	return


l_color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
l_label = ["Big", "Small"]

def display_sigmoids(ll_weight, lx_star, p, score_meaning, title, l_col=l_color, l_lab=l_label):
# =============================================
# Membership sigmoid displayer
# =============================================
# input :
# ll_weight: numpy array of float, steepness of the slope of the sigmoid functions 
# lx_star: numpy array of float, Values of the abscissa where the two sigmoids cross
# p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 
# printer_interp : validation sample results printer
# score_meaning : dict, dict of optimal network's variables (see INIT.py for def each variables)
# title : str, Run title, used to label all saved outputs
# l_lab : list of str, linguistic descriptor labels
# l_col : list of str, color label for plot
# output:
# None
    
    ll_b = trad_linked_b(ll_weight, lx_star, p) # biais computation 

    x_values = np.linspace(-5, 5, 1001)
    
    for l_weight, l_b, meaning in zip(ll_weight, ll_b, score_meaning):
        plt.vlines([0, 1], -10, 10, "k", linestyle='--', linewidth=1)

        for color, label, (weight, b) in zip(l_col, l_lab, zip(l_weight, l_b)):
            plt.plot(x_values, sigmoid(weight, b, x_values), color=color, label=label)

        plt.legend()
        plt.title(f"Membership functions of {meaning}'s descriptors")
        plt.ylim(-0.2, 1.2)
        plt.xlim(-1, 2)

        path = os.path.join(os.getcwd(), 'output_temp', 'membership_func', title)

        os.makedirs(path, exist_ok=True)

        plt.savefig(os.path.join(path, f'{meaning}.png'))
        plt.close()

    return