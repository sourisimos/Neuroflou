#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:22:08 2023

@author: tsesmat
"""
import argparse
import time

## Import
# Scripts
from tools.rd_weights import rd_weights
from nfn_structure.classif import classif
from nfn_structure.TrainingTree import TrainingTree
from tools.var_interp import var_interp
from tools.displayers import display_text, display_sigmoids
from data_scripts.loaders import load_data
from INIT import cs_title, network_variables, cs_p, cs_K,\
    cs_SENIORITY, cs_ITER_GLOB, cs_NBREP_r, cs_NBREP_t, cs_EPSILON, cs_CE_FOR_SLOPE,\
    cs_rd_type, cs_folder, cs_categ, cs_disp_loading, cs_title_protec, cs_cost_function




def main(arg):
# =============================================
# Function to be executed to run the program  #
# =============================================
# input :
# args : parser, voir la définition des composants du parser
#
# output:
# None

	

	title = arg.title 
	folder = arg.folder 
	categ = arg.categ
	ITER_GLOB = arg.iter_glob
	NBREP_r = arg.nbrep_r
	NBREP_t = arg.nbrep_t
	SENIORITY = arg.seniority
	CE_FOR_SLOPE = arg.ce_for_slope 
	EPSILON = arg.epsilon
	p = arg.p
	K = arg.k 
	rd_type = arg.rd_type
	disp_loading = arg.disp_loading
	title_protec = arg.title_protec
	cost_function = arg.cost_function

	const_num_dict = {
		"ITER_GLOB": ITER_GLOB,
		"NBREP_r": NBREP_r,
		"NBREP_t": NBREP_t,
		"SENIORITY": SENIORITY,
		"CE_FOR_SLOPE": CE_FOR_SLOPE,
		"EPSILON": EPSILON,
		"p": p,
		"K": K
	}

	# overwritting protection
	t = title + '_' + time.strftime('%H:%M:%S', time.gmtime(time.time() )) * (title_protec*1) 

	disp_loading and print("Started")
   	
	## Initialisation
	key_data = folder + '_' + categ
	ini_netw_var = network_variables[key_data]
	# Load the datas
	(learning_data, validation_data), data_desc = load_data(folder, categ, is_training_phase = True)

	l_input_learning = learning_data['scores']
	target_learning = learning_data['output']
	l_input_validation = validation_data['scores']
	target_validation = validation_data['output']

    # Randomization
	if rd_type: 
		ini_netw_var = rd_weights(ini_netw_var, rd_type)

    ## Learning phase
    # Training
	tree = TrainingTree(ini_netw_var, learning_data, disp_loading, t, const_num_dict, cost_function)
	opt_netw_var, printer_perf = tree.train()

    # Classification
	output_opt_lear, co_opt_lear, printer_lear = classif(opt_netw_var, ini_netw_var, p, cost_function, K, l_input_learning, target_learning, False)
	output_opt_val, co_opt_val, printer_vali = classif(opt_netw_var, ini_netw_var, p, cost_function, K, l_input_validation, target_validation, False)
	
    #Interpretation
	printer_interp = var_interp(opt_netw_var, p, data_desc['input'], data_desc['output'], t)

	## Display
	display_text(t, printer_perf, printer_lear, printer_vali, printer_interp, opt_netw_var, network_variables[key_data], const_num_dict, categ, folder, cost_function)
	display_sigmoids(opt_netw_var['w_num'], opt_netw_var['x_star'], p, data_desc['input'], t)

	return 

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--title", help="OPTIONAL - string, nom de fichiers sous lesquels vont être mis les résultats -- DEFAULT VALUE: see INIT.py, cs_title", default=cs_title)
parser.add_argument("-tp", "--title_protec", help="OPTIONAL - bool, associe ou non une clé unique (selon l'horaire) au titre afin d'éviter d'écraser une sortie -- DEFAULT VALUE: see INIT.py, cs_title_protec", type=bool, default=cs_title_protec)
parser.add_argument("-f", "--folder", help="OPTIONAL - string, nom du dossier parent des données-- DEFAULT VALUE: see INIT.py, cs_folder", default=cs_folder)
parser.add_argument("-c", "--categ", help="OPTIONAL - string, nom des données d'entrainement -- DEFAULT VALUE: see INIT.py, cs_categ", default=cs_categ)
parser.add_argument("-ig", "--iter_glob", help="OPTIONAL - int, nombre d'itérations de descente de gradient effectuées au total -- DEFAULT VALUE: see INIT.py, cs_ITER_GLOB", type=int, default=cs_ITER_GLOB)
parser.add_argument("-nr", "--nbrep_r", help="OPTIONAL - int, nb de descente de gradient effectuées sur chaque réseau voisin lors de l'exploration -- DEFAULT VALUE: see INIT.py, cs_NBREP_r", type=int, default=cs_NBREP_r)
parser.add_argument("-nt", "--nbrep_t", help="OPTIONAL - int, nb de descentes de gradient effectuées sur le réseau optimal sélectionné -- DEFAULT VALUE: see INIT.py, cs_NBREP_t", type=int, default=cs_NBREP_t)
parser.add_argument("-s", "--seniority", help="OPTIONAL - int, nb minimum de réseaux qui doivent être fermés avant qu'un booléen déjà échangé puisse l'être à nouveau -- DEFAULT VALUE: see INIT.py, cs_SENIORITY", type=int, default=cs_SENIORITY)
parser.add_argument("-cs", "--ce_for_slope", help="OPTIONAL - int, calibre la longueur de la mémoire tampon utilisée pour estimer (linéairement) la pente de la fonction cout -- DEFAULT VALUE: see INIT.py, cs_CE_FOR_SLOPE", type=int, default=cs_CE_FOR_SLOPE)
parser.add_argument("-e", "--epsilon", help="OPTIONAL - float, pas de l'algorithme de descente de gradient -- DEFAULT VALUE: see INIT.py, cs_EPSILON", type=float, default=cs_EPSILON)
parser.add_argument("-p", "--p", help="OPTIONAL - float in [0, 1], valeur de croisement des deux sigmoïdes représentant les descripteurs grand et petit pour un score donné -- DEFAULT VALUE: see INIT.py, cs_p", type=float, default=cs_p)
parser.add_argument("-cf", "--cost_function", help="OPTIONAL - {squared_error, cross_entropy}, type de la fonction cout d'apprentissage -- DEFAULT VALUE: see INIT.py, cs_cost_function",  default=cs_cost_function)
parser.add_argument("-k", "--k", help="OPTIONAL - float, valeur du facteure devant le critère de normalisation de la fonction cout -- DEFAULT VALUE: see INIT.py, cs_K", type=float, default=cs_K)
parser.add_argument("-rd", "--rd_type", help="OPTIONAL - {'bool', 'num', None}, type d'aléatoire -- DEFAULT VALUE: see INIT.py, cs_rd_type", default=cs_rd_type)
parser.add_argument("-dl", "--disp_loading", help="OPTIONAL - bool, active ou non l'affichage la progression de l'entrainement -- DEFAULT VALUE: see INIT.py, cs_disp_loading", type=bool, default=cs_disp_loading)

args = parser.parse_args()

## Excution of main prog
if __name__ == "__main__":
    main(args)
