#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:00:16 2023

@author: tsesmat
"""
## Import
# Libraries
import numpy as np

## Global parameters
# title : String, sous quel nom de fichier vont être mis les résultats
# folder : str, nom du fichier parent
# categ : str, nom de l'échantillon d'entrainement
# ITER_GLOB : int, nombre d'itérations de descente de gradient effectuées au total
# NBREP_r : int, nb de descentes de gradient effectuées sur chaque réseau voisin lors de l'exploration
# NBREP_t : int, nb de descentes de gradient effectuées sur le réseau optimal sélectionné
# SENIORITY : int, Nombre minimum de réseaux qui doivent être fermés avant qu'un booléen déjà échangé puisse l'être à nouveau.
# CE_FOR_SLOPE : int, calibre la longueur de la mémoire tampon utilisée pour estimer (linéairement) la pente de la fonction cout.
# EPSILON : float, pas de l'algorithme de descente de gradient
# p : float in [0, 1], Valeur de croisement des deux sigmoïdes représentant les descripteurs grand et petit pour un score donné
# cost_function: {'squared_error', 'cross_entropy'}, type fonction cout pour l'apprentissage
# K : float, valeur du facteure devant le critère de normalisation de la fonction cout
# rd_type : {"bool", "num", None}, type d'aléatoire
# disp_loading: bool, active ou non l'affichage la progression de l'entrainement
# title_protec: bool, associe ou non une clé unique (selon l'horaire) au titre afin d'éviter d'écraser une sortie

cs_title = "Test"
cs_folder = "cancer"
cs_categ = "reduced"
cs_ITER_GLOB = 100 #20000
cs_NBREP_r = 10
cs_NBREP_t = 10
cs_SENIORITY = 20
cs_CE_FOR_SLOPE = 5
cs_EPSILON = 0.1
cs_p = 0.5
cs_cost_function = 'squared_error'
cs_K=0
cs_rd_type = None
cs_disp_loading = True
cs_title_protec = True



## Weights and biais
# w_num: numpy array of float, l'inclinaison de la pente des fonctions sigmoïdes
# x_star: numpy array of float, Valeurs des abscisses où se croisent les deux sigmoïdes
# w_bool1: double nested list of bool, configuration des règles pour la couche de conjonction
# w_bool2: double nested list of bool, configuration des règles pour la couche disjonction


network_variables = {

    #####################################################
    #     Crashed_plane_classic                         # 
    #####################################################
    "crashed_plane_classic": {
        "w_num":np.array([[1.0, -1.0], [1., -1.0]]),
        "x_star": np.array([0.5, 0.5]),
        "w_bool1": [[True, False, False, True], [False, True, True, False],
                      [False, True, False, True], [False, False, False, False]],
        
        "w_bool2": [[True, False, False, False], [False, True, False, False],
                      [False, False, True, False]]
        },

    #####################################################
    #     Cancer_reduced                                #
    #####################################################
    "cancer_reduced": {
        "w_num": np.array([[1.0, -1.0], [1., -1.0], [1., -1.0]]),
        "x_star": np.array([0.5, 0.5, 0.2]),
        "w_bool1": [[ False, False, False, False,  True, False],
                      [False,  False, False, False, False,  True],
                      [ False,  False, False, False,  False, False],
                      [False, False, False, False, False, False]],
        
        "w_bool2": [[ False, True, False, False],
                      [True, False, False, False]]
        },
    "cancer_new_test": {
        "w_num": np.array([[1.0, -1.0], [1., -1.0], [1., -1.0]]),
        "x_star": np.array([0.5, 0.5, 0.2]),
        "w_bool1": [[ False, False, False, False,  True, False],
                      [False,  False, False, False, False,  True],
                      [ False,  False, False, False,  False, False],
                      [False, False, False, False, False, False]],
        
        "w_bool2": [[ False, True, False, False],
                      [True, False, False, False]]
        },
    #####################################################
    #     Cancer_classic                                #
    #####################################################
    "cancer_classic": {
        "w_num" : np.array([[1.0, -1.0], [1., -1.0], [1., -1.0], [1., -1.0], [1., -1.0]]),
        "x_star": np.array([0.5, 0.5, 0.5, 0.3, 0.2]),
        "w_bool1": [[ False, False,  False, False,  False, False,  False, False,  True, False],
                      [False,  False, False,  False, False,  False, False,  False, False,  True],
                      [ False,  False, False,  False,  False, False, False, False,  False, False],
                      [False, False, False, False, False, False, False,  False, False, False]],
        
        "w_bool2": [[ False, True, False, False],
                      [True, False, False, False]]
        },
    }
