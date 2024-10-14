#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:57:37 2023

@author: tsesmat
"""
## Imports
# Scripts
from tools.trad_b import trad_linked_b

def var_interp(opt_netw_var, p, score_meaning, classes_name, title):
# =============================================
# Intrepeter to human understandable of a 
# netw characteristics
# =============================================
# input :
# opt_netw_var : dict, dict of optimal network's variables (see INIT.py for def each variables)
# p : float in [0, 1], Crossing value for the two sigmoids representing the large and small descriptors for a score 
# score_meaning : dict of str, scores description
# classes_name : dict of str, class description
# title : str, Run title, used to label all saved outputs
# 
# output:
# printer : str, Interpretation results printer

    printer = "\n\nInterprétations:"
    
    w1, x_star, w2, w3 = opt_netw_var.values()
    
    b1 = trad_linked_b(w1, x_star, p)

    # Frontier between descriptors translation
    for i, meaning in enumerate(score_meaning):
        printer += f"\n La frontière grand/petit de {meaning} est {str(-b1[i][0]/w1[i][0])}"
    printer += "\n\n Les règles de décision sont:"

    # Traduction of conjunction layer rules
    and_rules = []
    for lbool1 in w2:
        not_first_and = False
        rule = ''
        for i, bool1 in enumerate(lbool1):
            if bool1: 
                i_s = i//2
                is_big = (i%2 == 0)
                rule += " ET "*not_first_and + score_meaning[i_s] + " est" + " grand"*is_big + " petit"*(not is_big)
                not_first_and = True
        and_rules.append(rule)
        
    # Trduction of disjunction layer rules (based on conjunction layer rules)
    rules = []
    for lbool2 in w3:
        not_first_or = False
        rule = ''
        for i, bool2 in enumerate(lbool2):
            if bool2: 
                is_big = (i%2 == 0)
                rule += " OU "*not_first_or + and_rules[i] 
                not_first_or = True
        rules.append(rule)
        

    for i, classe in enumerate(classes_name):
        printer += f"\n  La classe {classe} est prédite si: {rules[i]}"
        
    return printer

            