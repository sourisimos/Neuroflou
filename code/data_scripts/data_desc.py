#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:42:31 2023

@author: tsesmat
"""
## Imports
# Libraries

import json
import os

def data_desc(score_label, class_label, folder_name, sample_name):
# =================================================
# Fonction qui génère un fichier json contenant   #
# les descriptions linguistiques des différentes  #
# entrées et sorties                              #
# =================================================
# input :
# score_label : list of str, labels des scores pris en compte dans cet échantillon
# class_label : list of str, labels des entrés 
# folder_name : str, nom du dossier parent contenant le fichier de données initial
# sample_name : str, nom donnée à cet échantillon
# 
# output:
# None
    path = os.getcwd() + '/../datasets/'+ folder_name +'/'

    data = {}
    data["input"] = score_label
    data["output"] = class_label
    with open(path + sample_name + "_label" + ".json", "w") as f:
        json.dump(data, f)
    return
