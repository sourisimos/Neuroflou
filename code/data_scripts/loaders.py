#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:23:36 2023

@author: tsesmat
"""
## Import
# Libraries
import pandas as pd
import os
import json




def load_data(folder, categ, is_training_phase, sepa = ' '):
# =============================================
# Function that load data from a csv
# =============================================
# input :
# folder : str, parent folder of the data file
# categ : str, main name of the file 
# sample_type : {"learning", "validation"}, type of the data
# sepa : ?, separator symbole in the data file
#
# output:
# data_dict: dict, dict of the datas divided in "scores" and "output"

    data_dict={}
    # load the data file
    path = os.getcwd() + '/../datasets/'+ folder +'/'
    with open(path + categ + "_label" +  ".json") as f:
        data_desc = json.load(f)

    for sample_type in ['learning', 'validation']:
        file = path + categ + ('_' + sample_type if is_training_phase else '') + '.txt'
        df = pd.read_csv(file,
                         sep=sepa, header=None,
                         names=data_desc['input'] + data_desc['output'] )
    
        # initialisation
        l_input = [] # list of scores
        l_output = [] # list of 1 hot vectors for the corresponding class
        
        # filling the lists
        for i in range(len(df)):            
            l_input.append([df[inp][i] for inp in data_desc['input']])
            l_output.append([df[out][i] for out in data_desc["output"]])
        data_dict[sample_type] = {"scores": l_input, "output": l_output}

    return data_dict.values(), data_desc
