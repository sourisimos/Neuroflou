#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 08:59:30 2023

@author: tsesmat
"""
## Import
# Libraries
import pandas as pd
import os



def stratified_sampling(frac, folder_name, sample_name, class_label):
# ===============================================
# Function qui effectue un echantillonnage      # 
# stratifié d'un jeu de donnée formaté pour le  #
# séparer en un échnatillon d'apprentissage et  #
# un échantillon de validation.                 #
# ===============================================
# input :
# frac : flaot in [0, 1], proportion of training data and validation data 
# there is a ratio of frac training data and 1-frac validation data
# folder : str, name of the parent folder of the datas
# file : str, file name 
# col_names : list of str, name of the different class
#
# output:
# None


    # Pah initialization and creation if data folder doesn't exist yet
    path = os.getcwd() + '/../datasets/'+ folder_name +'/'
    os.makedirs(path, exist_ok=True)
        
        
    # Initialisation
    sample = pd.read_csv(path + sample_name + '.txt',
                               sep=' ', header=None,
                               names=class_label)
    df_learning = pd.DataFrame()
    df_validation = pd.DataFrame()

    # Stratified sampling
    for name in class_label:
        grouped = sample[sample[name] == 1 ]
        df = grouped.reset_index()
        part_frac = df.sample(frac = frac)
        rest_part_complem = df.drop(part_frac.index)
    
        df_learning = pd.concat([df_learning, part_frac], ignore_index=True)
        df_validation = pd.concat([df_validation, rest_part_complem], ignore_index=True)
    
    # Save as a temporary file the two sample
    df_learning.to_csv(path+'temp_learning.txt', header=False, index=False, sep=' ')
    df_validation.to_csv(path+'temp_validation.txt', header=False, index=False, sep=' ')
    
    # Clean blank lines due to sample() method
    for file_dir in [path+sample_name+'_learning.txt', path+sample_name+'_validation.txt']:
        temp_file = path+'temp_validation.txt' if file_dir == path+sample_name+'_validation.txt'\
            else path+'temp_learning.txt'
        
        # directky writting in the correct folder
        with open(
            temp_file, 'r') as r, open(
                file_dir, 'w') as o:
    
            for line in r:
                if line.strip():
                    o.write(line)
                    
    # Delete the temp file
    os.remove(path+'temp_learning.txt')
    os.remove(path+'temp_validation.txt')
    return 
