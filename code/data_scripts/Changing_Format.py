#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:18:58 2023

@author: tsesmat
"""
## Import
# Libraries
import pandas as pd
import os

# ===================================================
# Scripts that transform original raw datas into the
# correct format :                                  
# Score1 Score2 .... 1-hot vector for the classes
# Defaut separator is " " in load_data.py       
# ===================================================


def changing_format(score_label, score_id, class_label, class_id, folder_name, sample_name, file_name, sep, header):
# =================================================
# Fonction qui convertit un fichier de données    #
# en le bon format de donnée:                    #
# scores (normalisés) one-hot vecteur des classes #
# separateur: espace                              #
# =================================================
# input :
# score_label : list of str, labels des scores pris en compte dans cet échantillon
# score_id : list of int, position(s) des labels considérès dans le fichier de données initial
# class_label : list of str, labels des entrés 
# class_id : list of int, position(s) des entrées dans la fichier de données initial
# folder_name : str, nom du dossier parent contenant le fichier de données initial
# sample_name : str, nom donnée à cet échantillon
# file_name : str, fichier de données initial
# sep : str, separateur entre les informations dans le fichier de données initial
# header : bool, gère la présence d'un 'Header' i.e. une première ligne (dans le fichier de données initial) donnant la description des données
# 
# output:
# None

    path = os.getcwd() + '/../datasets/'+ folder_name +'/'

    # detection du format du fichier brute
    is_one_hot = len(class_id) > 1
    
    # gestion de la présence d'un header
    if header:
        df_all = pd.read_csv(path + file_name, sep=sep)
    elif not header:
        df_all = df_all = pd.read_csv(path + file_name, sep=sep, header = None)

    df = pd.DataFrame()
    
    # Création du fichier 
    if is_one_hot:
        for i, name in zip(score_id + class_id , score_label + class_label):
            df[name] = df_all.iloc[:,i]

    else:
        ind_class = class_id[0]

        for i, name in zip(score_id, score_label):
            df[name] = df_all.iloc[:,i]

        for c_lab in class_label:
            df[c_lab] = 1*(df_all.iloc[:,ind_class] == c_lab)
    
    # normalisation
    for s in score_label:
        df[s] = pd.to_numeric(df[s])
        df[s] = df[s]/df[s].max()

    df.to_csv(path + sample_name + '.txt', header=False, index=False, sep=' ')
    
    return