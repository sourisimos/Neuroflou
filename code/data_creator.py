#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:15:42 2023

@author: tsesmat
"""
import argparse

from data_scripts.Changing_Format import changing_format
from data_scripts.stratified_sampling import stratified_sampling
from data_scripts.data_desc import data_desc


def data_creator(args):
# =================================================
# Fonction qui a parti d'un fichier de données
# brute, le converti en le bon format, créée un 
# fichier json avec les descriptions de données
# et effectue un échantillonnage stratifié
# =================================================
# args.input :
# score_label : list of str, labels des scores pris en compte dans cet échantillon
# score_id : list of int, position(s) des labels considérès dans le fichier de données initial
# class_label : list of str, labels des entrés 
# class_id : list of int, position(s) des entrées dans la fichier de données initial
# folder_name : str, nom du dossier parent contenant le fichier de données initial
# sample_name : str, nom donnée à cet échantillon
# file_name : str, fichier de données initial
# sep : str, separateur entre les informations dans le fichier de données initial
# header : bool, gère la présence d'un 'Header' i.e. une première ligne (dans le fichier de données initial) donnant la description des données
# frac : loat in [0, 1], proportion des données totales allouée à l'échantillon d'apprentissage
# output:
# None

    # Conversion du fichier brute dans le bon format
    changing_format(args.score_label, args.score_id, args.class_label, args.class_id, args.folder_name, args.sample_name, args.file_name, args.sep, args.header)
    # Création du fichier json avec la description des données
    data_desc(args.score_label, args.class_label, args.folder_name, args.sample_name)
    # Echantillonnage stratifié du jeu de donnée
    stratified_sampling(args.frac, args.folder_name, args.sample_name, args.class_label)
      
    return 

parser = argparse.ArgumentParser()

parser.add_argument("-fi", "--file_name", help="REQUIRED - str, fichier de données initial", required=True)
parser.add_argument("-sl", "--score_label", help="REQUIRED - list of str, labels des scores pris en compte dans cet échantillon",  nargs='+', required=True)
parser.add_argument("-si", "--score_id", help="REQUIRED - list of int, position(s) des labels considérès dans le fichier de données initial", nargs='+', required=True, type=int)
parser.add_argument("-cl", "--class_label", help="REQUIRED - list of str, labels des entrés", nargs='+', required=True)
parser.add_argument("-ci", "--class_id", help="REQUIRED - list of int, position(s) des entrées dans la fichier de données initial",  nargs='+', required=True, type = int)
parser.add_argument("-fo", "--folder_name", help="REQUIRED - str, nom du dossier parent contenant le fichier de données initial", required=True)
parser.add_argument("-sa", "--sample_name", help="REQUIRED - str, nom donnée à cet échantillon", required=True)
parser.add_argument("-s", "--sep", help="REQUIRED - str, separateur entre les informations dans le fichier de données initial", required=True)
parser.add_argument("-hd", "--header", action="store_true", help="None, si spécifiée présence d'un 'Header' i.e. une première ligne dans le fichier de données donnant la description des données")
parser.add_argument("-fr", "--frac", help="OPTIONAL - float in [0, 1], proportion des données totales allouée à l'échantillon d'apprentissage -- DEFAULT VALUE: 0.7", type=float, default=0.7)

args = parser.parse_args()

## Excution of the prog
if __name__ == "__main__":
    data_creator(args)