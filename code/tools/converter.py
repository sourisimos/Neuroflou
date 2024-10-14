#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:58:29 2023

@author: tsesmat
"""

def convert_strnb_to_tf(str_nb, desc_bool):
# =============================================
# Function that convert a number to its bool
# config. To do so we use binary transcription 
# and then 0 means False et 1 means True
# =============================================
# str_nb : str(int), id of a network as an int converted into string 
# desc_bool : double netsted list of bool, bool configuration of reference netw
#             used as structure patern for bool config
#
# output:
# new_desc : double netsted list of bool, the bool config relative to the id
    
    nb = int(str_nb)
    str_bin = bin(nb)[2:][::-1] # binary conversion
    new_desc = []

    # we go 1 by 1 trought every bit and set it to T/F in the structure
    for ll_weight in desc_bool:
        nb_out = len(ll_weight)
        nb_in = len(ll_weight[0])
        l_new_desc = []

        for i in range(nb_out):
            len_l_bool = len(str_bin)
            bits = [bit == '1' for bit in str_bin[:nb_in]]

            if len_l_bool < nb_in:
                bits.extend([False] * (nb_in - len_l_bool))

            l_new_desc.append(bits)
            str_bin = str_bin[nb_in:]

        new_desc.append(l_new_desc)

    return new_desc


def convert_tf_to_strnb(desc_bool):
# =============================================
# Function that convert a bool config into its 
# unique str(int) value
# =============================================
# desc_bool : double netsted list of bool, bool configuration
#
# output:
# key_nb : str(int), unique decimal translated value a the bool config

    nb = sum(b * 2**i for i, b in enumerate(sum(sum(desc_bool, []), [])))
    key_nb = str(nb)
    return key_nb

