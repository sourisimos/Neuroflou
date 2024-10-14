#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:11:04 2023

@author: tsesmat
"""
## Import
# libraries
import copy

def deep_copy_dbl_nested_list(l):
# =============================================
# Function that make a deep copy of a double
# nested list
# =============================================
# l : double nested list
#
# output:
# cop : double nested list, deep copy of l

    cop = copy.deepcopy(l)
    return cop

def deep_copy_spl_nested_list(l):
# =============================================
# Function that make a deep copy of a simple
# nested list
# =============================================
# l : simple nested list
#
# output:
# cop : simple nested list, deep copy of l
    return copy.deepcopy(l)


"""
# Version sans import légèrement moins rapide mais faisant la même chose
def deep_copy_dbl_nested_list(l):
# =============================================
# Function that make a deep copy of a double
# nested list
# =============================================
# l : double nested list
#
# output:
# ret : double nested list, deep copy of l

    ret = [[cell[:] for cell in layer]for layer in l]
    return ret

def deep_copy_spl_nested_list(l):
# =============================================
# Function that make a deep copy of a simple
# nested list
# =============================================
# l : simple nested list
#
# output:
# ret : simple nested list, deep copy of l
    ret = [cell[:] for cell in l]
    return ret
"""