#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:12:35 2023

@author: tsesmat
"""  
## Import
# Librarys
import matplotlib.pyplot as plt
import time
import sys
import itertools
import os

# Scripts
from tools.converter import convert_strnb_to_tf, convert_tf_to_strnb
from optimisation.optim_num import optim_num
from optimisation.optim_bool import optim_bool
from tools.taboo_func import init_taboo_seniority, comp_seniority
from tools.deep_copy import deep_copy_dbl_nested_list
from tools.clean_useless import clean_useless


class TrainingTree:
# =============================================================================
# Training class that compute optimal network variables 
# =============================================================================
    def __init__(self, ini_netw_var, DATA, disp_loading, title, const_num_dict, cost_function):
    # =============================================
    # Init method
    # =============================================
    # input :
    # ini_netw_var : dict, dict of initial network's variables (see INIT.py for def each variables)
    # p: numpy array of float, Value of the absis where the sigmoid cross "p"
    # SENIORITY : int, Minimum number of networks that must be closed before an already exchanged Boolean can be exchanged again. 
    # ITER_GLOB : int, Number of iterations of gradient descent on all networks considered.
    # DATA : dict, dict of the datas divided in "scores" and "output"
    # NBREP_r : int, Nb of gradient descent iterations on each network during the exploration phase
    # NBREP_t : int, Nb of gradient descent iterations on the selected network during the exploitation phase# EPSILON : list of int, Step of the gradient descent algorithm
    # EPSILON : float, step of the gradient descent algorithm
    # CE_FOR_SLOPE : int, Calibrates the buffer length used to estimate (linearly) the CE slope.
    # disp_loading: bool, Activate or not the display of loadings messages
    # K : float, factor of the noramlisation criterion in the CE
    # title : str, Run title, used to label all saved outputs
     
        self.start = time.time()

        # Initialisation of the global variables
        self.ITER_GLOB = const_num_dict['ITER_GLOB'] 
        self.DATA = DATA
        self.NBREP_r = const_num_dict['NBREP_r']
        self.NBREP_t = const_num_dict['NBREP_t']
        self.EPSILON = const_num_dict['EPSILON']
        self.SENIORITY = const_num_dict['SENIORITY']
        self.CE_FOR_SLOPE = const_num_dict['CE_FOR_SLOPE']
        self.p = const_num_dict['p']
        self.disp_loading = disp_loading
        self.K = const_num_dict['K']
        self.title = title
        self.cost_function = cost_function
        # Initialisation of the first network in the dict
        self.desc_ini = [ini_netw_var['w_bool1'], ini_netw_var['w_bool2']]
        bool_seniority = init_taboo_seniority(self.desc_ini, self.SENIORITY)
        self.id_ini = convert_tf_to_strnb(self.desc_ini) # convert the boolean structure into a unique decimal nb
        
        self.w_num_ini, self.x_star_ini = ini_netw_var['w_num'], ini_netw_var['x_star']
        
        w_num_ini,\
        x_star_ini,\
        ce_evol_ini,\
        slope_ini = optim_num(ini_netw_var, self.p, self.NBREP_t, self.EPSILON,
                              self.DATA, [], self.CE_FOR_SLOPE, self.K, self.cost_function)
    
        self.rzo_dict = {self.id_ini: {'weights_num': w_num_ini,
                               'x_star': x_star_ini,
                               'ce_evol': ce_evol_ini,
                               'slope': slope_ini,
                               'state': 'open',
                               'bool_seniority': bool_seniority}}   # The tree is managed as a dict of a dict
        
        self.ghost_dict = {} # Dict for redundent network
        
        self.it_cum = self.NBREP_t

        # List of colors for the plot
        self.cols= ['aqua','black', 'blue', 'blueviolet', 'brown',
       'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
       'cornflowerblue', 'crimson', 'darkblue', 'darkcyan',
       'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
       'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
       'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray',
       'darkslategrey', 'darkviolet', 'deeppink',
       'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia',
       'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey',
       'hotpink', 'indianred', 'indigo', 'lawngreen', 'lightblue', 'lightcoral', 'lightgray', 'lightgreen',
       'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen',
       'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue',
       'lime', 'limegreen', 'magenta', 'maroon',
       'mediumaquamarine', 'mediumblue', 'mediumorchid',
       'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
       'mediumvioletred', 'midnightblue',
       'navy', 'olive',
       'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod',
       'palegreen', 'paleturquoise', 'palevioletred', 'peru', 'pink', 'plum',
       'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown',
       'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen'
       , 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey',
       'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
       'turquoise', 'violet', 'wheat', 'yellow', 'yellowgreen']*100

    def rzo_close(self, rzo_id):
    # =============================================
    # Neighboor discovering method
    # =============================================
    # input :
    # rzo_id : network we wanna discover the neighborhood. It's also it boolean configuration in decimal 
    #
    # output :
    # None
     
        # Initialize a temporary dict for just considered network
        neighb_rzo_dict = {}

        # get variables of the network n rzo_id
        weights_num = self.rzo_dict[rzo_id]["weights_num"]
        x_star = self.rzo_dict[rzo_id]["x_star"]
        bool_seniority = self.rzo_dict[rzo_id]["bool_seniority"]
        desc_bool = convert_strnb_to_tf(rzo_id, self.desc_ini) # convert the decimal nb into its equivalent in boolean structure
        
        # Computation of the neighborhood
        for i_co, couche in enumerate(desc_bool): # compute each neighboors network by swithching 1 by 1 each bool 
            nb_cell = len(couche)
            nb_weight = len(couche[0]) 

            for i_ce in range(nb_cell):
                """
                If a layer is unused in the next layer, we do not considere it
                """
                if (i_co == 0 and [desc_bool[1][x][i_ce] for x in range(len(desc_bool[1]))].count(True) == 0):
                    continue

                for i_w in range(nb_weight):
                    # We do not consider the network if we are in the wrong conditions
                    # We consider that we only have 2 layer!!!! 
                    """
                    Current condition are : 
                        At least one bool True on the last layer
                        Bool seniority superior to set seniority
                    """
                    if ((i_co == 1)\
                        and (desc_bool[i_co][i_ce].count(True) == 1)\
                        and  desc_bool[i_co][i_ce][i_w] == True)\
                        or bool_seniority[i_co][i_ce][i_w] < self.SENIORITY:  # finding the last layer and make sure taboo rules is respected
                        continue

                    # Boolean switching
                    desc_copy = deep_copy_dbl_nested_list(desc_bool)
                    desc_copy[i_co][i_ce][i_w] = not(desc_copy[i_co][i_ce][i_w])

                    # Special opening of disjunction layer
                    """
                    If a disjunction cell is set to true but the corresponding
                    conjunction cell is full of False, we also set the most promising
                    boolean in that conjunction cell to true.
                    If a disjunction cell is set to false and no other disjunction
                    cell uses its relative conjunction cell, we set each boolean
                    in it to False.
                    We do not update seniority in these two cases
                    """
                    if i_co == len(desc_bool) - 1:
                        if desc_copy[i_co][i_ce][i_w] and desc_copy[i_co-1][i_w].count(True) == 0:
                            neighb_rzo_dict = self.open_disj(neighb_rzo_dict, desc_copy, i_co, i_ce, i_w, weights_num, x_star)
                            continue # the new neighboor has already been added via open_disj
                        
                        elif (not desc_copy[i_co][i_ce][i_w]) and ([desc_copy[i_co][x][i_w] for x in range(nb_cell)].count(True) == 0):
                            desc_copy = self.close_disj(desc_copy, i_co, i_w)
                        
                    id_ = convert_tf_to_strnb(desc_copy)

                    # Initializing and saving the network in the dict
                    if id_ not in self.rzo_dict and id_ not in self.ghost_dict: # if the just-build network isnt already in the dict, we add it 
                        netw_var = {'w_num': weights_num, 'x_star': x_star, 'w_bool1': desc_copy[0], 'w_bool2': desc_copy[1]}
                        
                        weights_num_son, x_star_son, ce_evol_son, slope =\
                            optim_num(netw_var, self.p, self.NBREP_r, self.EPSILON,
                                      self.DATA, [], self.CE_FOR_SLOPE, self.K, self.cost_function)
                        
                        neighb_rzo_dict[id_] = {"weights_num": weights_num_son,
                                                "x_star": x_star_son,
                                                "ce_evol": ce_evol_son,
                                                "slope": slope,
                                                 "state": "open",
                                                 "switch_coord": (i_co, i_ce, i_w)}

                        self.it_cum += self.NBREP_r

        # Selec the one network that will be added to rzo_dict
        rem_iter = self.ITER_GLOB - self.it_cum
        optim_neighb = optim_bool(neighb_rzo_dict, rem_iter)
        
        # Add the network (and compute its taboo_seniority)
        self.add_network_to_dict(rzo_id, neighb_rzo_dict[optim_neighb], optim_neighb)
        self.reset_unactive_bool(optim_neighb)
        
        self.rzo_dict[rzo_id]["state"] = "close" # the network no rzo_id is now close
        self.rzo_dict[rzo_id].pop("bool_seniority") # so we can delete its list of bool_seniority

        return

    def open_disj(self, neighb_rzo_dict, desc_copy, i_co, i_ce, i_w, weights_num, x_star):
    # =============================================
    # Method that put the most promising conjunction
    # wieght to True if a disjunction cell as been set
    # to True put none of the wieghts of the 
    # corresponding conjunction cell is True
    # =============================================
    # input :
    # neighb_rzo_dict : network we wanna discover the neighborhood. It's also it boolean configuration in decimal 
    # desc_copy : double nested list of bool, boolean configuration of the network whose disjunction has been set to True
    # i_co : int, index of the layer
    # i_ce : int, index of the cell in the layer
    # i_w : int, index of the weight in the cell
    # weights_num: numpy array of float, steepness of the slope of the sigmoid functions 
    # x_star: numpy array of float, Values of the abscissa where the two sigmoids cross
    # 
    # output :
    # neighb_rzo_dict : dict, dict of legal neigboorhood of a network

        # Exploration of a deeper neighborhood to get a legal one
        for i_w_conj in range(len(desc_copy[i_co-1][i_w])):
            new_desc = deep_copy_dbl_nested_list(desc_copy)
            new_desc[i_co-1][i_w][i_w_conj] = True
            id_ = convert_tf_to_strnb(new_desc)

            # Initializing and saving the network in the dict
            if id_ not in self.rzo_dict and id_ not in self.ghost_dict: # if the just-build network isnt already in the dict, we add it 
                netw_var = {'w_num': weights_num, 'x_star': x_star, 'w_bool1': desc_copy[0], 'w_bool2': desc_copy[1]}
                
                weights_num_son, x_star_son, ce_evol_son, slope =\
                    optim_num(netw_var, self.p, self.NBREP_r, self.EPSILON,
                              self.DATA,[], self.CE_FOR_SLOPE, self.K, self.cost_function)
                
                neighb_rzo_dict[id_] = {"weights_num": weights_num_son,
                                        "x_star": x_star_son,
                                        "ce_evol": ce_evol_son,
                                        "slope": slope,
                                        "state": "open",
                                        "switch_coord": (i_co, i_ce, i_w)}
                
                self.it_cum += self.NBREP_r

        return neighb_rzo_dict

    def close_disj(self, desc_copy, i_co, i_ce):
    # =============================================
    # Method that set to False every bool of a conjunction layer 
    # if none of the disjunction layer are using it
    # =============================================
    # input :
    # desc_copy : double nested list of bool, boolean configuration of the network whose disjunction has been set to True
    # i_co : int, index of the layer
    # i_ce : int, index of the cell in the layer
    # 
    # output :
    # new_desc : double nested list of bool, boolean configuration of the network with conjunction full set to False
    
        new_desc = deep_copy_dbl_nested_list(desc_copy)
        new_desc[i_co-1][i_ce] = [False] * len(new_desc[i_co-1][i_ce]) 
		
        return new_desc 
	
    def reset_unactive_bool(self, id_):
    # =============================================
    # Method that reset num variables if they 
    # are unused in the network
    # =============================================
    # input :
    # id_ : int, id of a network and also its bool config in decimal
    # 
    # output :
    # None
        desc_bool = convert_strnb_to_tf(str(id_), self.desc_ini)
        
        for j in range(len(desc_bool[0][0])):
            a = [desc_bool[0][i][j] for i in range(len(desc_bool[0]))]

            if a.count(True) == 0:
                nb_desc = len(desc_bool[0][0]) / len(self.x_star_ini)
                i_c = int(j//nb_desc)
                i_w = int(j%nb_desc)
                self.rzo_dict[id_]['weights_num'][i_c][i_w] = self.w_num_ini[i_c][i_w]

        for j in range(int(len(desc_bool[0][0])/2)):
            if [desc_bool[0][i][2*j:2*j+1] for i in range(len(desc_bool[0]))].count(True) == 0:
                self.rzo_dict[id_]['x_star'][j] = self.x_star_ini[j]
        return

    def add_network_to_dict(self, parent_rzo_id, rzo, id_rzo):
    # =============================================================================
    # Method that add a network to the main dict
    # =============================================================================
    # input :
    # parent_rzo_id : int, id of a parent network and also its bool config in decimal
    # rzo : dict, caracterics of the network 
    # id_rzo : int, id of the network and also its bool config in decimal
    # 
    # output :
    # None

        (i_co, i_ce, i_w) = rzo["switch_coord"]
        rzo.pop('switch_coord') # delete key/value since it is useless now

        parent_seniority = self.rzo_dict[parent_rzo_id]["bool_seniority"]
        rzo["bool_seniority"] = comp_seniority(parent_seniority, i_co, i_ce, i_w)
        
        self.rzo_dict[id_rzo] = rzo

        return

    def attrib_color(self, rzo):
    # =============================================================================
    # Method that assigns a specific color to a relevant network.
    # Relevant network means that the network have been the optimal one at least one time
    # There is more than hundred colors that are attributed periodicly
    # =============================================================================
    # rzo: dict, caracterics of the network
            
        rzo['color'] = self.cols[0]
        self.cols.pop(0)

        return


    def plot(self, rzo):
    # =============================================================================
    # Mthod that plots the CE for the considered network (rzo)
    # according to the number of iterations. 
    # We plot it as a droite between first and last value
    # =============================================================================
    # rzo: dict, caracterics of the network
    
        lx = [self.it_cum - self.NBREP_t   + 1, self.it_cum] # Absissa
        ly = rzo["ce_evol"][-2:] # Ordered (?)

        col = rzo["color"]

        plt.plot(lx, ly, color=col)

        return 

    def train(self):
    # =============================================================================
    # Method that return booth numerical and boolean trained variables
    # =============================================================================
    
        # Initialsiation
        id_nfn = self.id_ini     
        po = 0.1
        while self.it_cum < self.ITER_GLOB:
            

            
            # Ploting the current evolution
            if not ('color' in self.rzo_dict[id_nfn]):
                self.attrib_color(self.rzo_dict[id_nfn])
            self.plot(self.rzo_dict[id_nfn])
            
            # Get the values from the dict for the numerical optimization 
            desc_opt = convert_strnb_to_tf(id_nfn, self.desc_ini)
            netw_var = {'w_num': self.rzo_dict[id_nfn]['weights_num'], 'x_star':  self.rzo_dict[id_nfn]['x_star'], 'w_bool1': desc_opt[0], 'w_bool2': desc_opt[1]}
            ce_evol = self.rzo_dict[id_nfn]['ce_evol']

            # Numerical optimization
            self.rzo_dict[id_nfn]['weights_num'],\
            self.rzo_dict[id_nfn]['x_star'],\
            self.rzo_dict[id_nfn]['ce_evol'],\
            self.rzo_dict[id_nfn]['slope'] = optim_num(netw_var, self.p, self.NBREP_r, self.EPSILON,
                                                       self.DATA, ce_evol, self.CE_FOR_SLOPE, self.K, self.cost_function)
            self.it_cum += self.NBREP_t

            # Open choseen network if it is close
            if self.rzo_dict[id_nfn]["state"] == "open" :
                self.rzo_close(id_nfn)

            # Removing ghost every 10% of iterations
            if self.it_cum/self.ITER_GLOB > po:
                self.remove_ghost()
                po += 0.1

            # Boolean optimization
            rem_iter = self.ITER_GLOB - self.it_cum
            id_nfn = optim_bool(self.rzo_dict, rem_iter)
            
            if self.disp_loading: 
                sys.stdout.write("\rAvancé: {0} %    ".format(round((self.it_cum/self.ITER_GLOB) * 100, 1)))
                sys.stdout.flush()
        path = os.getcwd()+'/output_temp/graph/'
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + str(self.title)+'.png')
        plt.close()

        # Get the id of the network who gave the min C-E
        rzo_min = 10
        for rzo in self.rzo_dict: 
            if self.rzo_dict[rzo]['ce_evol'][-1] < rzo_min:
                rzo_min = self.rzo_dict[rzo]['ce_evol'][-1]
                id_min = rzo

        # Simplifie the best network
        w1_optim = self.rzo_dict[id_min]['weights_num']
        x_star_optim = self.rzo_dict[id_min]['x_star']
        desc_bool = convert_strnb_to_tf(id_min, self.desc_ini)
        opt_netw_var = {"w_num": w1_optim,
                        "x_star": x_star_optim,
                        "w_bool1": desc_bool[0],
                        "w_bool2": desc_bool[1]
                        }

        opt_netw_var['w_bool1'], opt_netw_var['w_bool2'] = clean_useless(opt_netw_var, self.p, self.K, self.DATA, "better", self.cost_function)  

        printer_result = f"Running time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start))}"
        printer_result += f"\nNumber of networks opened: {str(len(self.rzo_dict) + len(self.ghost_dict))} dont {str(len(self.ghost_dict))} doublon(s)"

        return opt_netw_var, printer_result


    def remove_ghost(self):
    # =============================================================================
    # Method that removed network wich are equivalent (they have the same reduced)
    # boolean config. 
    # We last the most optimised of same networks
    # Redundent network are due to unoptimised exploration
    # =============================================================================

        dict_clean = {}
        
        # Reducing every config to it's minimum.
        for rzo_id in self.rzo_dict:
            if self.rzo_dict[rzo_id]["ce_evol"] != float('inf'):
                desc_bool_id = convert_strnb_to_tf(rzo_id, self.desc_ini)
                
                netw_var = {'w_num': self.rzo_dict[rzo_id]['weights_num'],
                            'x_star': self.rzo_dict[rzo_id]['x_star'],
                            'w_bool1': desc_bool_id[0],
                            'w_bool2': desc_bool_id[1]}
                
                desc_bool1, desc_bool2 = clean_useless(netw_var, self.p, self.K, self.DATA, "equal", self.cost_function)
                
                dict_clean[rzo_id] = convert_tf_to_strnb([desc_bool1, desc_bool2])
            
        
        for rzo_id_clean, rzo_id2_clean in itertools.product(dict_clean, dict_clean):
            
            if dict_clean[rzo_id_clean] == dict_clean[rzo_id2_clean]: # Comparison of every network reduced config
                if self.rzo_dict[rzo_id_clean]["ce_evol"][-1] < self.rzo_dict[rzo_id2_clean]["ce_evol"][-1]\
                    and self.rzo_dict[rzo_id2_clean]["state"] == "close": # removal of the less optimized of the two, provided that its neighborhood has already been discovered  
                        self.ghost_dict[rzo_id2_clean] = None # Adding the worst to a ghost dict of "removed but considered networks"
                    
                elif self.rzo_dict[rzo_id_clean]["ce_evol"][-1] > self.rzo_dict[rzo_id2_clean]["ce_evol"][-1]\
                    and self.rzo_dict[rzo_id_clean]["state"] == "close": # removal of the less optimized of the two, provided that its neighborhood has already been discovered 
                        self.ghost_dict[rzo_id_clean] = None # Adding the worst to a ghost dictionary of “networks removed but considered”

        # Removed ghost networks from main dict
        for rzo_ghost in self.ghost_dict:
            if rzo_ghost in self.rzo_dict:
                self.rzo_dict.pop(rzo_ghost)

        return 