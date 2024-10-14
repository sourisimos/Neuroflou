#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:41:53 2023

@author: tsesmat
"""
## Import 
# Libraries
import numpy as np

# Scripts
from nfn_structure.FuzzificationLayer import FuzzyficationLayer
from nfn_structure.AndLayer import AndLayer
from nfn_structure.OrLayer import OrLayer
from tools.math_function import squared_error, cross_entropy

class NeuroFuzzyNetwork:
# =============================================
# NeuroFuzzy Network class
# =============================================

    def __init__(self, ini_netw_var, p, K, cost_function):
    # =============================================
    # init method
    # =============================================
    # input :
    # ini_netw_var : dict, dict of network initial variables (see INIT.py for def each variables)
    # p: numpy array of float, Value of the abscissa where the sigmoid crosses "p".
    # K : float in [0, 1], Crossover value of the two sigmoids representing the large and small descriptors for a given score
        
        self.ll1_weight, self.lx_star, self.ll2_weight, self.ll3_weight = ini_netw_var.values()
        self.p = p
        self.K = K

        self.output_size = len(self.ll3_weight)
        self.input_size = len(self.ll1_weight)
        self.cost_function = cost_function
        """
        # self.descriptors_nb = len(ll1_weight) 
        # piste : on veut pouvoir avoir un tableau avec le nb pour chaque input
        # (nb de descripteur potentiellement variable)
        """
        
        # Initialiation of each layer
        self.fuzzyfication = FuzzyficationLayer(self.ll1_weight, self.lx_star, self.p)
        self.conjunction = AndLayer(self.ll2_weight)
        self.disjunction = OrLayer(self.ll3_weight)

        self.grad_cum_w = 0
        self.grad_cum_x = 0
        self.grad_cum_w_num = 0
        self.grad_cum_x_num = 0

    
    def activate(self, l_input_, l_target_=None, train=False, compute_grad_num=False):
    # =============================================
    # Activation function method
    # =============================================
    # input :
    # l_input_ : list of floats in [0, 1], list of scores for a single measure
    # l_target_ : list of elements in {0, 1}, one hot vectors of the corresponding measures
    # train: bool, activate the training phase or not
    # compute_grad_num: bool, compute the num gradient or not
    # 
    # output :
    # output : list of float in [0, 1], prevision of class membership after scores have passed through the NFN
        
        # NFN pass forward
        out_temp1 = self.fuzzyfication.activate(l_input_, train)
        out_temp2 = self.conjunction.activate(out_temp1, train)
        self.output = self.disjunction.activate(out_temp2, train)
        
        # Learning phase
        if train:
            
            next_sensib1 = 2 * (np.array([(out - target)\
                                          for target, out in zip(l_target_, self.output)])
                                + self.K * ( np.sum(self.output) - 1))
                
            next_sensib2 = self.disjunction.back_propagation(next_sensib1)
            next_sensib3 = self.conjunction.back_propagation(next_sensib2)
            
            # Gradient used in steepest descent algorithme is the mean on the batch
            self.grad_cum_w +=\
                self.fuzzyfication.back_propagation(next_sensib3)[0]
            self.grad_cum_x +=\
                self.fuzzyfication.back_propagation(next_sensib3)[1]
            
            # The num gradient is used to ensure that the theoretical gradient is correctly implemented.
            if compute_grad_num:
                self.grad_cum_w_num += self.addGradNum(l_input_, l_target_)[0]
                self.grad_cum_x_num += self.addGradNum(l_input_, l_target_)[1]

        return self.output

    def cost(self, output, ly):
        # =============================================
        # Network's cost function method
        # =============================================
        # input :
        # output : list of floats in [0, 1], prevision of class membership after scores have passed through the NFN
        # l_target_ : list of elements in {0, 1}, real observed one hot vectors
        #
        # output :
        # ce : list of int, prevision of class membership after scores have passed through the NFN
        
        # Ajouter au fur et à mesure les autre tuype de fonctions cout
        if self.cost_function == 'squared_error':    
            ce = squared_error(output, ly, self.K)
        elif self.cost_function == 'cross_entropy':
            ce = cross_entropy(output, ly, self.K)
        return ce

    def addGradNum(self, l_input_, l_target_):

        grad_w_num = np.zeros((3, 2))
        grad_x_num = np.zeros(3)

        l_w_av = np.copy(self.ll1_weight)
        l_w_ap = np.copy(self.ll1_weight)
        l_x_av = np.copy(self.lx_star)
        l_x_ap = np.copy(self.lx_star)

        for i in range(3):
            l_x_av[i] -= 0.0000005
            l_x_ap[i] += 0.0000005
            self.fuzzyfication =\
                FuzzyficationLayer(self.ll1_weight, l_x_ap, self.p)
            out_temp1_ap = self.fuzzyfication.activate(l_input_)
            out_temp2_ap = self.conjunction.activate(out_temp1_ap)
            output_ap = self.disjunction.activate(out_temp2_ap)
            # output_ap = self.normalization.activate(out_temp3_ap)

            self.fuzzyfication =\
                FuzzyficationLayer(self.ll1_weight, l_x_av, self.p)
            out_temp1_av = self.fuzzyfication.activate(l_input_)
            out_temp2_av = self.conjunction.activate(out_temp1_av)
            output_av = self.disjunction.activate(out_temp2_av)
            # output_av = self.normalization.activate(out_temp3_av)

            grad_x_num[i] +=\
                (self.cost(output_ap, l_target_) -
                 self.cost(output_av, l_target_)) / 0.000001

            for j in range(2):
                l_w_av[i, j] -= 0.0000005
                l_w_ap[i, j] += 0.0000005


                self.fuzzyfication =\
                    FuzzyficationLayer(l_w_ap, self.lx_star, self.p)
                out_temp1_ap = self.fuzzyfication.activate(l_input_)
                out_temp2_ap = self.conjunction.activate(out_temp1_ap)
                output_ap = self.disjunction.activate(out_temp2_ap)
                # output_ap = self.normalization.activate(out_temp3_ap)

                self.fuzzyfication =\
                    FuzzyficationLayer(l_w_av,  self.lx_star, self.p)
                out_temp1_av = self.fuzzyfication.activate(l_input_)
                out_temp2_av = self.conjunction.activate(out_temp1_av)
                output_av = self.disjunction.activate(out_temp2_av)
                # output_av = self.normalization.activate(out_temp3_av)

                grad_w_num[i, j] +=\
                    (self.cost(output_ap, l_target_) -
                     self.cost(output_av, l_target_)) / 0.000001

            

                l_w_av = np.copy(self.ll1_weight)
                l_w_ap = np.copy(self.ll1_weight)
            l_x_av = np.copy(self.lx_star)
            l_x_ap = np.copy(self.lx_star)

        return grad_w_num, grad_x_num


    def grad(self, size, compute_grad_num=False):
        """
        A automatiser pour donner une sortie de taille adaptative au nb de
        parametre à modifier
        """
        # =============================================
        # Network gradient computation method
        # =============================================
        # input :
        # size : int, size of the sample
        # compute_grad_num: bool, compute the num gradient or not
        #
        # output :
        # dict_grad : dict of np array matrix of float, gradient of num weights and biais relative to output

        grad_weight = self.grad_cum_w/size
        grad_x_star = self.grad_cum_x/size

        grad_weight = np.reshape(grad_weight, (-1, 2)) # A automatiser

        if compute_grad_num:
            grad_weight_num = self.grad_cum_w_num/size
            grad_x_num = self.grad_cum_x_num/size
            grad_weight_num = np.reshape(grad_weight_num, (-1, 2)) # A automatiser

            print('\nGradX: \n', grad_x_star)
            print('\nGradXnum: \n', grad_x_num)
            print('\nTaux weight: \n',
                  (grad_weight - grad_weight_num)/grad_weight)
            print('\nTaux x_star: \n', (grad_x_star - grad_x_num)/grad_x_star)

        dict_grad = {"grad_w": grad_weight, "grad_x": grad_x_star}
        
        return dict_grad

    def desc_grad(self, epsilon, size, compute_grad_num):
        """
        A automatiser pour donner une sortie de taille adaptative au nb de
        parametre à modifier
        """
        # =============================================
        # Network steepest_descent method
        # =============================================
        # input :
        # epsilon : float, step of the gradient descent algorithm
        # size : int, size of the sample
        # compute_grad_num: bool, compute the num gradient or not

        grad = self.grad(size, compute_grad_num)
        
        self.ll1_weight = self.ll1_weight - epsilon * grad["grad_w"]
        self.lx_star = self.lx_star - epsilon * grad["grad_x"]
        
        # Projection on [ 0 ; 1] to force the x_star value to be within the membership rate definition interval
        for i_x in range(len(self.lx_star)): 
            if self.lx_star[i_x] <= 0:
                self.lx_star[i_x] = 0
            elif self.lx_star[i_x] >= 1:
                self.lx_star[i_x] = 1
                
