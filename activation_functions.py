# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:24:25 2020

@author: DiSoJi


This file is to add activation functions as required for easy modularity
"""

import numpy as np

#Wrapper class for sigmoid actiation function and it's derivative
class Sigmoid_Func():
    
    def run(self, x):
        return 1 / (1 + np.exp(-x)) 
    
    def run_derivative(self,x):
        sig = self.run(x)
        return sig * (1 - sig)

#ReLu function. To Show vanishing gradient problem in [0,1] range described in main file
#Depending on the initial weights it takes more or less epochs but always dies.
#With a good balance between learning rate, epochs and good initial weights it can converge to the solution.
#Or at least be kinda close
class ReLu_Func():  #Still needs some improving
    
    def __init__(self,leakage = 0):
        self.leakage = leakage
        
    def run(self,x):
    
        return np.where(x > 0, x, x * self.leakage) #Returns results of parametrized ReLu( if leakage is 0 we have normal ReLu)
        
    def run_derivative(self,x): #Using different approach as run for time measuring purposes, add them if considered necesary
        out = np.copy(x) #Copy to prevent pass-by-reference issues
        out[out<=0] = self.leakage #Each element is compared with 0, if it's equal or less then becomes 0
        out[out>0] = 1 #Each element is compared with 0, if it's greater it becomes 1
        return out

#final = np.full((x.shape[0],1),0) #Coulum with the same amount of rows, filled with zeros for ReLu function comparison    
#out = np.array(np.maximum(x,final)) #Previously used as ReLu, seved here for later reference
        

#Softmax functions still need testing
class Softmax_Func(): #Softmax function to show how it has the inf problem that throws NaN
    
    def run(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def run_derivative(self,x): #Not intended to be used so it's left blank
        return 0
    
class Softmax_Stable_Func(): #Softmax with balance. It shifts the issue away *ba *dum *tss
    def run(self, x):
        shiftx = x - np.max(x) #Obtaining the max value we can shift the values to the "left" by that value, so we can obtain a negative domain
        exps = np.exp(shiftx) #Calculates the exp of the shifted values
        return np.divide(exps, np.sum(exps)) #Good'ol softmax
    
    def run_derivative(self,x): #Not yet implemented. 
        return 0
    