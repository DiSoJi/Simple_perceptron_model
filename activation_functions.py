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
        return self.run(x) * (1 - self.run(x))