# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:34:46 2020

@author: DiSoJi
"""
import numpy as np
import perceptron_model as perceptron
import activation_functions as act_funcs


sigmoid_func = act_funcs.Sigmoid_Func() #Define out activation function
neural_network = perceptron.Perceptron(sigmoid_func, bias=0, input_size=3, learning_rate=1) #Define the perceptron with the act.func and the number of inputs

#Some training inputs
training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

#Their respective output 
training_outputs = np.array([[0,1,1,0]]).T #Transposed to have it as a single coulum matriz with 4 rows

#Begin training with the data batch and define number of epochs
neural_network.train(training_inputs, training_outputs, 1000) 

#Let's see the final weights! Yay!
print(neural_network.weights)

#Let's input a new input to test the Neuron
i1 = str(input("Input 1: "))
i2 = str(input("Input 2: "))
i3 = str(input("Input 3: "))


while(True): #ctrl+C if you want to exit
#In case you make some mistake with the inputs I added a try. Thank me later (probably at 3:00 am or something like that)
    try:
        
        print("Perceptrons result: ")
        result = neural_network.predict(np.array([i1, i2, i3]))
        prediction = 1 if result > 0.5 else 0
        print(prediction)
        print("Exact Output: ")
        print(result)
        
    except:
        print("Stop fooling around and input valid numbers")
    
