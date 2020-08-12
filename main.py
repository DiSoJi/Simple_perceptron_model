# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:34:46 2020

@author: DiSoJi
"""
import numpy as np
import perceptron_model as perceptron
import activation_functions as act_funcs


"""
print("Start")
#This area is in case you want to test something before starting everything
#Just erase the semicolons
input("End")

"""
sigmoid_func = act_funcs.Sigmoid_Func() #Define activation function as sigmoid
relu_func = act_funcs.ReLu_Func() #Define activation function as ReLu

#Define the perceptron with the act.func and the number of inputs
neural_network = perceptron.Perceptron(relu_func, bias=0, input_size=3, learning_rate=1) 

#Some training inputs, for a problem where the first value of each input is the one that dictates the output,
#so should be weighted the most (and in such a way that overshadows the others)
#This can be considered a logistic regretion model
#Or even a binary classification problem
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

#Their respective output 
training_outputs = np.array([[0,1,1,0]]).T #Transposed to have it as a single coulum matriz with 4 rows

#Begin training with the data batch and define number of epochs

epochs = 10000
for epoch in range(epochs): #Training loop removed from the neuron's model. This makes training an network possible
    neural_network.train(training_inputs, training_outputs) 

#Let's see the final weights! Yay!
print("Final Weights")
print(neural_network.weights)
while(True): #ctrl+C if you want to exit
    #Let's input a new input to test the Neuron
    #Did them separated because i couldn't be bothered apparently
    i1 = str(input("Input 1: "))
    i2 = str(input("Input 2: "))
    i3 = str(input("Input 3: "))

    #In case you make some mistake with the inputs I added a try. Thank me later (probably at 3:00 am or something like that)
    try:
        
        result = neural_network.predict(np.array([i1, i2, i3]))
        print("Exact Output: ")
        print(result) #If using sigmoid you can compare with 0.5, where above it is 1 and bellow is 0
            
    except:
        print("Stop fooling around and input valid numbers")
    
