# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:35:36 2020

@author: DiSoJi

CLARIFICATION: Commonly a perceptron is considered a single layered neural network that returns a 1 or a 0.
                This is a single layer single neuron Perceptron model for academic purposes (that's why it uses back propagation)
"""


import numpy as np

#Auto descriptive name
class Perceptron():
    
    #Initialization method.
    #Here we define which activation function we are gonna use, initialize the weights, the bias and the learning rate
    def __init__(self, activation_function, input_size=3, bias=0, learning_rate = 1): #Some default values for simplicity
        #Uncomment in case you want to seed the initialization of the weights for debbuggin' purposes (or as i call it: play around, thwarthing plans)
        #np.random.seed(1) #Change 1 to the seed you wish
        self.weights = 2 * np.random.random((input_size, 1)) #3x1 matrix with random values
        self.activation_function = activation_function 
        self.bias = bias
        self.learning_rate = learning_rate
    
    #Run function, this returns the prediction for an input set
    #Dot product of input set with the weights and the addition of bias, called transfer function: 
    #sum(weight*input + bias) for every input, as linear function as you can see
    #Tranfer function is passed to the activation function, the result is our "prediction"
    def predict(self, inputs):
        #The bias value is multiplied by the amount of inputs since it is being added 
        #to the overall result of the sum of the w*x
        transfer_func = np.dot(inputs.astype(float), self.weights) + self.bias #This is equivalent to E(w*x) + b
        output = self.activation_function.run(transfer_func) #This is equivalent to o = f(E(w*x) + b)
        return output
    
    #Backpropagation weight correction algorithm
    #We calculate the outputs for the inputs (predictions)
    #Then we calculate the error for each training output with the predictions
    #We calculate the derivative of the activation functions with the predictions and multiply with the previous
    #This is multiplied by the learning rate
    #Then we calculate the dot procut with the inputs
    #This give us an error derivative - the change for the original weights (adjusments)
    def back_propagation(self,pred_error,inputs, outputs):
        return np.dot(inputs.T, self.learning_rate * pred_error * self.activation_function.run_derivative(outputs))

    #Training method
    #We get the predictions for the training inputs
    #We calculate the error with the predictions - the difference
    #Then we get the adjustments (using back propagation) and therefore adjust the weights
    def train(self, inputs, outputs, epochs):
        for epoch in range(epochs):
            perceptron_output = self.predict(inputs)
            
            prediction_error = outputs - perceptron_output
            
            weights_adjustments = self.back_propagation(prediction_error, inputs, perceptron_output)
            
            self.weights += weights_adjustments
            
