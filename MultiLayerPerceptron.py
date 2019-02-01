#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-
"""
    Created on Fri Nov 23 16:24:27 2018
    @author: Sina Ahmadi  (ahmadi.sina@outlook.com)
"""
from __future__ import division
from random import randrange
from random import random
from random import shuffle
import math

class Perceptron:
    def __init__(self, n_epochs, learning_rate, n_hidden, activation_type, n_folds=None, train_test_prop=None):
        """ Class constructor"""
        self.n_epochs = n_epochs
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.train_test_prop = train_test_prop
        self.activation_type = activation_type
        self.network = list()
        
    def run(self, dataset):
        """
        The main function that runs the MLP.
        Gets the dataset, number of the folds (in the case of Cross-Validation) or the proportion of train-set/test-set
        """
        scores = list()
        
        # Cross validation
        if self.n_folds != None and self.train_test_prop == None:
            # Create folds of dataset
            folds = self.fold_dataset(dataset)
            for fold in folds:
                # Remove one fold each time and use it as test set
                train_set = list(folds)
                train_set.remove(fold)
                train_set = sum(train_set, [])
                test_set = list()
                for row in fold:
                    row_copy = list(row)
                    test_set.append(row_copy)
                    row_copy[-1] = None
                # Train the model and get its prediction for the test set (the test fold)  
                predicted = self.stochastic_gradient_descent(train_set, test_set)
                actual = [row[-1] for row in fold]
                # Evaluate the accuracy of the prediction
                accuracy = self.evaluate(actual, predicted)
                scores.append(accuracy)
        
        # Dataset split
        elif self.train_test_prop != None and self.n_folds == None:
            # Split the dataset into 2/3 train set and 1/3 test set
            for random_index in range(10):
                # Shuffle the dataset
                shuffle(dataset)
                train_set = [dataset[:int(len(dataset)*self.train_test_prop)]]
                train_set = sum(train_set, [])
                
                test_set = dataset[int(len(dataset)*self.train_test_prop):]
                
                # Train the model and get its prediction for the test set 
                predicted = self.stochastic_gradient_descent(train_set, test_set)
                actual = [row[-1] for row in test_set]
                # Evaluate the accuracy of the prediction
                accuracy = self.evaluate(actual, predicted)
                scores.append(accuracy)
        
        else:
            raise Exception("Running case not recognized!")
        
        return scores
    
    def initializer(self, n_inputs, n_outputs):
        """ Gets the number of the input neurons and output neurons of the network,
        and initializes the weight vectors for each layer (as a dictionary called "weights". More dictionaries will be added to the network later.)
        The created network is a list defined in the constructor of the class, therefore accessible to all functions.
        """
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(self.n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(self.n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)
            
    def stochastic_gradient_descent(self, train_set, test_set):
        """This is where the network is trained. First the network is initialized by the number of the input neurons, i.e., number of features, and 
        number of the output neurons (classes to be predicted. Counting these two numbers is done automatically. Therefore, datasets with any number 
        of classes and features can be trained. 
         """
        self.network = list()
        n_inputs = len(train_set[0]) - 1
        n_outputs = len(set([row[-1] for row in train_set]))
        self.initializer(n_inputs, n_outputs)
        # Train the model, calculate the error and update the weights
        self.trainer(train_set, n_outputs)
        predictions = list()
        # For each test instance, get the predition of the model
        for row in test_set:
            prediction = self.predict(row)
            predictions.append(prediction)
        return predictions
        
    def fold_dataset(self, dataset):
        """ If the cross-validation option is evoked (i.e., n_folds != None), this function splits the dataset into the number of folds"""
        dataset_split = list()
        fold_size = int(len(dataset) / self.n_folds)
        for i in range(self.n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(list(dataset)))
                fold.append(list(dataset).pop(index))
            dataset_split.append(fold)
        return dataset_split
    
    def trainer(self, train_set, n_outputs):
        """ Gets the training set and the number of output neurons, calculated the feed-forward network and the error in comparison to the expected values
        and updates the weights according to the update rule. This process is iterated as many as the number of the epochs."""
        for epoch in range(self.n_epochs):
            for row in train_set:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                self.backpropagation_error(expected)
                self.update_weights(row)
    
    def activation_function(self, x): 
        """ Three non-linear functions are defined as the actiation function for the output layer"""
        if self.activation_type == "Sigmoid":
            return 1.0 / (1.0 + math.exp(-x))
        elif self.activation_type == "Tanh":
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        elif self.activation_type == "ReLU":
            if x < 0:
                return 0
            else:
                return x
        else:
            raise Exception("Activation function not defined.")
    
    def activation_function_derivative(self, x):
        """Gets an input and calculated the output of the derivative of the activation function for the gradient decent"""
        if self.activation_type == "Sigmoid":
            return x * (1.0 - x)
        elif self.activation_type == "Tanh":
            return 1.0 - self.activation_function(x)
        elif self.activation_type == "ReLU":
            if x < 0:
                return 0
            else:
                return 1.0
        else:
            raise Exception("Activation function not defined.")
        
    def weigh_layer(self, weights, inputs):
        """Calculates the overall weight of a layer by multiplying the input values and the weight vector"""
        layer_weight = weights[-1]
        for i in range(len(weights)-1):
            layer_weight += weights[i] * inputs[i]
        return layer_weight

    def forward_propagate(self, inputs):
        """ The full process of the feed-forward network. Calculates the weight of the layer, then passes the output into the acitvation function. 
        The output of the feed-forward network for each layer is added as a dictionary "output" to the network."""
        for layer in self.network:
            new_inputs = list()
            for neuron in layer:
                layer_weight = self.weigh_layer(neuron['weights'], inputs)
                neuron['output'] = self.activation_function(layer_weight)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def backpropagation_error(self, expected):
        """This function is called in each epoch to calculate the back-propagation error which is different in the output layer and in the hidden layer.
        When in the output layer, the error is the difference of the expected output and the prediction. The error values are then propagated 
        in the network in the backward direction from the output layer to the hidden layer. The error of each neuron is calculated as (expected value - predicted value) * the derivative of the output.
        The error in the hidden layer for each neuron is calculated as (weight * error in the neuron) * the derivative of the output.
        All the errors are saved in the delta dictionary in the network.  
        """
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            # Check i not as the last layer
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.activation_function_derivative(neuron['output'])

    def update_weights(self, row):
        """For a given data instance, after the error is calculated (in delta) and the feed-forward (in output), the weights are updated following this rule:
        error * learning rate * input. For each neuron in a layer that rule is applied and accumulated. """
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.learning_rate * neuron['delta']
            
    def evaluate(self, actual, predicted):
        """Gets the expected outputs and the predictions, and calculates the accuracy by dividing the commonn cases over the whole number of instances."""
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def predict(self, row):
        """Given a new data instance, calculates the output of the trained model.
        In the case of the multi-class prediction, the prediction is a vector of the length of the classes. Therefore, for each prediction the highest value is returned and its index is 
        used as the class of the input. 
        """
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))


    
