#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-
"""
    Created on Fri Nov 23 16:27:27 2018
    @author: Sina Ahmadi  (ahmadi.sina@outlook.com)
"""
from __future__ import division
import csv
import MultiLayerPerceptron

def create_dataset(filename):
    """Gets the dataset and returns the pre-processed one """
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
            
    # Convert strings to float in the data instances of the features
    for i in range(len(dataset[0])-1):
        for row in dataset:
            row[i] = float(row[i].strip())
    # Convert strings to int in the target classes
    column = len(dataset[0])-1
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    
    return dataset

def dataset_boundaries(dataset):
    """Finds the lower and the upper values in each feature and returns a list of tuples containing (lower, upper)"""
    lower_upper_bounds = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

def normalize_dataset(dataset):
    """Divides each value by the difference of the upper and the lower values in that feature."""
    lower_upper_bounds = dataset_boundaries(dataset)
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - lower_upper_bounds[i][0]) / (lower_upper_bounds[i][1] - lower_upper_bounds[i][0])
    return dataset

if __name__ == "__main__":
    filename = 'owls.csv'
    # Preprocess the dataset
    dataset = create_dataset(filename)
    # normalize input variables
    dataset = normalize_dataset(dataset)
    # Instantiate the Perceptron class
    # by data split
    #MLP = MultiLayerPerceptron.Perceptron(n_epochs=20, train_test_prop=2/3, learning_rate=0.4, n_hidden=5, activation_type="Sigmoid")
    # by cross-validation
    MLP = MultiLayerPerceptron.Perceptron(n_epochs=20, n_folds=10, learning_rate=0.4, n_hidden=5, activation_type="Sigmoid")
    # Evaluate the trained model
    scores = MLP.run(dataset)
    average = str( float("%0.2f"%(sum(scores)/len(scores)) ))
    scores = [str(float("%0.2f"%s)) for s in scores]
    print " ".join( scores ) +  " " + str( average )
    