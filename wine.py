# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 12:59:24 2019

@author: Anthony Pitts
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

def euclidean_distance(a,b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))

def load_data(csv_filename):
    num_columns_to_cut = 0 #cutoff for training vs. testing dataset
    rows_in_dataset = 0
    #open file to find number of lines
    with open(csv_filename, 'r') as f:
        for line in f:
            rows_in_dataset = rows_in_dataset + 1
    #data_nd_array is numby array of data set
    data_nd_array= np.zeros((rows_in_dataset,11))
    with open(csv_filename, 'r') as f:
        row_count = 0
        for line in f:
            line = line.replace("\n","")
            #spliting by colon gets array of wine data per wine sample
            line_data = line.split(";") 
            #line_data size = 11 (index 0 - 10)
            column_count=0
            for data_point in line_data:
                #does not include the "quality" value in 12th column
                if column_count == 11:
                    column_count=column_count+1
                else:
                    try:
                        #adding data to np array would fail if couldnt be converted to float
                        data_nd_array[row_count, column_count] = float(data_point)
                        column_count = column_count + 1
                    except:
                        column_count = column_count + 1
            
            nondata_line = 0 #counts number of faulty datapoints on line
            #checks how many fault datapoints on line
            for ind_column in data_nd_array[row_count]:
                if ind_column == "0.00e+00" or ind_column ==0:
                    nondata_line = nondata_line + 1
            #if every data point was fault, write over that line w next line
            if nondata_line != 11:
                row_count = row_count+1
            else:
                num_columns_to_cut = num_columns_to_cut + 1
    #removes faulty columns in dataset
    data_nd_array = np.resize(data_nd_array,(data_nd_array.shape[0]-num_columns_to_cut,11))
    return data_nd_array

#splits data into training set and testing set
def split_data(dataset, ratio):
    #ratio determines the line that the training set becomes testing data set
    row_limit_training_set= int(dataset.shape[0] * ratio)
    training_set = np.empty((row_limit_training_set,11))
    testing_set = np.empty(((dataset.shape[0]-row_limit_training_set),11))
    row_count_training_set = 0
    row_count_testing_set = 0
    
    for row in dataset:
        #if under training set row limit, add to training set
        if row_count_training_set < row_limit_training_set:
            training_set[row_count_training_set] = row
            row_count_training_set = row_count_training_set + 1
        #else add to testing data set
        else:
            testing_set[row_count_testing_set] = row
            row_count_testing_set = row_count_testing_set + 1
    return (training_set, testing_set)
def compute_centroid(data):
    """
    Returns a 1D array (a vector), representing the centroid of the data
    set. 
    """
    return np.mean(data, axis=0)
    
def experiment(ww_train, rw_train, ww_test, rw_test):
    """
    Train a model on the training data by creating a centroid for each class.
    Then test the model on the test data. Prints the number of total 
    predictions and correct predictions. Returns the accuracy. 
    """
    #gets the centroids
    red_wine_centroid = compute_centroid(rw_train)
    white_wine_centroid = compute_centroid(ww_train)
    
    rw_predictions = []
    ww_predictions = []
    for each_wine in ww_test:
        #if prediction is more likely white
        if euclidean_distance(each_wine, white_wine_centroid) <= euclidean_distance(each_wine, red_wine_centroid):
            ww_predictions.append("white")
        #if prediction is more likely red
        else:
            ww_predictions.append("red")
            
    for each_wine in rw_test:
        #if prediction is more likely white
        if euclidean_distance(each_wine, white_wine_centroid) <= euclidean_distance(each_wine, red_wine_centroid):
            rw_predictions.append("white")
        #if prediction is more likely red
        else:
            rw_predictions.append("red")
    #accuracy
    proportion_correct = .5 * ((rw_predictions.count("red")/len(rw_predictions)) + (ww_predictions.count("white")/len(ww_predictions)))
    print("total number of predictions: " + str(len(ww_predictions) + len(rw_predictions)))
    print("number of correct predictions: " + str(ww_predictions.count("white") + rw_predictions.count("red")))
    print("Accuracy: " + str(proportion_correct))
    return proportion_correct
    
def learning_curve(ww_training, rw_training, ww_test, rw_test):
    """
    Perform a series of experiments to compute and plot a learning curve.
    """
    proportion_correct_list = []
    np.random.shuffle(rw_training)
    np.random.shuffle(ww_training)
    #performs a multitude of tests
    for each_train in range (1,ww_training.shape[0]):
        proportion_correct_list.append(experiment(ww_training[:each_train,:], rw_training[:each_train,:], ww_test, rw_test))
    #returns an accuracy list over time, which could be used for plotting a learning curve
    return proportion_correct_list
    
def cross_validation(ww_data, rw_data, k):
    """
    Perform k-fold crossvalidation on the data and print the accuracy for each
    fold. 
    """
    #splits red and white wine into testing and training data sets
    split_rw = np.array_split(rw_data,k)
    split_ww = np.array_split(ww_data,k)
    
    proportion_correct_list = []
    #splits the dataset k ways
    for each_split in range(k):
        #splits the splitted data set
        rw_test_data = split_rw[each_split]
        ww_test_data = split_ww[each_split]
        
        #trains the AI on the splits data set
        rw_train_data = np.vstack(split_rw[:each_split]+split_rw[each_split:])
        ww_train_data = np.vstack(split_ww[:each_split]+split_ww[each_split:])
        
        #keeps track of accuracy of each test
        proportion_correct_list.append(experiment(ww_train_data,rw_train_data, ww_test_data, rw_test_data))
    #returns the average accuracy using cross validation
    average_accuracy = sum(proportion_correct_list) / len(proportion_correct_list)
    return average_accuracy
    
if __name__ == "__main__":
    
    ww_data = load_data('whitewine.csv')
    rw_data = load_data('redwine.csv')

    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    experiment(ww_train, rw_train, ww_test, rw_test)
    
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    learning_curve(ww_train, rw_train, ww_test, rw_test)
    
    k = 10
    acc = cross_validation(ww_data, rw_data,k)
    print("{}-fold cross-validation accuracy: {}".format(k,acc))
    