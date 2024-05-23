import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Convert a csv file to an array for analysing
def convert_csv_to_array(filepath):
    df = pd.read_csv(filepath, dtype=int)
    data = np.array(df).T

    x,y = data.shape

    answers = data[0]
    pixels = data[1:y]

    return answers, pixels

def init_parameters():
    weights1 = np.random.randn(10,784) 
    biases1 = np.random.randn(10,1) 
    weights2 = np.random.randn(10,784) 
    biases2 = np.random.randn(10,1) 

    return weights1, biases1, weights2, biases2

#Activation Fuctions
def activation(value):
    return np.maximum(0, unactivated_inputs) #Relu
    # return (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value)) # Tanh
    # return 1.0 / (1.0 + np.exp(-value)) # Sigmoid

def softmax(value):
    return np.exp(unactivated_inputs)/np.sum(exp(unactivated_inputs))


def main():
    #Import data for training and testing
    train_sol, train_data = convert_csv_to_array("MNIST_DATA/mnist_train.csv")
    test_sol, test_data = convert_csv_to_array("MNIST_DATA/mnist_test.csv")


if __name__ == "__main__":
    main()