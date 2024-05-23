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

def init_parameters(nodes):
    weights1 = np.random.randn(nodes,784) 
    biases1 = np.random.randn(nodes,1) 
    weights2 = np.random.randn(nodes,784) 
    biases2 = np.random.randn(nodes,1) 

    return weights1, biases1, weights2, biases2

def forward_propogation(weights1, biases1, weights2, biases2, data):
    layer1 = weights1.dot(data) + biases1 
    layer2 = weights2.dot(activation(layer1)) + biases2
    output = softmax(layer2)

    return output

#Activation Fuctions
def activation(value):
    return np.maximum(0, value) #Relu
    # return (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value)) # Tanh
    # return 1.0 / (1.0 + np.exp(-value)) # Sigmoid

def softmax(value):
    return np.exp(value)/np.sum(exp(value))


def main():
    #Import data for training and testing
    train_sol, train_data = convert_csv_to_array("MNIST_DATA/mnist_train.csv")
    test_sol, test_data = convert_csv_to_array("MNIST_DATA/mnist_test.csv")


if __name__ == "__main__":
    main()