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

def backwards_propogation(layer1, output_layer1, layer2, output_layer2, weights2, input_layer, answers):
    one_hot_answers = one_hot(answers) 
    answer_size = answers.size

    error_layer2 = output_layer2 - one_hot_answers
    error_weightings2 = 1/answer_size*error_layer2.dot(output_layer1.T)
    error_biases2 = 1/answer_size*np.sum(error_layer2, 2)

    error_layer1 = weights2.T.dot(error_layer2) + delta_activation(layer1)
    error_weightings1 = 1/answer_size*error_layer1.dot(input_layer.T)
    error_biases1 = 1/answer_size*np.sum(error_layer1, 2)

    return error_weightings1, error_biases1, error_weightings2, error_biases2

#Activation Fuctions
def activation(value):
    return np.maximum(0, value) #Relu
    # return np.where(value >= 0, value, 0.1*value) # Leaky ReLU
    # return (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value)) # Tanh
    # return 1.0 / (1.0 + np.exp(-value)) # Sigmoid

def delta_activation(value):
    return value > 0
    # return np.where(value >= 0, 1, 0.1) # Leaky ReLU
    # return 1.0 - np.power(activation(value),2) # Tanh
    # return activation(value)*(1.0 - activation(value)) # Sigmoid

def softmax(value):
    return np.exp(value)/np.sum(exp(value))

def one_hot(answers):
    one_hot = np.zeroes((answers.size, answers.max()+1))
    one_hot[np.arrange(answers.size), answers] = 1
    #Transpose so each column is an example
    one_hot = one_hot.T
    return one_hot


def main():
    #Import data for training and testing
    train_sol, train_data = convert_csv_to_array("MNIST_DATA/mnist_train.csv")
    test_sol, test_data = convert_csv_to_array("MNIST_DATA/mnist_test.csv")


if __name__ == "__main__":
    main()