'''Import neccessary libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Convert a csv file to an array for analysing'''
def convert_csv_to_array(filepath):
    df = pd.read_csv(filepath, dtype=int)
    data = np.array(df)
    x,y = data.shape
    data = data.T

    answers = data[0]
    pixels = data[1:y]/255.0 #Normalise pixel data 

    return answers, pixels, x, y

'''Initialise starting weights and biases with random values from [-0.5,0.5]'''
def init_parameters(nodes):
    weights1 = np.random.rand(nodes,784) - 0.5
    biases1 = np.random.rand(nodes,1) - 0.5
    weights2 = np.random.rand(10,nodes) - 0.5
    biases2 = np.random.rand(10,1) - 0.5

    return weights1, biases1, weights2, biases2

'''Run forward through the network once, based on input weights and biases for each layer'''
def forward_propogation(weights1, biases1, weights2, biases2, input):
    layer1 = weights1.dot(input) + biases1 
    activated_layer1 = activation(layer1)
    layer2 = weights2.dot(activated_layer1) + biases2
    activated_layer2 = softmax(layer2)

    return layer1, activated_layer1, layer2, activated_layer2

'''Run backwards through the network once, calculating the error at each stage and the altering the weights and biases based on how wrong each step was'''
def backwards_propogation(data, layer1, layer2, activated_layer1, activated_layer2, weights1, weights2, biases1, biases2, answers, learning_rate, a):
    one_hot_answers = one_hot(answers) 

    error_layer2 = activated_layer2 - one_hot_answers
    error_weightings2 = 1/a*error_layer2.dot(activated_layer1.T)
    error_biases2 = 1/a*np.sum(error_layer2)

    error_layer1 = weights2.T.dot(error_layer2)*delta_activation(layer1)
    error_weightings1 = 1/a*error_layer1.dot(data.T)
    error_biases1 = 1/a*np.sum(error_layer1)

    weights1 -= learning_rate*error_weightings1
    biases1 -= learning_rate*error_biases1
    weights2 -= learning_rate*error_weightings2
    biases2 -= learning_rate*error_biases2

    return weights1, biases1, weights2, biases2

'''Return the integer predicted by the network. The prediction is the value with the highest probability'''
def prediction(activated_layer2):
    return np.argmax(activated_layer2, 0)

'''Calculate the accuracy of the network with the current set of predictions'''
def accuracy(prediction, answer):
    return np.sum(prediction == answer)/answer.size

'''Train the network by applying forward propogation and backwards propogation for a given number of iterations, changing the weights and biases each time based on how correct the network's guesses were'''
def train_network(data, answers, iterations, learning_rate, nodes, a):
    #Initialise random weights and biases to begin
    weights1, biases1, weights2, biases2 = init_parameters(nodes)

    #Iterate through forward prop and back prop changing the weights and biases each time
    for x in range(iterations):
        layer1, activated_layer1, layer2, activated_layer2 = forward_propogation(weights1, biases1, weights2, biases2, data)
        weights1, biases1, weights2, biases2 = backwards_propogation(data, layer1, layer2, activated_layer1, activated_layer2, weights1, weights2, biases1, biases2, answers, learning_rate, a)

        # #Print a running accuracy every 50 iterations
        # if (x%50 == 0):
        #     print(f"Iteration: {x}")
        #     print(f"Accuracy: {accuracy(prediction(activated_layer2), answers)}")

    return weights1, biases1, weights2, biases2

def test_network(data, answers, weights1, weights2, biases1, biases2):
    layer1, activated_layer1, layer2, activated_layer2 = forward_propogation(weights1, biases1, weights2, biases2, data)
    return accuracy(prediction(activated_layer2), answers)



'''Below are all the activation functions required for the training'''
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
    return np.exp(value)/sum(np.exp(value))

def one_hot(answers):
    one_hot = np.zeros((answers.size, answers.max()+1))
    one_hot[np.arange(answers.size), answers] = 1
    #Transpose so each column is an example
    one_hot = one_hot.T
    return one_hot

'''Main function loop for training and testing and graphing'''
def main():
    #Import data for training and testing
    train_sol, train_data, a, b = convert_csv_to_array("MNIST_DATA/mnist_train.csv")
    test_sol, test_data, c, d = convert_csv_to_array("MNIST_DATA/mnist_test.csv")

    #BELOW VALUES CAN BE CHANGED
    learning_rate = 0.1 #Rate at which the errors are factored into weights and biases
    iterations = 500 #Times the nerual network will iterate and change weights + biases
    nodes = 15 #Number of nodes in the first layer

    #Train Network
    weights1, biases1, weights2, biases2 = train_network(train_data, train_sol, iterations, learning_rate, nodes, a)

    accuracy = test_network(test_data, test_sol, weights1, weights2, biases1, biases2)

    print(f"Accuracy: {round(100*accuracy,2)}%")


if __name__ == "__main__":
    main()