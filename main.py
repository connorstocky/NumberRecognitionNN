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

def main():
    #Import data for training and testing
    train_sol, train_data = convert_csv_to_array("MNIST_DATA/mnist_train.csv")
    test_sol, test_data = convert_csv_to_array("MNIST_DATA/mnist_test.csv")


if __name__ == "__main__":
    main()