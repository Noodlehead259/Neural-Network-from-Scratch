#importing libraries
import pandas as pd
import numpy as np



#creating neuron class
class neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def output(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
    

#activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

#loss function
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-10
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))



#importing and applying one-hot encoding to the dataset
df = pd.read_csv("heart.csv")
df = pd.get_dummies(df, columns=["ChestPainType", "Sex", "RestingECG", "ExerciseAngina", "ST_Slope"])



