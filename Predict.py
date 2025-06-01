import pandas as pd
import numpy as np

class neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def output(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
    
input = []
