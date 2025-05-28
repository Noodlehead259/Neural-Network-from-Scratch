#importing libraries
import pandas as pd
import numpy as np



#creating neuron class
class neuron:
    def __init__(self, weights, bias, inputs):
        self.weights = weights
        self.bias = bias
        self.inputs = inputs

    def output(self):
        size = len(self.weights)
        ans = 0
        for i in range(size):
            ans += self.weights[i] * self.inputs[i]
        return ans + self.bias
    


#importing and applying one-hot encoding to the dataset
df = pd.read_csv("heart.csv")
df = pd.get_dummies(df, columns=["ChestPainType", "Sex", "RestingECG", "ExerciseAngina", "ST_Slope"])



