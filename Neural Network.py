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


#normalizing
X = df.drop("HeartDisease", axis=1)
X = X.astype(float).to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)

y = df["HeartDisease"].values.reshape(-1, 1)

#split manually (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#initialize 8 neurons in hidden layer
np.random.seed(42)
input_size = X.shape[1]
hidden_size = 8

hidden_layer = []
for _ in range(hidden_size):
    weights = np.random.randn(input_size)
    bias = 0
    hidden_layer.append(neuron(weights, bias))

#initialize output neuron
output_weights = np.random.randn(hidden_size)
output_bias = 0
output_neuron = neuron(output_weights, output_bias)

#training loop
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    predictions = []
    hidden_outputs = []

    for x in X_train:
        #forward pass through hidden layer
        hidden_activations = []
        for h in hidden_layer:
            z = h.output(x)
            a = relu(z)
            hidden_activations.append(a)
        hidden_outputs.append(hidden_activations)

        #forward pass through output neuron
        z_out = output_neuron.output(np.array(hidden_activations))
        a_out = sigmoid(z_out)
        predictions.append(a_out)

    predictions = np.array(predictions).reshape(-1, 1)
    loss = binary_cross_entropy(y_train, predictions)

    #BACKPROP manually
    d_preds = predictions - y_train  #(n_samples, 1)

    #gradients for output neuron
    d_output_weights = np.zeros_like(output_weights)
    d_output_bias = 0

    for i in range(len(X_train)):
        d_output_weights += d_preds[i][0] * np.array(hidden_outputs[i])
        d_output_bias += d_preds[i][0]

    d_output_weights /= len(X_train)
    d_output_bias /= len(X_train)

    #gradients for hidden layer
    d_hidden_weights = [np.zeros_like(n.weights) for n in hidden_layer]
    d_hidden_biases = [0 for _ in range(hidden_size)]

    for i in range(len(X_train)):
        x = X_train[i]
        hidden_zs = [h.output(x) for h in hidden_layer]
        hidden_as = [relu(z) for z in hidden_zs]
        d_sigmoid = sigmoid_derivative(output_neuron.output(np.array(hidden_as)))

        for j in range(hidden_size):
            grad = d_preds[i][0] * output_neuron.weights[j] * relu_derivative(hidden_zs[j])
            d_hidden_weights[j] += grad * x
            d_hidden_biases[j] += grad

    for j in range(hidden_size):
        d_hidden_weights[j] /= len(X_train)
        d_hidden_biases[j] /= len(X_train)

    #update weights
    output_neuron.weights -= learning_rate * d_output_weights
    output_neuron.bias -= learning_rate * d_output_bias

    for j in range(hidden_size):
        hidden_layer[j].weights -= learning_rate * d_hidden_weights[j]
        hidden_layer[j].bias -= learning_rate * d_hidden_biases[j]

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
