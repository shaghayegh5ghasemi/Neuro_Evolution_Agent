import numpy as np
import random


class NeuralNetwork():

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        # initialize weights and biases randomly
        self.weights.append(np.random.normal(size=(layer_sizes[1], layer_sizes[0])))
        self.weights.append(np.random.normal(size=(layer_sizes[2], layer_sizes[1])))
        self.biases.append(np.zeros((layer_sizes[1], 1)))
        self.biases.append(np.zeros((layer_sizes[2], 1)))

    # sigmoid is considered as activation function
    def activation(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self, x):
        hidden_layer_output = self.activation(self.weights[0] @ x + self.biases[0])
        output = self.activation(self.weights[1] @ hidden_layer_output + self.biases[1])
        return output
