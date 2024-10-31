#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
===============================================================================
CopyRight (c) By freeenergylab.
@Description:
Implement the error backpropagation algorithm for neural network.

@Author: Pengfei Li
@Date: Oct. 30th, 2024
===============================================================================
"""
from copy import deepcopy
from math import exp
from random import random, seed

class NeuralNetwork(object):
    """This class is designed to implement neural network.
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, n_epochs, learning_rate):
        """Initialize a neural network."""
        # Algorithm Line 1-2
        self.network = list()
        hidden_layer = [{'weights':[random() for w in range(n_inputs + 1)]} for n in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for w in range(n_hidden + 1)]} for n in range(n_outputs)]
        self.network.append(output_layer)
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    @staticmethod
    def weighted_sum(weights, inputs):
        """Calculate weighted sum for inputs."""
        # keep bias term alive with constant 1.0 as input
        activation = weights[-1] * 1.0
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    @staticmethod
    def sigmoid(activation):
        """Define a sigmoid activation function, also called squashing function,
        whose outputs are in range of (0,1); and its gradient f'(x)=f(x)*(1-f(x)).
        """
        return 1.0 / (1.0 + exp(-activation))

    @staticmethod
    def sigmoid_derivative(output):
        """Calculate the derivative of sigmoid function."""
        return output * (1.0 - output)

    def forward(self, data):
        """Propagate the inputs forward to compute the outputs."""
        inputs = deepcopy(data)
        # Algorithm Line 5-11
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.weighted_sum(neuron['weights'], inputs)
                neuron['output'] = self.sigmoid(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def backward(self, expected):
        """Propagate the deltas backward from the output layer to the input layer."""
        # Algorithm Line 12-17
        for l in reversed(range(len(self.network))):
            layer = self.network[l]
            errors = list()
            # Algorithm Line 13
            if l == len(self.network)-1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j])
            else:
                # Algorithm Line 15
                for i in range(len(layer)):
                    error = 0.0
                    # Algorithm Line 16
                    for neuron in self.network[l+1]:
                        error += (neuron['weights'][i] * neuron['delta'])
                    errors.append(error)
            for i in range(len(layer)):
                neuron = layer[i]
                neuron['delta'] = errors[i] * self.sigmoid_derivative(neuron['output'])

    def update(self, data):
        """Update the weights using the deltas. Stochastic gradient descent (SGD) is used,
        the weights are updated after every training data.
        """
        # Algorithm Line 18-20
        for l in range(len(self.network)):
            inputs = data[:-1]
            if l != 0:
                inputs = [neuron['output'] for neuron in self.network[l-1]]
            for neuron in self.network[l]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= self.learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] -= self.learning_rate * neuron['delta'] * 1.0 # update bias term

    def train(self, dataset):
        """Train this neural network."""
        # Algorithm Line 3
        for epoch in range(self.n_epochs):
            squared_loss = 0
            # Algorithm Line 4
            for data in dataset:
                outputs = self.forward(data)
                expected = [0]*int(self.n_outputs)
                expected[data[-1]] = 1
                squared_loss += sum([0.5*(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward(expected)
                self.update(data)
            print('epoch=%3d, learning_rate=%.2f, squared_loss=%.2f' % (epoch+1, self.learning_rate, squared_loss))

    def predict(self, data):
        """Make a prediction with this trained neural network."""
        outputs = self.forward(data)
        return outputs.index(max(outputs))

if __name__ == "__main__":
    """Test on training a neural network with backpropagation algorithm."""
    seed(2024)
    # dataset format in [input1, input2, label]
    dataset = [
        [  2.7810836, 2.550537003, 0],
        [1.465489372, 2.362125076, 0],
        [3.396561688, 4.400293529, 0],
        [ 1.38807019, 1.850220317, 0],
        [ 3.06407232, 3.005305973, 0],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716,  1.77106367, 1],
        [8.675418651,-0.242068655, 1],
        [7.673756466, 3.508563011, 1],
        ]
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([data[-1] for data in dataset]))
    nn = NeuralNetwork(n_inputs=n_inputs, n_hidden=6, n_outputs=n_outputs, n_epochs=60, learning_rate=0.3)
    nn.train(dataset)
    for data in dataset:
        prediction = nn.predict(data)
        print('Expected=%d, Predicted=%d' % (data[-1], prediction))