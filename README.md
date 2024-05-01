# ZeroToNet
ZeroToNet is a neural network implementation built entirely from scratch, with no external dependencies. This repository contains the code for constructing and training neural networks.

## Overview

ZeroToNet provides a comprehensive framework for creating neural networks from scratch. It includes modules as Dense architecture.

## Features

- Implementation of core neural network components (layers, activation functions, loss functions)
- Support for customizable network architectures and hyperparameters
- Modular design for easy extension and experimentation

## Installation

To get started with ZeroToNet, simply clone this repository to your local machine: ```git clone https://github.com/khaaaleed-5/ZeroToNet.git```

## Usage

To use ZeroToNet, import the necessary modules into your Python script:

```python
from ANN import NeuralNetwork, DenseLayer, ActivationFunction, LossFunction
# Define your network architecture
network = NeuralNetwork()
network.add_layer(DenseLayer(input_size=784, output_size=128,activation='relu'))
network.add_layer(DenseLayer(input_size=128, output_size=10,activation='relu'))
network.add_layer(DenseLayer(input_size=10, output_size=1,activation='sigmoid'))
