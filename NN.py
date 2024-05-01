import numpy as np

class Activations:
    def __init__(self):
        pass

    def RELU(self, inputs):
        output = np.maximum(0, inputs)
        return output

    def Sigmoid(self, inputs):
        output = 1 / (1 + np.exp(-inputs))
        return output

    def Softmax(self, inputs):
        # output = np.exp(inputs) / np.sum(np.exp(inputs),axis=1,keepdims=True)
        inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # avoid numerical instability by subtracting max value
        output = inputs / np.sum(inputs, axis=1, keepdims=True)
        return output

    def Linear(self, inputs):
        return inputs


class Metrics:
    def __init__(self) -> None:
        pass

    def Categorical_crossentropy(self, y_true, y_pred):
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss

    def Binary_crossentropy(self, y_true, y_pred):
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
        return loss

    def Sparse_categorical_crossentropy(self, y_true, y_pred):
        loss = -np.log(y_pred).mean()
        return loss
    
    def mean_squared_error(self,y_true,y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse
    
    def accuracy(self, y_true, y_pred):
        y_pred_labels = (y_pred >= 0.5).astype(int)
        acc = (y_pred_labels == y_true).mean()
        return acc


class Dense:
    def __init__(self, input_shape, neurons, activation="relu"):
        self.weights = 0.5 * np.random.randn(input_shape, neurons)
        self.bias = np.random.randn(1, neurons)
        self.activation = activation
        self.activations = Activations()
        self.input = None  # Store input for backward pass
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self,inputs):
        self.input = inputs
        a = np.dot(inputs, self.weights) + self.bias  # forward propagation
        if self.activation == "relu":
            self.output = self.activations.RELU(a)

        elif self.activation == "sigmoid":
            self.output = self.activations.Sigmoid(a)

        elif self.activation == "softmax":
            self.output = self.activations.Softmax(a)

        elif self.activation == "linear":
            self.output = self.activations.Linear(a)
        
        return self.output
    
    def backward(self, gradient_activation, learning_rate):
        if self.activation == "relu":
            gradient_input = gradient_activation * (self.output > 0)  # Derivative of ReLU
        elif self.activation == "sigmoid":
            gradient_input = gradient_activation * (self.output * (1 - self.output))  # Derivative of Sigmoid
        elif self.activation == "softmax":
            gradient_input = gradient_activation  # No need to compute derivative of Softmax due to Computation resources
        elif self.activation == "linear":
            gradient_input = gradient_activation  # No need to compute derivative of Linear. it will outputs constant

        self.gradient_weights = np.dot(self.input.T, gradient_input)
        self.gradient_bias = np.sum(gradient_input, axis=0, keepdims=True)

        gradient_input = np.dot(gradient_input, self.weights.T)  # Back-propagate the gradient

        # Update weights and bias
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias

        return gradient_input
    


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.metrics = Metrics()

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_gradient, learning_rate):
        gradient_input = loss_gradient
        for layer in reversed(self.layers):
            gradient_input = layer.backward(gradient_input, learning_rate)

    def predict(self, inputs):
        output = self.forward(inputs)
        if len(output.shape) == 1:  # Ensure output shape is (1, N)
            output = output.reshape(1, -1)
        return output
    
    def fit(self,X,y_true,epochs=10,Loss='binary_cross_entropy',learning_rate=0.001):
        losses = [self.metrics.Binary_crossentropy,
                  self.metrics.Categorical_crossentropy,
                  self.metrics.Sparse_categorical_crossentropy,
                  self.metrics.mean_squared_error]

        if Loss == 'Binary_cross_entropy':
            loss = 0    
        
        elif Loss == 'Categorical_cross_entropy':
            loss = 1
        
        elif Loss == 'Sparse_categorical_cross_entropy':
            loss = 2
        
        elif Loss == 'Mean_squared_error':
            loss = 3
            
        for epoch in range(epochs):
            y_pred = self.predict(X)
            acc = self.metrics.accuracy(y_true, y_pred)
            # Print progress
            print(f"Epoch {epoch}: Loss = {losses[loss](y_true,y_pred)}, Accuracy = {acc}")
    
            # Backpropagation
            loss_gradient = (y_pred - y_true) / len(y_true)
            self.backward(loss_gradient, learning_rate)
        
        
    