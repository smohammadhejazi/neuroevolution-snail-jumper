import numpy as np
import warnings


warnings.filterwarnings('ignore')


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # weights
        first_layer_weights = np.random.normal(size=(layer_sizes[1], layer_sizes[0]))
        second_layer_weights = np.random.normal(size=(layer_sizes[2], layer_sizes[1]))

        # biases
        # zero initial
        bias_first_layer = np.zeros((layer_sizes[1], 1))
        bias_second_layer = np.zeros((layer_sizes[2], 1))
        # normal initial
        # bias_first_layer = np.random.normal(size =(layer_sizes[1], 1))
        # bias_second_layer = np.random.normal(size= (layer_sizes[2], 1))

        # add to class attributes
        self.weights.append(first_layer_weights)
        self.weights.append(second_layer_weights)
        self.biases.append(bias_first_layer)
        self.biases.append(bias_second_layer)

        num_weight_params = self.weights[0].size + self.weights[1].size
        num_biases_params = self.biases[0].size + self.biases[1].size

        self.num_params = num_weight_params + num_biases_params

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        # sigmoid
        return 1 / (1 + np.exp(-x))

        # relu
        # return np.maximum(0, x)

        # leaky relu
        # return np.maximum(x * 0.1, x)

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        hidden_layer = self.activation(self.weights[0] @ x + self.biases[0])
        output_layer = self.activation(self.weights[1] @ hidden_layer + self.biases[1])
        return output_layer


# Test
if __name__ == '__main__':
    nn = NeuralNetwork([4, 10, 2])
    a = np.random.normal(size=(5, 1))
    print(a)
    print(nn.activation(a))
