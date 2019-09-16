import matplotlib.pyplot as plt
import numpy as np

class Glove(object):
    
    def __init__(self, tokens, coocurrence_matrix, word_dimensions=100, x_max=100, alpha=0.75, learning_rate=0.05):
        self.tokens =                   tokens
        self.X =                        coocurrence_matrix
        self.learning_rate =            learning_rate
        # initialize the word vector
        # note that the papers uses only two arrays (one for the words and one for the vectors) and doubles its size
        self.W =                        (np.random.rand(self.X.shape[0], word_dimensions) - 0.5) / float(word_dimensions)
        self.W_context =                (np.random.rand(self.X.shape[0], word_dimensions) - 0.5) / float(word_dimensions)
        self.biases =                   (np.random.rand(self.X.shape[0]) - 0.5) / float(word_dimensions)
        self.biases_context =           (np.random.rand(self.X.shape[0]) - 0.5) / float(word_dimensions)
        # gradients
        self.W_gradients =              np.ones(shape=self.W.shape, dtype=np.float64)
        self.W_context_gradients =      np.ones(shape=self.W_context.shape, dtype=np.float64)
        self.biases_gradients =         np.ones(shape=self.biases.shape, dtype=np.float64)
        self.biases_context_gradients = np.ones(shape=self.biases_context.shape, dtype=np.float64)
        # calculate the weight f(Xij)
        self.weights =                  np.where(self.X < x_max, (self.X/x_max)**alpha, 1)
        self.logXij =                   np.log(self.X)
        self.logXij =                   np.where(self.logXij==-np.inf, 0, self.logXij)
        # costs
        self.costs_history =            []

    def train(self, epochs=50):
        # iterate
        for i in range(epochs):
            # calculate the unweighted cost (used to compute the gradients)
            unweighted_costs = self.get_unweighted_costs()
            # compute the gradients
            gradients_word, gradients_context, gradients_word_bias, gradients_context_bias = self.get_gradients(unweighted_costs)
            # adagrad
            self.adagrad(gradients_word, gradients_context, gradients_word_bias, gradients_context_bias)
            # update gradients
            self.update_gradients(gradients_word, gradients_context, gradients_word_bias, gradients_context_bias)
            # calculate costs
            J = self.get_cost(unweighted_costs)
            self.costs_history.append(J)

    def get_unweighted_costs(self):
        # calculate the cost (wiT wj + bi + bj -log Xij)
        # self.W @ self.W_context.T is the dot products of each word vector and each context vector;
        # its shape is the same as the shape of self.W
        # self.biases and self.biases_context are vectors (self.W.shape[0], )
        # broadcasting will be performed by numpy so they match the dimension of the dot products
        # np.log returns the log of each member of the cooccurrence matrix
        return self.W @ self.W_context.T + self.biases + self.biases_context - self.logXij

    def get_cost(self, unweighted_costs):
        J = np.sum(np.sum(self.weights * unweighted_costs**2, axis=1), axis=0)
        return J
    
    def get_gradients(self, unweighted_costs):
        weighted_costs = np.sum(self.weights * unweighted_costs, axis=0)
        return (2 * weighted_costs[:,None] * self.W_context,  # gradients_word
                2 * weighted_costs[:,None] * self.W,          # gradients_context
                2 * weighted_costs,                           # gradients_word_bias
                2 * weighted_costs)                           # gradients_context_bias
    
    def adagrad(self, gradients_word, gradients_context, gradients_word_bias, gradients_context_bias):
        self.W -=             self.learning_rate * gradients_word / np.sqrt(self.W_gradients)
        self.W_context -=     self.learning_rate * gradients_context / np.sqrt(self.W_context_gradients)
        self.biases -=        self.learning_rate * gradients_word_bias / np.sqrt(self.biases_gradients)
        self.biases_context-= self.learning_rate * gradients_context_bias / np.sqrt(self.biases_context_gradients)
    
    def update_gradients(self, gradients_word, gradients_context, gradients_word_bias, gradients_context_bias):
        self.W_gradients +=              gradients_word**2
        self.W_context_gradients +=      gradients_context**2
        self.biases_gradients +=         gradients_word_bias**2
        self.biases_context_gradients += gradients_context_bias**2
