import numpy as np


class Perceptron:
    def __init__(self, x1, x2, weight, alpha):
        """
        Perceptron (Binary classifier).
        Assumes that augmented values (biases) are already added to x1, x2, and weight.
        Assumes x1 > 0 & x2 < 0 for binary classification.
        From the book, it doesn't classify 0 to anyone ???

        Args:
            x1: vector 1
            x2: vector 2
            weight: weight vector
            alpha: learning rate
        """
        if len(x1) != len(x2) or len(x2) != len(weight):
            raise Exception("Vectors need to be of same length")
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.weight = np.array(weight)
        self.alpha = alpha

    def find_decision_boundary(self):
        """
        Get a new weight vector from given x1, x2, and weight vector.
        New vector can be used as decision boundary if x1 and x2 are linearly separable.

        Returns:
            A new weight vector which can linearly separate two vectors.
            If they are not linearly separable, returns None.
        """

        converged = False
        epoch = 0
        initial_weight = self.weight
        while not converged:
            converged = True
            epoch += 1
            if epoch == 1000:
                print("Vectors are not linearly separable")
                self.weight = initial_weight
            # dot product of weight vector and x1
            x1w = np.dot(self.x1, self.weight)
            if x1w <= 0:
                self.weight = self.weight + self.x1 * self.alpha
                converged = False
            # dot product of weight vector and x2
            x2w = np.dot(self.x2, self.weight)
            if x2w >= 0:
                self.weight = self.weight - self.x2 * self.alpha
                converged = False


perceptron = Perceptron([3, 3, 1], [1, 1, 1], [0, 0, 0], 1)
perceptron.find_decision_boundary()
print(perceptron.weight)