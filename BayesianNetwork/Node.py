"""
Author: Mateusz Kołacz, 336360
"""
import random

"""
    Class that represents a node in the Bayesian Network
"""
class Node:

    def __init__(self, name, parents, probabilities):
        self.name = name
        self.parents = parents
        self.probabilities = probabilities

    """
        Method for generating a random data sample
    """
    def sample(self, parent_values):
        result = []
        for value in parent_values:
            probability = self.probabilities[value]
            if random.random() <= probability:
                result.append("T")
            else:
                result.append("F")
        return result

    """
        Method returning list of probabilities
    """
    def probability(self, inputs):
        return self.probabilities[inputs]
