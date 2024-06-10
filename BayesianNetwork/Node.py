import random


class Node:

    def __init__(self, name, parents, probabilities):
        self.name = name
        self.parents = parents
        self.probabilities = probabilities

    def sample(self, parent_values):
        result = []
        for value in parent_values:
            probability = self.probabilities[value]
            if random.random() <= probability:
                result.append("T")
            else:
                result.append("F")
        return result

    def probability(self, inputs):
        return self.probabilities[inputs]
