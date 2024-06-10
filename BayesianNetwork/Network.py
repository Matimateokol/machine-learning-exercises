import numpy as np


class Network:

    def __init__(self, nodes):
        self.nodes = nodes

    def sample(self, quantity):
        result = []
        used = []

        for node in self.nodes:
            if len(node.parents) == 0:
                result.append(node.sample(["" for _ in range(quantity)]))
                used.append(node.name)
            else:
                parent_result = []
                parent_values = []
                for parent in node.parents:
                    index = used.index(parent)
                    parent_result.append(result[index])
                for i in range(len(parent_result[0])):
                    parent_values.append("".join(tuple(tab[i] for tab in parent_result)))

                result.append(node.sample(parent_values))
                used.append(node.name)
        return np.transpose(result)
