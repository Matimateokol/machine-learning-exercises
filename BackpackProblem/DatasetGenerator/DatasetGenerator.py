import random as rnd
import numpy as np


class DatasetGenerator:
    def __init__(self, weights=np.array([]), values=np.array([])):
        self.weights = weights
        self.values = values

    def generate_dataset(self, items_count, max_value, max_weight):
        self.weights = np.array([rnd.randrange(1, max_weight + 1) for _ in range(items_count)])
        self.values = np.array([rnd.randrange(1, max_value + 1) for _ in range(items_count)])