"""
Author: Mateusz Kołacz, 336360
"""

import pandas as pd
from pandas import Series, DataFrame
import math
from typing import NamedTuple, Union

""" Union type for grouping Nodes and Leaves """
Node = Union['Node', 'Leaf']

""" Simple node representing attributes"""
class Node(NamedTuple):
    attribute: str
    children: dict[str, Node]

    """ If the input is a Leaf the answer is leaf.clazz, otherwise keep calling 'predict' method until the Leaf is passed as an argument """
    def predict(self, sample: Series):
        child = self.children.get(sample[self.attribute], next(iter(self.children.values())))
        return child.predict(sample)

    def to_string(self, depth: int = 0):
        output = ''
        output += f"{self.attribute}?\n"
        for value, child in self.children.items():
            output += f"{(depth + 1) * ' '}{self.attribute}={value} -> {child.to_string(depth + 2)}"
        return output

""" Leaf node representing output - answer of the model """
class Leaf(NamedTuple):
    clazz: str

    """ If the input is a leaf node the answer is clazz """
    def predict(self, _: Series):
        return self.clazz

    def to_string(self, _: int):
        return self.clazz + '\n'

"""
Calculating entropy of a dataset.
I(U) = - total sum of (frequency * log(frequency))
"""
def calculate_entropy(samples: DataFrame):
    return - sum(freq * math.log(freq) for freq in samples.iloc[:, 0].value_counts(normalize=True))

"""
Calculating entropy of a dataset splitted into subsets by an attribute.
Inf(d, U) = total sum of (|Uj| / |U|) * I(Uj)
"""
def calculate_attribute_entropy(d_attribute: Series, samples: DataFrame):
    entropy = 0
    for value, freq in samples[d_attribute].value_counts(normalize=True).items():
        entropy += freq * calculate_entropy(samples[samples[d_attribute] == value])
    return entropy

"""
Calculating info gain.
InfGain(d, U) = I(U) - Inf(d, U)
"""
def calculate_info_gain(samples: DataFrame, d_attribute: Series):
    return calculate_entropy(samples) - calculate_attribute_entropy(d_attribute, samples)

""" Decision Tree implementation """
def id3(samples: DataFrame):
    clazzes = samples.iloc[:, 0]
    if clazzes.nunique() == 1:
        return Leaf(clazzes.iloc[0])
    if len(samples.columns) == 1:
        return Leaf(clazzes.mode()[0])

    # Calculating attribute with max info gain
    best_attribute = None
    attributes = samples.iloc[:, 1:] # Excluding the class attribute
    max_info_gain = float("-inf")

    for column in attributes.columns:
        info_gain = calculate_info_gain(samples, column)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_attribute = column

    children = {
        value: id3(samples[samples[best_attribute] == value].drop(columns=best_attribute))
        for value in samples[best_attribute].unique()
    }
    return Node(best_attribute, children)

### for DEBUG purposes ###
def debug():
    dataset: DataFrame = pd.read_csv(f"data/breast+cancer/breast-cancer.data", header=None).sample(frac=1)
    # dataset: DataFrame = pd.read_csv(f"data/mushroom/agaricus-lepiota-reduced.data", header=None).sample(frac=1)
    training_set_size = int(len(dataset) * 0.6)
    decision_tree: Node = id3(dataset[:training_set_size])
    print(decision_tree.to_string())

if __name__ == '__main__':
    debug()