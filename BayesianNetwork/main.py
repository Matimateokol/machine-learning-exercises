"""
Author: Mateusz Kołacz, 336360
"""
import RandomDataGenerator

# Generating random data with Bayesian's RandomDataGenerator
SAMPLES = 10_000
SAMPLES2 = 100_000
NETWORK_STRUCTURE_FILENAME = "./data/network_structure.json"
DATASET_TO_GENERATE_FILENAME = './data/random_data_10_000'
DATASET_TO_GENERATE_FILENAME2 = './data/random_data_100_000'

RNG = RandomDataGenerator.RandomDataGenerator(SAMPLES, NETWORK_STRUCTURE_FILENAME)
RNG.generate_data(DATASET_TO_GENERATE_FILENAME) # Comment if data already generated

RNG2 = RandomDataGenerator.RandomDataGenerator(SAMPLES2, NETWORK_STRUCTURE_FILENAME)
RNG2.generate_data(DATASET_TO_GENERATE_FILENAME2) # Comment if data already generated

# Splitting the data for training and test dataset that will be used in ID3 classificator

import itertools
from ID3 import data_loader, decision_tree
from ID3.decision_tree import id3
from pandas import DataFrame, Series


SPLIT_RATIO = 0.6 # 3:2 ratio = 0.6
ITERATIONS = 50

RANDOM_DATA_FILENAME = "data/random_data_10_000"
random_data_loader = data_loader.DataLoader(RANDOM_DATA_FILENAME)

RANDOM_DATA_FILENAME2 = "data/random_data_100_000"
random_data_loader2 = data_loader.DataLoader(RANDOM_DATA_FILENAME2)

loaders = [random_data_loader, random_data_loader2]

for loader in loaders:
    dataset: DataFrame = loader.load_dataframe()
    classes: Series = dataset.iloc[:, 0].unique()
    training_size = int(len(dataset) * SPLIT_RATIO)
    results = {(predicted, actual): 0 for predicted, actual in itertools.product(classes, classes)}

    for _ in range(ITERATIONS):
        dataset = dataset.sample(frac=1)
        training_samples, testing_samples = dataset.iloc[:training_size, :], dataset.iloc[training_size:, :]

        decision_tree = id3(training_samples)

        for _, row in testing_samples.iterrows():
            results[(decision_tree.predict(row), row[0])] += 1

        print('.', end='')

    print()
    col_width = max(10, max(map(len, classes)) + 1)
    print(f"{'Predicted':{col_width}}{'Actual':{col_width}}Occurences")
    for k, v in results.items():
        count = v / ITERATIONS
        percentage = count / (len(dataset) - training_size) * 100
        print(f"{k[0]:{col_width}}{k[1]:{col_width}}{count:.2f} ({percentage:.2f}%")

    success = sum(v for k, v in results.items() if k[0] == k[1])
    print(f"Correctly classified {100 * success / (len(dataset) - training_size) / ITERATIONS:.2f}% of the sample")
    print()