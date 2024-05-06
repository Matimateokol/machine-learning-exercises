"""
Author: Mateusz Kołacz, 336360
"""

import itertools

import data_loader
from pandas import DataFrame, Series
from decision_tree import id3

def main():

    SPLIT_RATIO = 0.6 # 3:2 ratio = 0.6
    ITERATIONS = 50

    MUSHROOM_FILENAME = "data/mushroom/agaricus-lepiota.data"
    BREAST_CANCER_FILENAME = "data/breast+cancer/breast-cancer.data"
    MUSHROOM_REDUCED_FILENAME = "data/mushroom_reduced/agaricus-lepiota-reduced.data"
    MUSHROOM_ATTR_REDUCED_FILENAME = "data/mushroom_reduced/agaricus-lepiota-attributes-reduced.data"
    SOYBEAN_FILENAME = "data/soybean+large/soybean-large.data"

    mushroom_data_loader = data_loader.DataLoader(MUSHROOM_FILENAME)
    breast_cancer_data_loader = data_loader.DataLoader(BREAST_CANCER_FILENAME)
    mushroom_reduced_data_loader = data_loader.DataLoader(MUSHROOM_REDUCED_FILENAME)
    #mushroom_attr_reduced_data_loader = data_loader.DataLoader(MUSHROOM_ATTR_REDUCED_FILENAME)
    mushroom_reduced_data_loader = data_loader.DataLoader(SOYBEAN_FILENAME)
    soybean_data_loader = data_loader.DataLoader(SOYBEAN_FILENAME)

    loaders = [mushroom_data_loader, breast_cancer_data_loader, mushroom_reduced_data_loader, soybean_data_loader, mushroom_attr_reduced_data_loader]

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



if __name__ == '__main__':
    main()