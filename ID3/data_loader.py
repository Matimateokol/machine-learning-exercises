import csv
import random

class DataLoader:
    file_name = None

    def __init__(self, file_name):
        self.file_name = file_name

    def load_data(self):
        data = []
        with open(self.file_name, 'r', newline='') as f:
            content = csv.reader(f, delimiter=',')
            for row in content:
                data.append(row)
            return data

    def save_data(self, data):
        with open(self.file_name, 'w', newline='') as f:
            data_writer = csv.writer(f, delimiter=',')
            for row in data:
                data_writer.writerow(row)

    def split_data(self, data, ratio_training_set=3, ratio_test_set=2):
        training = self.load_data()
        test = []
        len_test = ratio_test_set * (len(training) // (ratio_training_set + ratio_test_set))
        for i in range(len_test):
            n = len(training)
            idx = random.radint(0, n - 1)
            test.append(training.pop(idx))

        self.save_data(self.file_name + "-training", training)
        self.save_data(self.file_name + "-test", test)
