import json
from Node import Node
from Network import Network
import numpy as np


class RandomDataGenerator:
    def __init__(self, samples_number, network_structure_filename):
        self.samples_number = samples_number
        self.network_structure_filename = network_structure_filename

    def get_nodes_from_file(self, filename):
        nodes_list = []
        file = open(filename, 'r')
        nodes = json.load(file)

        for node in nodes:
            nodes_list.append(Node(node['name'], node['parents'], node['probabilities']))

        file.close()
        return nodes_list

    def save_dataset(self, input_samples, filename):
        file = open(filename, 'w')
        for table in input_samples:
            string = ''
            for item in table:
                string += item
                string += ','
            file.write(string[:-1] + '\n')
        file.close()

    def percent_true(self, input_samples):
        tables = np.transpose(input_samples)
        percents = []
        for table in tables:
            count = 0
            for item in table:
                if item == 'T':
                    count += 1
            percents.append(count / len(table))
        return percents

    def generate_data(self, new_file_name):
        network = Network(self.get_nodes_from_file(self.network_structure_filename))
        samples = network.sample(self.samples_number)
        self.save_dataset(samples, new_file_name)

        print(self.percent_true(samples))
