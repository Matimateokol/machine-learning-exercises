import RandomDataGenerator

SAMPLES = 10_000
NETWORK_STRUCTURE_FILENAME = "./data/network_structure.json"
DATASET_TO_GENERATE_FILENAME = './data/random_data'

RNG = RandomDataGenerator.RandomDataGenerator(SAMPLES, NETWORK_STRUCTURE_FILENAME)
RNG.generate_data(DATASET_TO_GENERATE_FILENAME)