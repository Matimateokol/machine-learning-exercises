# Author: Mateusz Kolacz, 336360
from cec2017.functions import f2, f13
import numpy as np
import pandas as pd


class EvolutionaryAlgorithm:
    LOWER_BOUND = -100
    UPPER_BOUND = 100
    PARAMS_SIZE = 10

    Q_x = f2  # goal_function, function which we are going to optimize
    budget = 10000  # total budget
    population_size = 50  # mu value
    sigma = 3  # param needed for mutation
    t = 0  # current generation (iteration) counter
    tmax = budget / population_size  # max number of permitted generations (iterations)

    def __init__(self, budget, population_size, sigma, Q_x):
        self.budget = budget
        self.population_size = population_size
        self.sigma = sigma
        self.Q_x = Q_x
        self.t = 0
        self.tmax = budget / population_size

    def find_random_starting_point(self):
        return np.random.uniform(self.LOWER_BOUND, self.UPPER_BOUND, self.PARAMS_SIZE)

    # Initialize a list of 'mu' length initial random points,
    # where mu = POPULATION_SIZE
    def initialize_population(self):
        population = []
        for i in range(self.population_size):
            population.append(self.find_random_starting_point())
        return population

    # Assessment function
    def assess(x_vector):
        return EvolutionaryAlgorithm.Q_x(x_vector)

    # Function that mutates an individual from population
    def mutate(self, child):
        mutated_child = child + np.random.normal(0, self.sigma, size=len(child))
        return mutated_child

    # Reproduction function
    def tournament_selection(self, temp_population, scores):
        selected_indices = []
        for _ in range(len(temp_population)):
            tournament_indices = np.random.choice(len(temp_population), 2, replace=True)
            tournament_scores = [scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_scores)]
            selected_indices.append(winner_index)

        return [temp_population[i] for i in selected_indices]

    def run_evolutionary_algorithm(self):
        self.t = 0
        temp_population = self.initialize_population()

        while self.t < self.tmax:
            # print("Generation {}".format(self.t))
            scores = np.array([EvolutionaryAlgorithm.assess(ind) for ind in temp_population])
            children = self.tournament_selection(temp_population, scores)
            mutated_children = np.array([self.mutate(child) for child in children])
            temp_population = mutated_children
            self.t += 1
        final_population = temp_population
        final_scores = np.array([EvolutionaryAlgorithm.assess(ind) for ind in final_population])
        best_index = np.argmin(final_scores)
        best_individual = final_population[best_index]
        best_individual_score = final_scores[best_index]
        return best_individual, best_individual_score

def test_population_evolutionary_algorithm(exp_id, population, sigma, function_to_optimize, budget):
    ea1 = EvolutionaryAlgorithm(budget, population, sigma, function_to_optimize)
    results = []
    for _ in range(30):
        results.append(ea1.run_evolutionary_algorithm()[1])

    # Calculate stats
    min_val = np.min(results)
    mean_val = np.mean(results)
    std_val = np.std(results)
    max_val = np.max(results)

    return {
        'Maks. iteracji': int(ea1.tmax),
        'Rozmiar_populacji': "mu=" + str(population),
        'Sigma': sigma,
        'Minimum': min_val,
        'Srednia': mean_val,
        'Od. Stand.': std_val,
        'Maksimum': max_val
    }


def main():
    print("\n=============================================================\n")
    pd.options.display.float_format = '{:.2f}'.format
    population_sizes = [2, 4, 8, 16, 32, 64, 128]
    sigmas = [0.1, 1, 3, 5, 10, 50, 100]

    ### Population Test suite 1: f2 function ###
    exp_id = "1.1"
    df11 = pd.DataFrame(
        columns=['Maks. iteracji', 'Rozmiar_populacji', 'Sigma', 'Minimum', 'Srednia', 'Od. Stand.', 'Maksimum']
    )
    print("Eksperyment", exp_id, ": badanie wplywu rozmiaru populacji dla funkcji", f2.__name__)
    for ps in population_sizes:
        df11 = df11._append(test_population_evolutionary_algorithm("1.1", ps, 3, f2, 10_000), ignore_index=True)

    print(df11.to_markdown())
    print("\n=============================================================\n")

    ### Population Test suite 2: f13 function ###
    exp_id = "1.2"
    df12 = pd.DataFrame(
        columns=['Maks. iteracji', 'Rozmiar_populacji', 'Sigma', 'Minimum', 'Srednia', 'Od. Stand.', 'Maksimum']
    )
    print("Eksperyment", exp_id, ": badanie wplywu rozmiaru populacji dla funkcji", f13.__name__)
    for ps in population_sizes:
        df12 = df12._append(test_population_evolutionary_algorithm("1.2", ps, 3, f13, 10_000), ignore_index=True)

    print(df12.to_markdown())
    print("\n=============================================================\n")

    ## Mutation Test suite 3: f2 function ###
    exp_id = "2.1"
    df21 = pd.DataFrame(
        columns=['Maks. iteracji', 'Rozmiar_populacji', 'Sigma', 'Minimum', 'Srednia', 'Od. Stand.', 'Maksimum']
    )
    print("Eksperyment", exp_id, ": badanie wplywu mutacji dla funkcji", f2.__name__)
    for sigma in sigmas:
        df21 = df21._append(test_population_evolutionary_algorithm("2.1", 32, sigma, f2, 10_000), ignore_index=True)

    print(df21.to_markdown())
    print("\n=============================================================\n")

    ### Mutation Test suite 4: f13 function ###
    exp_id = "2.2"
    df22 = pd.DataFrame(
        columns=['Maks. iteracji', 'Rozmiar_populacji', 'Sigma', 'Minimum', 'Srednia', 'Od. Stand.', 'Maksimum']
    )
    print("Eksperyment", exp_id, ": badanie wplywu mutacji dla funkcji", f13.__name__)
    for sigma in sigmas:
        df22 = df22._append(test_population_evolutionary_algorithm("2.2", 64, sigma, f13, 10_000), ignore_index=True)

    print(df22.to_markdown())
    print("\n=============================================================\n")

    ### Mutation Test suite 5: f2 function ###
    exp_id = "3.1"
    df31 = pd.DataFrame(
        columns=['Maks. iteracji', 'Rozmiar_populacji', 'Sigma', 'Minimum', 'Srednia', 'Od. Stand.', 'Maksimum']
    )
    print("Eksperyment", exp_id, ": badanie wplywu 5x budzet dla funkcji", f2.__name__)
    df31 = df31._append(test_population_evolutionary_algorithm("3.1", 32, 1, f2, 50_000), ignore_index=True)

    print(df31.to_markdown())
    print("\n=============================================================\n")

    ### Mutation Test suite 6: f13 function ###
    exp_id = "3.2"
    df32 = pd.DataFrame(
        columns=['Maks. iteracji', 'Rozmiar_populacji', 'Sigma', 'Minimum', 'Srednia', 'Od. Stand.', 'Maksimum']
    )
    print("Eksperyment", exp_id, ": badanie wplywu 5x budzet dla funkcji", f13.__name__)
    df32 = df32._append(test_population_evolutionary_algorithm("3.2", 64, 3, f13, 50_000), ignore_index=True)

    print(df32.to_markdown())
    print("\n=============================================================\n")

if __name__ == '__main__':
    main()
