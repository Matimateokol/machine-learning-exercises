# Author: Mateusz Kolacz, 336360
import cec2017
from cec2017.functions import f2, f13
import numpy as np


class EvolutionaryAlgorithm:
    LOWER_BOUND = -100
    UPPER_BOUND = 100
    PARAMS_SIZE = 10

    Q_x = f2  # goal_function, function which we are going to optimize
    budget = 10000 # total budget
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


def main():

    population_sizes = [2, 4, 8, 16, 32, 64, 128]

    ### Test 1 ###
    ea1 = EvolutionaryAlgorithm(10_000, 50, 3, f2)
    results = []
    for _ in range(30):
        results.append(ea1.run_evolutionary_algorithm()[1])

    min_val = round(np.min(results), 2)
    max_val = round(np.max(results), 2)
    mean_val = round(np.mean(results), 2)
    std_val = round(np.std(results), 2)

    print("Population size: ", ea1.population_size, ", Sigma: ", ea1.sigma)
    print("Minimum:", min_val, ", Maximum:", max_val, ", Mean:", mean_val, "Standard Deviation:", std_val)


if __name__ == '__main__':
    main()
