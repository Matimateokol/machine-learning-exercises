# Author: Mateusz Kolacz, 336360
import cec2017
from cec2017.functions import f2, f13
import numpy as np


class EvolutionaryAlgorithm:
    LOWER_BOUND = -100
    UPPER_BOUND = 100
    PARAMS_SIZE = 10
    Q_x = f13  # goal_function, function which we are going to optimize

    budget = 10000
    population_size = 50  # mu value
    sigma = 3  # param needed for mutation
    t = 0  # current generation (iteration) counter
    tmax = budget / population_size  # max number of permitted generations (iterations)

    def __init__(self, budget, population_size, sigma):
        self.budget = budget
        self.population_size = population_size
        self.sigma = sigma
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
    def assess(self, x_vector):
        return self.Q_x(x_vector)

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
        return temp_population[selected_indices]

    def run_evolutionary_algorithm(self):
        self.t = 0
        temp_population = self.initialize_population()

        while self.t < self.tmax:
            scores = np.array([self.assess(ind) for ind in temp_population])
            children = self.tournament_selection(temp_population, scores)
            mutated_children = np.array([self.mutate(child) for child in children])
            temp_population = mutated_children
            self.t = + 1
        final_population = temp_population
        final_scores = np.array([self.Q_x(ind) for ind in final_population])
        best_index = np.argmin(final_scores)
        best_individual = final_population[best_index]
        best_individual_score = final_scores[best_index]
        return [best_individual, best_individual_score]


def main():
    ### Test 1 ###
    ea1 = EvolutionaryAlgorithm(10_000, 50, 3)
    results = []
    print(ea1.run_evolutionary_algorithm())
    # for _ in range(30):
    #     results.append(ea1.run_evolutionary_algorithm())
    #results.append(ea1.run_evolutionary_algorithm())

    # min = min(results)
    # max = max(results)
    # mean = np.mean(results)
    # std = np.std(results)

    # print("Population size: ", ea1.population_size, ", Sigma: ", ea1.sigma)
    # print("Minimum:", min, ", Maximum:", max, ", Mean:", mean, "Standard Deviation:", std)


if __name__ == '__main__':
    main()
