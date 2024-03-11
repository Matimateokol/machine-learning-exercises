import pandas as pd
import numpy as np
import time
import DatasetGenerator.DatasetGenerator as DatasetGenerator
import BruteForceSearch.BruteForceSearch as BruteForceSearch
import GreedySearch.GreedySearch as GreedySearch

# Author: Mateusz Kolacz, 336360

# Part 0: DATA PREPARATION STAGE
m = np.array([8, 3, 5, 2]) # items weight
M = np.sum(m)/2 # let the max backpack capacity be equal to half of the all items weight
p = np.array([16, 8, 9, 6]) # items value

num_of_bits = len(m) # number of items
binary_format = '{:0' + str(num_of_bits) + 'b}'
all_items = str("0b") + "1" * num_of_bits

df = pd.DataFrame(data={'value': p, 'weight': m, 'v/w': p / m}, columns=['value', 'weight', 'v/w'])
df['v/w'] = df['v/w'].apply(lambda x: '{:.2f}'.format(x))
print(df)


# PART 1: Full search (brute force)

# time execution measurement
start = time.process_time()

# build all possible combinations list
combinations = []
for i in range(int(all_items, base=2) + 1):
    # transform binary number into list of 0-1 digits
    combination = [int(bit) for bit in binary_format.format(i)]
    # add the items combination to combinations list as np.array()
    combinations.append(np.array(combination))

# find optimal solution (value, weight) with weight constraint by performing full search (brute force)
optVal = 0 # optimal value
optIdx = -1 # optimal solution index
optWeight = 0 # optimal weight
for idx in range(len(combinations)):
   total_weight = np.dot(combinations[idx].transpose(), m)
   total_value = np.dot(combinations[idx].transpose(), p)
   if total_value > optVal and total_weight <= M:
       optVal = total_value
       optWeight = total_weight
       optIdx = idx

end = time.process_time()
total = end - start
print("\n[Brute force] Execution time:", "{0:02f}s".format(total))
print(f'Optimal value: {optVal}, Optimal weight: {optWeight}')
print(f'Solution: {combinations[optIdx].tolist()}\n')


# PART 2: Greedy algorithm (Heuristics)

# time execution measurement
start = time.process_time()

# Initialize empty backpack and copy DataFrame for sorting
greedy_solution = np.zeros(len(m), dtype=int)
df_sorted = df.copy()

# Sort items by 'values / weight' ratio in non-ascending order
df_sorted.sort_values(by='v/w', ascending=False, inplace=True)

# Extract lists of values that will be used in for loop
weights = df_sorted['weight'].values
values = df_sorted['value'].values
indices = df_sorted.index.values

total_value = 0
total_weight = 0

# Greedy algorithm
for i in range(len(df_sorted)):
    if total_weight + weights[i] <= M:
        total_weight += weights[i]
        total_value += values[i]
        greedy_solution[indices[i]] = 1
    elif total_weight == M:
        break

end = time.process_time()
total = end - start
print("\n[Greedy algorithm] Execution time:", "{0:02f}s".format(total))
print(f'Total value: {total_value}, Total weight: {total_weight}')
print("The greedy solution:", greedy_solution.tolist())

print("\n")

# PART 3: PERFORMANCE COMPARISON

mu = [ 5, 10, 15, 20, 23, 24 ] # various number of items for testing

dg = DatasetGenerator.DatasetGenerator()

# BRUTE FORCE:
brute_force_measurements = []

for v in mu:
    times = []
    for i in range(25):
        dg.generate_dataset(v, 100, 100)

        # time execution measurement
        start = time.process_time()

        result = BruteForceSearch.BruteForceSearch.run(dg.weights, np.sum(dg.weights)/2, dg.values)

        end = time.process_time()
        total = end - start
        times.append(total)
    brute_force_measurements.append(times)

# GREEDY ALGORITHM:
greedy_algorithm_measurements = []

for v in mu:
    times = []
    for i in range(25):
        dg.generate_dataset(v, 100, 100)

        # time execution measurement
        start = time.process_time()

        result = GreedySearch.GreedySearch.run(dg.weights, np.sum(dg.weights)/2, dg.values)

        end = time.process_time()
        total = end - start
        times.append(total)
    greedy_algorithm_measurements.append(times)

# Calculating statistics from experiments:
brute_df = pd.DataFrame(columns=['Parameter', 'min', 'mean', 'std', 'max'])
greedy_df = pd.DataFrame(columns=['Parameter', 'min', 'mean', 'std', 'max'])

for i in range(len(mu)):
    brute_min = np.array(brute_force_measurements[i]).min()
    brute_max = np.array(brute_force_measurements[i]).max()
    brute_mean = np.array(brute_force_measurements[i]).mean()
    brute_std = np.array(brute_force_measurements[i]).std()
    df2 = {'Parameter': "mu="+str(mu[i]), 'min': brute_min, 'mean': brute_mean, 'std': brute_std, 'max': brute_max}
    brute_df = brute_df._append(df2, ignore_index=True)

    greedy_min = np.array(greedy_algorithm_measurements[i]).min()
    greedy_max = np.array(greedy_algorithm_measurements[i]).max()
    greedy_mean = np.array(greedy_algorithm_measurements[i]).mean()
    greedy_std = np.array(greedy_algorithm_measurements[i]).std()
    df2 = {'Parameter': "mu="+str(mu[i]), 'min': greedy_min, 'mean': greedy_mean, 'std': greedy_std, 'max': greedy_max}
    greedy_df = greedy_df._append(df2, ignore_index=True)

print("\n BRUTE: \n")
print(brute_df)

print("\n GREEDY: \n")
print(greedy_df)