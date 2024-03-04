import numpy as np
import pandas as pd


class GreedySearch:

    @staticmethod
    def run(m, M, p):

        # Build Dataframe
        df = pd.DataFrame(data={'value': p, 'weight': m, 'v/w': p / m}, columns=['value', 'weight', 'v/w'])
        df['v/w'] = df['v/w'].apply(lambda x: '{:.2f}'.format(x))

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

        return [total_value, total_weight, greedy_solution]
