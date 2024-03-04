import numpy as np
import pandas as pd


class BruteForceSearch:

    @staticmethod
    def run(m, M, p):

        num_of_bits = len(m)  # number of items
        binary_format = '{:0' + str(num_of_bits) + 'b}'
        all_items = str("0b") + "1" * num_of_bits

        df = pd.DataFrame(data={'value': p, 'weight': m, 'v/w': p / m}, columns=['value', 'weight', 'v/w'])
        df['v/w'] = df['v/w'].apply(lambda x: '{:.2f}'.format(x))

        # build all possible combinations list
        combinations = []
        for i in range(int(all_items, base=2) + 1):
            # transform binary number into list of 0-1 digits
            combination = [int(bit) for bit in binary_format.format(i)]
            # add the items combination to combinations list as np.array()
            combinations.append(np.array(combination))

        # find optimal solution (value, weight) with weight constraint by performing full search (brute force)
        opt_val = 0  # optimal value
        opt_idx = -1  # optimal solution index
        opt_weight = 0  # optimal weight
        for idx in range(len(combinations)):
            total_weight = np.dot(combinations[idx].transpose(), m)
            total_value = np.dot(combinations[idx].transpose(), p)
            if total_value > opt_val and total_weight <= M:
                opt_val = total_value
                opt_weight = total_weight
                opt_idx = idx

        return [opt_val, opt_weight, combinations[opt_idx].tolist()]
