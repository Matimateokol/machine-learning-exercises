# Importing libraries
import numpy as np

# Setting up conversion rates and sampling counts
conversionRates = [0.15, 0.04, 0.13, 0.11, 0.05]
N = 10000
d = len(conversionRates)

# Creating data set
X = np.zeros((N, d))
for i in range(N):
    for j in range(d):
        if np.random.rand() < conversionRates[j]:
            X[i][j] = 1

# Creating arrays for counting wins and loses
nPosReward = np.zeros(d)
nNegReward = np.zeros(d)

# Moving the best machine to Beta distribution and updating its wins and loses
for i in range(N):
    selected = 0
    maxRandom = 0
    for j in range(d):
        randomBeta = np.random.beta(nPosReward[j] + 1, nNegReward[j] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            selected = j
    if X[i][selected] == 1:
        nPosReward[selected] += 1
    else:
        nNegReward[selected] += 1

# Showing which machine is the best one
nSelected = nPosReward + nNegReward
for i in range(d):
    print("Machine number " + str(i + 1) + " has been chosen " + str(nSelected[i]) + " times")
print("Conclusion: the best machine is machine of number " + str(np.argmax(nSelected) + 1))