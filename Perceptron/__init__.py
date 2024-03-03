import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """Classifier - perceptron.
    
    Parameters
    ------------
    eta: float
        Learning ratio (in range of 0.0 and 1.0).
    n_iter: integer
        Number of runs through train data set.
    random_state: integer
        Random number generator seed that is for initializing random wages.

    Attributes
    ------------
    w_: single dimension array
        Wages after fitting.
    errors_: list
        Number of incorrect classifications (updates) in each age.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Train data fit.
        
        Parameters
        ------------
        X: {arraylike}, dimensions = [n_samples, n_features]
            Learning vectors, where n_samples
            means number of samples and
            n_features - number of features.
        y: arraylike, dimensions = [n_samples]
            Expected values.

        Returns
        -------
        self: object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculating total stimulation"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Returns  a class label after unit jump function calculation"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# TESTING PERCEPTRON ON IRIS DATA

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

# we choose varieties setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# we choose length of sepal and length of petal
X = df.iloc[0:100, [0, 2]].values

# generating data plot
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# Training the data model
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Ages')
plt.ylabel('Number of updates')
plt.show()


# Visualizing decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # configure chars generator and map of colors
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # draw decision region plot
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # draw samples plot
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx],
                    label=cl, edgecolor='black')


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()
