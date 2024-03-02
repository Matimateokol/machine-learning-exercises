import numpy as np
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
        return np.wher(self.net_input(X) >= 0.0, 1, -1)