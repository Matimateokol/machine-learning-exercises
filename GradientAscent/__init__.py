import cec2017
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from cec2017.functions import f1, f2, f3

# Author: Mateusz Kolacz, 336360

class GradientAscent:

    @staticmethod
    def booth_function(X):
        return (X[0] + 2 * X[1] - 7) ** 2 + (2 * X[0] + X[1] - 5) ** 2

    @staticmethod
    def calculate_gradient(func):
        return grad(func)

    @staticmethod
    def find_random_starting_point(lower_bound, upper_bound, size=2):
        return np.random.uniform(lower_bound, upper_bound, size)

    @staticmethod
    def steepest_ascent(X_start, y, learning_rate=0.1, max_iter=100, tolerance=1e-04):
        X = X_start.copy()
        grad_fct = GradientAscent.calculate_gradient(y)

        # visualize steps
        MAX_X = 100
        PLOT_STEP = 0.1
        x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
        y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
        Xx, Yy = np.meshgrid(x_arr, y_arr)
        Z = np.empty(Xx.shape)

        for i in range(Xx.shape[0]):
            for j in range(Xx.shape[1]):
                Z[i, j] = y(np.array([Xx[i, j], Yy[i, j]]))

        plt.clf()
        plt.xlim(-MAX_X, MAX_X)
        plt.ylim(-MAX_X, MAX_X)
        plt.title(y.__name__)
        plt.contour(Xx, Yy, Z, 20)

        i = 0
        while i < max_iter:

            delta_X = -learning_rate * grad_fct(X)
            if np.all(np.abs(delta_X) <= tolerance):
                break
            plt.arrow(X[0], X[1], delta_X[0], delta_X[1], head_width=2, head_length=4, fc='k', ec='k')
            X = X + delta_X

            q = y(X)
            print('q(x) = %.6f' % q)
            i += 1

        plt.show()
        print(f'Start point: {X_start}, Suspected optimum: x = {X}, y = {y(X)}')
        return X


# TESTING FUNCTIONS:
random_point = GradientAscent.find_random_starting_point(-100, 100, size=2)

GradientAscent.steepest_ascent(random_point, GradientAscent.booth_function, learning_rate=0.1, max_iter=1000)
# GradientAscent.steepest_ascent(random_point, GradientAscent.booth_function, learning_rate=0.05, max_iter=1000)
#GradientAscent.steepest_ascent(random_point, GradientAscent.booth_function, learning_rate=0.001, max_iter=1000)

# random_point10d = GradientAscent.find_random_starting_point(-100, 100, size=10)

# GradientAscent.steepest_ascent(random_point10d, f1, learning_rate=0.00000002, max_iter=1000)
# GradientAscent.steepest_ascent(random_point10d, f1, learning_rate=0.00000001, max_iter=1000)
# GradientAscent.steepest_ascent(random_point10d, f1, learning_rate=0.000000005, max_iter=1000)

# GradientAscent.steepest_ascent(random_point10d, f2, learning_rate=0.0000000000000000002, max_iter=1000)
# GradientAscent.steepest_ascent(random_point10d, f2, learning_rate=0.0000000000000000001, max_iter=1000)
# GradientAscent.steepest_ascent(random_point10d, f2, learning_rate=0.00000000000000000005, max_iter=1000)

# GradientAscent.steepest_ascent(random_point10d, f3, learning_rate=0.000000001, max_iter=1000)
# GradientAscent.steepest_ascent(random_point10d, f3, learning_rate=0.0000000005, max_iter=1000)
# GradientAscent.steepest_ascent(random_point10d, f3, learning_rate=0.0000000001, max_iter=1000)