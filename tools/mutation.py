import numpy as np


def mutation(start, finish, x=None, r=0.005):
    if x is None:
        return np.random.uniform(start, finish)
    return np.random.uniform(low=start if x - r < start else x - r, high=finish if x + r > finish else x + r)


def mutation_lb(start, finish, x=None, dr_min=0.001, dr_max=0.025):
    if x is None:
        res = np.random.uniform(start, finish)
    elif np.random.choice([-1, 1]) > 0:
        res = np.random.uniform(low=finish if x+dr_min > finish else x+dr_min,
                                high=finish if x+dr_max > finish else x+dr_max)
    else:
        res = np.random.uniform(low=start if x-dr_min < start else x-dr_min,
                                high=start if x-dr_max < start else x-dr_max)
    return np.round(res, 3)


def mutation_random(start, finish):
    """
    Random mutation within a range without restrictions
    :param start: starting range value
    :param finish: end value of range
    :return:
    """
    return start + np.round((finish - start) * np.random.random(), 3)

