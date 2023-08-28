import numpy as np


def mutation(start, finish, x=None, r=0.005):
    if x is None:
        return np.random.uniform(start, finish)
    return np.random.uniform(low=start if x - r < start else x - r, high=finish if x + r > finish else x + r)


def mutation_lb(start, finish, x=None, dr_min=0.0004, dr_max=0.028):
    """
    Mutation of a value with a lower bound
    :param start:
    :param finish:
    :param x:
    :param dr_min:
    :param dr_max:
    :return:
    """
    if x is None:
        return np.random.uniform(start, finish)

    if np.random.choice([-1, 1]) > 0:
        return np.random.uniform(low=finish if x+dr_min > finish else x+dr_min,
                                 high=finish if x+dr_max > finish else x+dr_max)
    else:
        return np.random.uniform(low=start if x-dr_min < start else x-dr_min,
                                 high=start if x-dr_max < start else x-dr_max)
