import numpy as np


class SteepestAscentHillClimbing:
    def __init__(self, iteration, start, finish, fitness):
        self.iteration = iteration
        self.start = start
        self.finish = finish
        self.fitness = fitness
        self.x0 = np.random.uniform(low=self.start, high=self.finish)
        self.good_mutation = []
        self.bad_mutation = []
        self.all_mutation = []

    def mutation(self, x, r=1):
        return np.random.uniform(low=self.start if x - r < self.start else x - r,
                                 high=self.finish if x + r > self.finish else x + r)

    def run(self):
        for i in range(self.iteration):
            fit = self.fitness(R1=self.x0)
            xq = self.mutation(self.x0)
            fit_xq = self.fitness(R1=xq)
            if fit_xq >= fit:
                self.good_mutation.append((xq, fit_xq))
                self.x0 = xq
            else:
                self.bad_mutation.append((xq, fit_xq))
            self.all_mutation.append((xq, fit_xq))
        return self.x0
