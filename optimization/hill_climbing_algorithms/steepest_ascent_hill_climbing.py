import numpy as np


class SteepestAscentHillClimbing:
    def __init__(self, iteration, start, finish, fitness):
        self.iteration = iteration
        self.start = start
        self.finish = finish
        self.fitness = fitness

        self.x0 = np.random.uniform(low=self.start, high=self.finish)

    def mutation(self, x, r=1):
        return np.random.uniform(low=self.start if x - r < self.start else x - r,
                                 high=self.finish if x + r > self.finish else x + r)

    def run(self):
        for i in range(self.iteration):
            fit = self.fitness(self.x0)
            xq = self.mutation(self.x0)
            fit_xq = self.fitness(xq)

            x = xq if fit_xq > fit else self.x0

            if self.x0 == x:
                self.x0 = self.mutation(self.x0)
            else:
                self.x0 = x
