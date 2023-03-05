import numpy as np

class SimpleSearch:
    def __init__(self, start, finish, fitness, delta=0.01):
        self.start = start
        self.finish = finish
        self.fitness = fitness
        self.delta = delta

    def search(self):
        i = 0
        iteration = (self.finish - self.start) // self.delta

        x0, x1 = self.start, self.start
        fit_x1 = self.fitness(x1)
        while i != iteration:
            i += 1
            x0 += self.delta
            fit_x0 = self.fitness(x0)
            if fit_x0 > fit_x1:
                x1 = x0
                fit_x1 = self.fitness(x1)
        return x1, fit_x1
