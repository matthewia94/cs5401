__author__ = 'matt'
__email__ = 'mia2n4@mst.edu'

import json
import random
import time


class SatSolver:
    def __init__(self):
        self.config_file = 'config/default.cfg'
        self.config_params = {}
        self.num_vars = int()
        self.num_clauses = int()
        self.cnf = list()
        self.rand_seed = int()
        self.num_runs = int()
        self.fit_evals = int()

        self.read_config()
        random.seed(self.rand_seed)
        self.read_cnf()

    def read_config(self):
        with open(self.config_file) as data_file:
            self.config_params = json.load(data_file)

        self.rand_seed = self.config_params['rand_seed']
        if self.rand_seed is None:
            self.rand_seed = int(time.time())

    def read_cnf(self):
        f = open(self.config_params['cnf_file'], 'r')
        comment = True
        while comment:
            line = f.readline()
            comment = line.split()[0] == 'c'

        cfg_vars = line.split()
        if cfg_vars[0] != 'p':
            print 'cnf file does not follow the correct format'
            exit()
        else:
            self.num_vars = int(cfg_vars[2])
            self.num_clauses = int(cfg_vars[3])

        for line in f:
            self.cnf.append(line.split()[:-1])

    def generate_perm(self):
        perm = range(self.num_vars)

        for i in perm:
            perm[i] = bool(random.randint(0, 1))

        return perm

    def run_experiment(self):
        for i in range(self.config_params['runs']):
            start_time = time.time()
            self.run_fitness_evals()
            end_time = time.time()
            print end_time - start_time

    def run_fitness_evals(self):
        best_fit = 0
        for i in range(self.config_params['fit_evals']):
            p = self.generate_perm()
            fit = self.fitness_eval(p)
            if fit > best_fit:
                best_fit = fit
            if fit == self.num_clauses:
                break

    def fitness_eval(self, perm):
        fitness = 0
        for i in self.cnf:
            current_clause = False
            for j in i:
                sign = int(j)
                var = abs(sign)
                if sign < 0:
                    current_clause = current_clause or not perm[var-1]
                else:
                    current_clause = current_clause or perm[var-1]
            if current_clause:
                fitness += 1

        return fitness


def main():
    s = SatSolver()
    s.run_experiment()

if __name__ == "__main__":
    main()