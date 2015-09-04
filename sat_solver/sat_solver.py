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

        self.read_config()
        random.seed(self.rand_seed)
        self.read_cnf()

        self.log_file = open(self.config_params['log'], 'w+')

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
        self.init_log()
        for i in range(self.config_params['runs']):
            self.log_file.write('Run ' + str(i+1) + '\n')
            start_time = time.time()
            self.run_fitness_evals()
            end_time = time.time()
            elapsed_time = end_time - start_time

    def run_fitness_evals(self):
        best_fit = 0
        for i in range(self.config_params['fit_evals']):
            p = self.generate_perm()
            fit = self.fitness_eval(p)

            if fit > best_fit:
                best_fit = fit
                self.log_file.write(str(i+1) + '\t' +str(fit) + '\n')

            if fit == self.num_clauses:
                break

        self.log_file.write('\n')

    def fitness_eval(self, perm):
        fitness = 0
        for i in self.cnf:
            current_clause = False
            # print i
            # print perm
            for j in i:
                sign = int(j)
                var = abs(sign)
                if sign < 0:
                    current_clause = current_clause or (not perm[var-1])
                else:
                    current_clause = current_clause or perm[var-1]
            if current_clause:
                fitness += 1

        return fitness

    def init_log(self):
        self.log_file.write('CNF filename: ' + str(self.config_params['cnf_file']) + '\n')
        self.log_file.write('Random number seed value: ' + str(self.rand_seed) + '\n')
        self.log_file.write('Number of runs: ' + str(self.config_params['runs']) + '\n')
        self.log_file.write('Number of fitness evaluations per run: ' + str(self.config_params['fit_evals']) + '\n')
        self.log_file.write('\n' + 'Result Log' + '\n\n')


def main():
    s = SatSolver()
    s.run_experiment()

if __name__ == "__main__":
    main()
