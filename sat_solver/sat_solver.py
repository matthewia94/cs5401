__author__ = 'matt'
__email__ = 'mia2n4@mst.edu'

import json
import random
import time
import sys
import matplotlib as plt


class SatSolver:
    def __init__(self, args):
        if len(args) > 1:
            self.config_file = str(args[0])
        else:
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
        best_run = 10000
        best_perm = list()
        best_fit = 0
        for i in range(self.config_params['runs']):
            self.log_file.write('Run ' + str(i+1) + '\n')
            start_time = time.time()
            perm, run, fit = self.run_fitness_evals()
            if run < best_run:
                best_perm = perm
                best_run = run
                best_fit = fit
            end_time = time.time()
            elapsed_time = end_time - start_time

        self.write_sol(best_fit, best_perm)

    def run_fitness_evals(self):
        best_fit = 0
        perm = list()
        num_fit_evals = self.config_params['fit_evals']
        for i in range(self.config_params['fit_evals']):
            perm = self.generate_perm()
            fit = self.fitness_eval(perm)

            if fit > best_fit:
                best_fit = fit
                self.log_file.write(str(i+1) + '\t' +str(fit) + '\n')

            if fit == self.num_clauses:
                num_fit_evals = i
                break

        self.log_file.write('\n')
        return perm, num_fit_evals, best_fit

    def fitness_eval(self, perm):
        fitness = 0
        for i in self.cnf:
            current_clause = False
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

    def write_sol(self, val, perm):
        sol_file = open(self.config_params['solution'], 'w+')
        sol_file.write('c Solution for: ' + self.config_params['cnf_file'] + '\n')
        sol_file.write('c MAXSAT fitness value: ' + str(val) + '\n')
        sol_file.write('v ')
        for i in range(len(perm)):
            if perm[i]:
                sol_file.write(str(i+1) + ' ')
            else:
                sol_file.write(str(-(i+1)) + ' ')


def main():
    s = SatSolver(sys.argv)
    s.run_experiment()

if __name__ == "__main__":
    main()
