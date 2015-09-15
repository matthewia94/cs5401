__author__ = 'matt'
__email__ = 'mia2n4@mst.edu'

import json
import random
import time
import sys
import matplotlib.pyplot as plt


class SatSolver:
    def __init__(self, args):
        # Check for command line arguments
        if len(args) > 1:
            self.config_file = str(args[1])
        else:
            self.config_file = 'config/default.cfg'

        self.config_params = {}
        self.num_vars = int()
        self.num_clauses = int()
        self.cnf = list()
        self.rand_seed = int()

        # Read the files (config and cnf)
        self.read_config()
        random.seed(self.rand_seed)
        self.read_cnf()

        # Create a log file
        self.log_file = open(self.config_params['log'], 'w+')

    # Parses the Json config file
    def read_config(self):
        with open(self.config_file) as data_file:
            self.config_params = json.load(data_file)

        self.rand_seed = self.config_params['rand_seed']
        # If there was no rand_seed key use the current time
        if self.rand_seed is None:
            self.rand_seed = int(time.time())

    # Parse the cnf file
    def read_cnf(self):
        f = open(self.config_params['cnf_file'], 'r')

        # Check for comments at the beginning of the file
        comment = True
        while comment:
            line = f.readline()
            comment = line.split()[0] == 'c'

        # After any leading comments should be the problem line
        cfg_vars = line.split()
        if cfg_vars[0] != 'p':
            print 'cnf file does not follow the correct format'
            exit()
        else:
            self.num_vars = int(cfg_vars[2])
            self.num_clauses = int(cfg_vars[3])

        # Read each line in to a list
        for line in f:
            # Make sure the line is not a comment
            if line.split()[0] != 'c':
                self.cnf.append(line.split()[:-1])

    # Create a random permutation of boolean values
    def generate_perm(self):
        perm = range(self.num_vars)

        for i in perm:
            perm[i] = bool(random.randint(0, 1))

        return perm

    # Run the entire experiment where each run is multiple fitness evaluations
    def run_experiment(self):
        self.init_log()
        best_run = 10000
        best_perm = list()
        best_fit = 0
        best_fit_vals = list()
        best_fit_vals_iter = list()

        # Multiple runs
        for i in range(self.config_params['runs']):
            self.log_file.write('Run ' + str(i+1) + '\n')
            start_time = time.time()
            perm, run, fit, fit_vals, fit_vals_iter = self.run_fitness_evals()
            # Log information about the current best run
            if run < best_run:
                best_perm = perm
                best_run = run
                best_fit = fit
                best_fit_vals = fit_vals
                best_fit_vals_iter = fit_vals_iter

            # Calculate time taken
            end_time = time.time()
            elapsed_time = end_time - start_time

        # Log the best solution in a file
        self.write_sol(best_fit, best_perm)

        # Ensure that graph goes to total number of fitness evals
        if max(best_fit_vals_iter) < self.config_params['fit_evals']:
            best_fit_vals_iter.append(self.config_params['fit_evals'])
            best_fit_vals.append(self.num_clauses)

        # Create plot to show fitness progression
        plt.step(best_fit_vals_iter, best_fit_vals, where='post')
        plt.axis([1, max(best_fit_vals_iter), 0, self.num_clauses])
        plt.ylabel('Fitness')
        plt.xlabel('Evaluation')
        plt.title('Fitness graph for ' + self.config_params['cnf_file'])
        plt.savefig(self.config_params['graph'])

    # Run the fitness evals for a given run
    def run_fitness_evals(self):
        best_fit = 0
        perm = list()
        fit_vals = list()
        fit_vals_iter = list()
        num_fit_evals = self.config_params['fit_evals']
        for i in range(self.config_params['fit_evals']):
            perm = self.generate_perm()
            fit = self.fitness_eval(perm)

            if fit > best_fit:
                best_fit = fit
                self.log_file.write(str(i+1) + '\t' +str(fit) + '\n')
                fit_vals.append(best_fit)
                fit_vals_iter.append(i+1)

            if fit == self.num_clauses:
                num_fit_evals = i
                break

        self.log_file.write('\n')
        return perm, num_fit_evals, best_fit, fit_vals, fit_vals_iter

    # Run a single fitness evaluation for a given permutation
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

    # Initialize the log file with meta data
    def init_log(self):
        self.log_file.write('CNF filename: ' + str(self.config_params['cnf_file']) + '\n')
        self.log_file.write('Random number seed value: ' + str(self.rand_seed) + '\n')
        self.log_file.write('Number of runs: ' + str(self.config_params['runs']) + '\n')
        self.log_file.write('Number of fitness evaluations per run: ' + str(self.config_params['fit_evals']) + '\n')
        self.log_file.write('\n' + 'Result Log' + '\n\n')

    # Write the solution file
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
