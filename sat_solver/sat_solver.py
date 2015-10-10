__author__ = 'matt'
__email__ = 'mia2n4@mst.edu'

import json
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy
from operator import attrgetter
import math


class SatPerm:
    def __init__(self, perm, fit):
        self.perm = perm
        self.fit = fit

    def __add__(self, other):
        return self.fit + other

    def __radd__(self, other):
        return self.fit + other

    def __cmp__(self, other):
        if self.fit == other:
            return 0
        elif self.fit < other:
            return -1
        else:
            return 1


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
            perm[i] = random.randint(0, 1)

        return perm

    # Run the entire experiment where each run is multiple fitness evaluations
    def run_experiment(self):
        self.init_log()
        best_sol = list()
        best_fit = 0
        best_fit_vals = list()

        # Multiple runs
        for i in range(self.config_params['runs']):
            self.log_file.write('Run ' + str(i+1) + '\n')
            start_time = time.time()

            log, sol, fits = self.run_evolution()

            # Write run to log file
            self.write_log(log)

            # Calculate time taken
            end_time = time.time()
            elapsed_time = end_time - start_time

            if self.fitness_eval(sol) > best_fit:
                best_sol = sol
                best_fit = self.fitness_eval(sol)
                best_fit_vals = fits

        # Log the best solution in a file
        self.write_sol(self.fitness_eval(best_sol), best_sol)

        # Create plot to show fitness progression
        plt.boxplot(best_fit_vals)
        plt.ylim([0, self.num_clauses + int(.1*self.num_clauses)])
        plt.tick_params(axis='x', which='major', labelsize=6)
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
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
                    current_clause = current_clause or (not bool(perm[var-1]))
                else:
                    current_clause = current_clause or bool(perm[var-1])
            if current_clause:
                fitness += 1

        return fitness

    # Initialize the log file with meta data
    def init_log(self):
        self.log_file.write('CNF filename: ' + str(self.config_params['cnf_file']) + '\n')
        self.log_file.write('Random number seed value: ' + str(self.rand_seed) + '\n')
        self.log_file.write('Number of runs: ' + str(self.config_params['runs']) + '\n')
        self.log_file.write('Number of fitness evaluations per run: ' + str(self.config_params['fit_evals']) + '\n')
        self.log_file.write('Log filename: ' + self.config_params['log'] + '\n')
        self.log_file.write('Solution filename: ' + self.config_params['solution'] + '\n')
        self.log_file.write('Population size: ' + str(self.config_params['population_size']) + '\n')
        self.log_file.write('Offspring size: ' + str(self.config_params['offspring']) + '\n')
        self.log_file.write('Fitness static limit: ' + str(self.config_params['term_n']) + '\n')
        self.log_file.write('Parent selection: ' + self.config_params['parent_sel'] + '\n')
        self.log_file.write('Survival selection: ' + self.config_params['survival_sel'] + '\n')
        self.log_file.write('k-tournament size (parent): ' + str(self.config_params['tourn_size_parent']) + '\n')
        self.log_file.write('k-tournament size (survival): ' + str(self.config_params['tourn_size_child']) + '\n')
        self.log_file.write('Terminate on fitness evals: ' + str(self.config_params['termination_evals']) + '\n')
        self.log_file.write('Terminate on avg population fitness: ' + str(self.config_params['term_avg_fitness']) + '\n')
        self.log_file.write('Terminate on best population fitness: ' + str(self.config_params['term_best_fitness']) + '\n')
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

    # Write a run to the log file
    def write_log(self, log):
        for i in log:
            self.log_file.write(str(i[0]) + '\t' + str(i[1]) + '\t' + str(i[2]) + '\n')
        self.log_file.write('\n')

    # Population initialization which is uniform random
    def pop_initialization(self):
        population = list()
        for i in range(self.config_params['population_size']):
            temp_perm = self.generate_perm()
            population.append(SatPerm(temp_perm, self.fitness_eval(temp_perm)))

        return population

    # Parent selection function
    def fps_parent_selection(self, population):
        mating_pool = list()

        # Probability distribution of becoming a parent
        tot_fit = sum(population)
        prob_parent = [x.fit / float(tot_fit) for x in population]

        num_parents = self.config_params['population_size']

        # Randomly pick parents using the generated probability distribution
        parents_index = numpy.random.choice(a=range(num_parents), size=num_parents, p=prob_parent, replace=False)
        parents_index = parents_index.tolist()

        for i in parents_index:
            mating_pool.append(population[i])

        return mating_pool

    def k_tournament_parent_selection(self, population):
        mating_pool = list()

        current_member = 0
        while current_member < len(population):
            tournament = list()
            # Pick tourn_size_parents randomly
            for i in range(0, self.config_params['tourn_size_parent']-1):
                r = random.randint(0, len(population)-1)
                tournament.append(population[r])

            mating_pool.append(max(tournament))
            current_member += 1

        return mating_pool

    # Generate the children given parents and population, uses n point crossover
    def children_generation(self, mating_pool):
        children = list()

        n = random.randint(1, self.num_vars-1)

        for i in range(0, self.config_params['offspring'], 2):
            child = []

            parent = 0
            for j in range(0, self.num_vars+1, self.num_vars/n):
                if parent == 0:
                    child.extend(mating_pool[i].perm[max((j-self.num_vars/n, 0)):j])
                    parent = 1
                else:
                    child.extend(mating_pool[i+1].perm[max((j-self.num_vars/n, 0)):j])
                    parent = 0

            if self.num_vars > len(child):
                child.extend(mating_pool[i].perm[self.num_vars-(self.num_vars % n):])

            children.append(SatPerm(child, self.fitness_eval(child)))

        return children

    # Choose survivors using truncation based on fitness
    def truncation_survivor_selection(self, population):
        population.sort(key=lambda x: x.fit, reverse=True)
        population = population[:self.config_params['population_size']]
        return population

    def k_tournament_survivor_selection(self, population):
        new_pop = list()

        current_member = 0
        while current_member < self.config_params['population_size']:
            tournament = list()
            # Pick tourn_size_parents randomly
            for i in range(0, self.config_params['tourn_size_child']-1):
                r = random.randint(0, len(population)-1)
                tournament.append(population[r])

            loser = max(tournament, key=getattr('fit'))

            new_pop.append(loser)
            population.remove(loser)
            current_member += 1

        return new_pop

    def run_evolution(self):
        log = list()
        fits = list()

        pop = self.pop_initialization()
        fitness_count = self.config_params['population_size']
        best_unchanged = 0
        avg_unchanged = 0

        best_fit = max(pop, key=attrgetter('fit')).fit
        avg_fit = sum(pop)/len(pop)
        best = pop[[i.fit for i in pop].index(best_fit)].perm
        term_n = self.config_params['term_n']

        log.append((fitness_count, avg_fit, best_fit))

        while fitness_count < self.config_params['fit_evals'] and best_unchanged < term_n and avg_unchanged < term_n:
            if self.config_params['parent_sel'] == 'k-tournament':
                par = self.k_tournament_parent_selection(pop)
            else:
                par = self.fps_parent_selection(pop)

            children = self.children_generation(par)
            pop = par + children

            if self.config_params['survival_sel'] == 'k-tournament':
                pop = self.k_tournament_survivor_selection(pop)
            else:
                pop = self.truncation_survivor_selection(pop)

            fitness_count += self.config_params['offspring']

            cur_best_fit = max(pop)
            cur_avg_fit = sum(pop)/len(pop)

            # Update termination conditions
            if cur_best_fit > best_fit:
                best_fit = cur_best_fit
                best_unchanged = 0
                best = pop[[i.fit for i in pop].index(best_fit)].perm
            elif self.config_params['term_best_fitness']:
                best_unchanged += 1

            if cur_avg_fit > avg_fit:
                avg_fit = cur_avg_fit
                avg_unchanged = 0
            elif self.config_params['term_avg_fitness']:
                avg_unchanged += 1

            fits.append([i.fit for i in pop])
            log.append((fitness_count, avg_fit, best_fit))

        return log, best, fits


def main():
    s = SatSolver(sys.argv)
    s.run_experiment()

if __name__ == "__main__":
    main()
