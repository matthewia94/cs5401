__author__ = 'matt'
__email__ = 'mia2n4@mst.edu'

import json
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy
from operator import attrgetter


class SatPerm:
    def __init__(self, perm, fit, dc_fit, mutation):
        self.perm = perm
        self.fit = fit
        self.mutation = mutation
        self.dc_fit = dc_fit
        self.level = int()

    def __add__(self, other):
        return self.level + other

    def __radd__(self, other):
        return self.level + other

    def __cmp__(self, other):
        if self.level == other:
            return 0
        elif self.level > other:
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
        self.log_file.close()

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

        f.close()

    # Create a random permutation of boolean values
    def generate_perm(self):
        perm = range(self.num_vars)

        for i in perm:
            # -1 is don't care, 0 is false, 1 is true
            perm[i] = random.randint(-1, 1)

        return perm

    # Run the entire experiment where each run is multiple fitness evaluations
    def run_experiment(self):
        self.init_log()
        best_sol = list()
        best_level = 0
        best_fit_vals = list()

        # Multiple runs
        for i in range(self.config_params['runs']):
            self.log_file = open(self.config_params['log'], 'a')
            self.log_file.write('Run ' + str(i+1) + '\n')
            self.log_file.close()
            start_time = time.time()

            log, sol, fits, pop = self.run_evolution()

            # Write run to log file
            self.write_log(log)

            # Calculate time taken
            end_time = time.time()
            elapsed_time = end_time - start_time

            if max(pop) > best_level:
                best_sol = pop
                best_level = max(pop)
                best_fit_vals = fits

        # Log the best solution in a file
        sol_file = open(self.config_params['solution'], 'w+')
        sol_file.write('c Solution for: ' + self.config_params['cnf_file'] + '\n')
        sol_file.close()
        for i in best_sol:
            if i.level == max(pop):
                self.write_sol(i)

        # Create plot to show fitness progression
        plt.plot([i.dc_fit for i in pop], [i.fit for i in pop], 'ro')
        plt.ylim([0, self.num_clauses + int(.1*self.num_clauses)])
        plt.tick_params(axis='x', which='major', labelsize=6)
        plt.ylabel('Fitness')
        plt.xlabel("Don't cares")
        plt.title('Fitness graph for ' + self.config_params['cnf_file'])
        plt.savefig(self.config_params['graph'])

    # Run a single fitness evaluation for a given permutation
    def fitness_eval(self, perm):
        fitness = 0
        for i in self.cnf:
            current_clause = False
            for j in i:
                sign = int(j)
                var = abs(sign)

                # If there is a don't care variable evaluate the line as false
                if perm[var-1] == -1:
                    current_clause = False
                    break
                else:
                    if sign < 0:
                        current_clause = current_clause or (not bool(perm[var-1]))
                    else:
                        current_clause = current_clause or bool(perm[var-1])
            if current_clause:
                fitness += 1

        return fitness

    # The fitness based on the number of don't cares
    def dc_fitness(self, perm):
        dc = 0

        for i in perm:
            if i == -1:
                dc += 1

        return dc

    def find_dominations(self, perms):
        dominates_matrix = list()
        for i in range(len(perms)):
            dominates = list()
            for j in range(len(perms)):
                if perms[i].fit > perms[j].fit and perms[i].dc_fit > perms[j].dc_fit:
                    dominates.append(j)
            dominates_matrix.append(dominates)

        return dominates_matrix

    def pareto_sorting(self, perms):
        levels = list()
        level = list()
        level.append(0)
        levels.append(level)

        dominations = self.find_dominations(perms)

        elements = range(1, len(perms[1:]))

        for i in elements:
            not_dominated = True
            for j in range(len(levels)):
                for k in range(len(levels[j])):
                    if i in dominations[k]:
                        not_dominated = False
                if not_dominated:
                    for k in levels[j]:
                        if k in dominations[i]:
                            levels[j].remove(k)
                            elements.append(k)
                    levels[j].append(i)
                    break
            if not not_dominated:
                level = list()
                level.append(i)
                levels.append(level)

        for i in range(len(levels)):
            for j in levels[i]:
                perms[j].level = i

        return perms

    # Initialize the log file with meta data
    def init_log(self):
        self.log_file = open(self.config_params['log'], 'w+');
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
        self.log_file.write('Terminate on avg population level: ' + str(self.config_params['term_avg_fitness']) + '\n')
        self.log_file.write('Terminate on best population level: ' + str(self.config_params['term_best_fitness']) + '\n')
        self.log_file.write('Mutation rate: ' + str(self.config_params['mutation']) + '\n')
        self.log_file.write('R-elitism: ' + str(self.config_params['r-elitism']) + '\n')
        self.log_file.write('R value: ' + str(self.config_params['r']) + '\n')
        self.log_file.write('Survival strategy: ' + str(self.config_params['survival_strategy']) + '\n')
        self.log_file.write('Self-adaption: ' + str(self.config_params['self-adaption']) + '\n')
        self.log_file.write('Seeded: ' + str(self.config_params['seeded']) + '\n')
        if self.config_params['seeded']:
            self.log_file.write('Seed file: ' + str(self.config_params['seed_file']) + '\n')
        self.log_file.write('\n' + 'Result Log' + '\n\n')
        self.log_file.close()

    # Write the solution file
    def write_sol(self, perm):
        sol_file = open(self.config_params['solution'], 'a')
        sol_file.write('c MAXSAT fitness value: ' + str(perm.fit) + '\n')
        sol_file.write('c Robustness fitness value: ' + str(perm.dc_fit) + '\n')
        sol_file.write('v ')
        for i in range(len(perm.perm)):
            if perm.perm[i] != -1:
                if perm.perm[i]:
                    sol_file.write(str(i+1) + ' ')
                else:
                    sol_file.write(str(-(i+1)) + ' ')
        sol_file.close()

    # Write a run to the log file
    def write_log(self, log):
        self.log_file = open(self.config_params['log'], 'a')
        for i in log:
            self.log_file.write(str(i[0]) + '\t' + str(i[1]) + '\t' + str(i[2]) + '\t' + str(i[3]) + '\t' + str(i[4]) +
                                '\n')
        self.log_file.write('\n')
        self.log_file.close()

    # Population initialization which is uniform random
    def pop_initialization(self):
        population = list()
        if self.config_params['seeded']:
            population.extend(self.seed())
        for i in range(self.config_params['population_size'] - len(population)):
            temp_perm = self.generate_perm()
            population.append(SatPerm(temp_perm, self.fitness_eval(temp_perm), self.dc_fitness(temp_perm),
                                      self.config_params['mutation']))

        return population

    # Parent selection functions
    def fps_parent_selection(self, population):
        mating_pool = list()

        # Probability distribution of becoming a parent
        tot_fit = sum(population)
        prob_parent = [x.level / float(tot_fit) for x in population]

        num_parents = self.config_params['offspring'] + 1

        # Randomly pick parents using the generated probability distribution
        parents_index = numpy.random.choice(a=range(len(population)), size=population, p=prob_parent, replace=False)
        parents_index = parents_index.tolist()

        for i in parents_index:
            mating_pool.append(population[i])

        return mating_pool

    def k_tournament_parent_selection(self, population):
        mating_pool = list()

        current_member = 0
        while current_member < self.config_params['offspring'] + 1:
            tournament = list()
            # Pick tourn_size_parents randomly
            for i in range(0, self.config_params['tourn_size_parent']):
                r = random.randint(0, len(population)-1)
                tournament.append(population[r])

            mating_pool.append(max(tournament))
            current_member += 1

        return mating_pool

    def uniform_rand_parent_selection(self, population):
        mating_pool = list()

        for i in range(self.config_params['offspring'] + 1):
            mating_pool.append(population[random.randint(0, len(population)-1)])

        return mating_pool

    # Generate the children given parents and population, uses n point crossover
    def children_generation(self, mating_pool):
        children = list()

        n = random.randint(1, self.num_vars-1)

        for i in range(0, self.config_params['offspring'], 1):
            child = []

            # Recombination to create offspring
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

            # Run self-adaption if it is turned on
            if self.config_params['self-adaption']:
                mutate_rate = self.recombine_param(mating_pool[i].mutation, mating_pool[i+1].mutation)
                mutate_rate = self.mutate_param(mutate_rate)
            else:
                mutate_rate = mating_pool[i].mutation

            # Mutation of the offspring
            child = self.mutate_perm(child, mutate_rate)

            children.append(SatPerm(child, self.fitness_eval(child), self.dc_fitness(child), mutate_rate))

        return children

    def mutate_perm(self, perm, rate):
        for j in range(len(perm)):
            val = random.randint(0, 100)
            if val < 100*rate:
                # If mutation should happen pick between the other two options randomly
                if perm[j] == -1:
                    perm[j] = random.randint(0, 1)
                elif perm[j] == 1:
                    perm[j] = random.randint(-1, 0)
                else:
                    # Use -2 as 1 to allow a contiguous range
                    perm[j] = random.randint(-2, -1)
                    if perm[j] == -2:
                        perm[j] = 1
        return perm

    # Average parameters together
    def recombine_param(self, param1, param2):
        return (param1 + param2) / 2.0

    # Modify the parameter
    def mutate_param(self, param1):
        return param1 + (random.gauss(0, 1) / 100.0)

    # Choose survivors using truncation based on fitness
    def truncation_survivor_selection(self, population):
        population.sort(key=lambda x: x.level, reverse=True)
        population = population[:self.config_params['population_size']]
        return population

    def k_tournament_survivor_selection(self, population):
        new_pop = list()

        current_member = 0
        while current_member < self.config_params['population_size']:
            tournament = list()
            # Pick tourn_size_parents randomly
            for i in range(0, self.config_params['tourn_size_child']):
                r = random.randint(0, len(population)-1)
                tournament.append(population[r])

            loser = max(tournament)

            new_pop.append(loser)
            population.remove(loser)
            current_member += 1

        return new_pop

    def uniform_rand_survivor_selection(self, population):
        new_pop = list()

        for i in range(self.config_params['offspring']):
            new_pop.append(population[random.randint(0, len(population)-1)])

        return new_pop

    def fps_survivor_selection(self, population):
        new_pop = list()

        # Probability distribution of becoming a parent
        tot_fit = sum(population)
        prob_pop = [x.level / float(tot_fit) for x in population]

        num_pop = self.config_params['population_size']

        # Randomly pick parents using the generated probability distribution
        pop_index = numpy.random.choice(a=range(len(population)-1), size=num_pop, p=prob_pop, replace=False)
        pop_index = pop_index.tolist()

        for i in pop_index:
            new_pop.append(population[i])

        return new_pop

    def restart(self, population):
        new_pop = list()

        population.sort(key=lambda x: x.level, reverse=True)
        new_pop.extend(population[:self.config_params['r']])

        for i in range(self.config_params['population_size'] - self.config_params['r']):
            entity = self.generate_perm()
            new_pop.append(SatPerm(entity, self.fitness_eval(entity), self.dc_fitness(entity), self.config_params['mutation']))

        return new_pop

    def seed(self):
        seeds = list()

        with open(self.config_params['seed_file']) as f:
            for line in f:
                perm = list()
                for i in line.split():
                    if i == 'X':
                        perm.append(-1)
                    else:
                        i = int(i)
                        if i > 0:
                            perm.append(1)
                        else:
                            perm.append(0)

                seeds.append(SatPerm(perm, self.fitness_eval(perm), self.dc_fitness(perm), self.config_params['mutation']))

        return seeds

    def run_evolution(self):
        log = list()
        fits = list()

        pop = self.pop_initialization()
        pop = self.pareto_sorting(pop)
        fitness_count = self.config_params['population_size']
        best_unchanged = 0
        avg_unchanged = 0

        best_level = max(pop, key=attrgetter('level')).level
        avg_level = sum(pop)/len(pop)
        best = pop[[i.level for i in pop].index(best_level)].perm
        best_fit = max(pop, key=attrgetter('fit')).fit
        best_dc = max(pop, key=attrgetter('dc_fit')).dc_fit
        avg_fit = sum([i.fit for i in pop])/len(pop)
        avg_dc = sum([i.dc_fit for i in pop])/len(pop)

        term_n = self.config_params['term_n']

        log.append((fitness_count, avg_fit, best_fit, avg_dc, best_dc))

        while fitness_count < self.config_params['fit_evals']:
            if (best_unchanged >= term_n or avg_unchanged >= term_n) and self.config_params['r-elitism']:
                pop = self.restart(pop)
                pop = self.pareto_sorting(pop)
                fitness_count += self.config_params['population_size'] - self.config_params['r']

            if self.config_params['parent_sel'] == 'k-tournament':
                par = self.k_tournament_parent_selection(pop)
            elif self.config_params['parent_sel'] == 'fps':
                par = self.fps_parent_selection(pop)
            else:
                par = self.uniform_rand_parent_selection(pop)

            children = self.children_generation(par)
            children = self.pareto_sorting(children)
            if self.config_params['survival_strategy'] == ',':
                pop = children
            else:
                pop += children

            if self.config_params['survival_sel'] == 'k-tournament':
                pop = self.k_tournament_survivor_selection(pop)
            elif self.config_params['survival_sel'] == 'truncation':
                pop = self.truncation_survivor_selection(pop)
            elif self.config_params['survival_sel'] == 'uniform-random':
                pop = self.uniform_rand_survivor_selection(pop)
            else:
                pop = self.fps_survivor_selection(pop)
            pop = self.pareto_sorting(pop)

            fitness_count += self.config_params['offspring']

            cur_best_level = max(pop).level
            cur_avg_level = sum(pop)/len(pop)
            best_fit = max(pop, key=attrgetter('fit')).fit
            best_dc = max(pop, key=attrgetter('dc_fit')).dc_fit
            avg_fit = sum([i.fit for i in pop])/len(pop)
            avg_dc = sum([i.dc_fit for i in pop])/len(pop)


            # Update termination conditions
            if cur_best_level > best_level:
                best_level = cur_best_level
                best_unchanged = 0
                best = pop[[i.level for i in pop].index(best_level)].perm
            elif self.config_params['term_best_fitness']:
                best_unchanged += 1

            if cur_avg_level > avg_level:
                avg_level = cur_avg_level
                avg_unchanged = 0
            elif self.config_params['term_avg_fitness']:
                avg_unchanged += 1

            fits.append([i.level for i in pop])
            log.append((fitness_count, avg_fit, best_fit, avg_dc, best_dc))

        return log, best, fits, pop


def main():
    s = SatSolver(sys.argv)
    s.run_experiment()

if __name__ == "__main__":
    main()
