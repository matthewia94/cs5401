__author__ = 'matt'
__eamil__ = 'mia2n4@mst.edu'

import random
import time
import json
import sys
from Queue import Queue
from tree import Tree
import numpy
import copy


class Agent:
    def __init__(self, x, y, max_x, max_y, board):
        self.x = x
        self.y = y
        self.max_x = max_x
        self.max_y = max_y
        self.board = board

    # Perform the specific action based on the character passed to the function
    # u: up
    # d: down
    # l: left
    # r: right
    def take_action(self, act):
        if act == 'u':
            self.move_up()
        elif act == 'd':
            self.move_down()
        elif act == 'l':
            self.move_left()
        elif act == 'r':
            self.move_right()
        else:
            print 'Invalid move!'

    # Move position up one space on the grid
    def move_up(self):
        if self.y - 1 >= 0:
            self.y -= 1

    # Move position down one space on the grid
    def move_down(self):
        if self.y + 1 <= self.max_y:
            self.y += 1

    # Move position right one space on the grid
    def move_right(self):
        if self.x + 1 <= self.max_x:
            self.x += 1

    # Move the position left on space on the grid
    def move_left(self):
        if self.x - 1 >= 0:
            self.x -= 1


class Pacman(Agent):
    def __init__(self, x, y, max_x, max_y, board, depth):
        Agent.__init__(self, x, y, max_x, max_y, board)
        self.pills = 0

        self.tree = Tree('', 0)
        self.functional = ['add', 'sub', 'mult', 'div', 'rand']
        self.terminal = ['x', 'y', 'float']
        self.max_depth = depth
        self.score = 0

        self.init_tree()

        self.log = ''

    def __add__(self, other):
        return self.score + other

    def __radd__(self, other):
        return self.score + other

    def __cmp__(self, other):
        if self.score == other:
            return 0
        elif self.score > other:
            return -1
        else:
            return 1

    # Randomly select an action from the list of valid actions
    def generate_action(self):
        valid = self.valid_actions()
        act = self.best_act(valid)
        self.take_action(act)

    # Choose the best action
    def best_act(self, valid):
        act = ''
        best_heur = -float("inf")
        for i in valid:
            if i == 'u':
                try:
                    act_heur = self.parse_tree(self.tree, self.x, self.y-1)
                except:
                    act_heur = -500
            elif i == 'd':
                try:
                    act_heur = self.parse_tree(self.tree, self.x, self.y+1)
                except:
                    act_heur = -500
            elif i == 'r':
                try:
                    act_heur = self.parse_tree(self.tree, self.x+1, self.y)
                except:
                    act_heur = -500
            elif i == 'l':
                try:
                    act_heur = self.parse_tree(self.tree, self.x-1, self.y)
                except:
                    act_heur = -500
            elif i == 'h':
                try:
                    act_heur = self.parse_tree(self.tree, self.x, self.y)
                except:
                    act_heur = -500
            if act_heur > best_heur:
                best_heur = act_heur
                act = i

        return act

    # Perform the actual move of ms pacman by updating her position
    def take_action(self, act):
        if act == 'u':
            self.move_up()
        elif act == 'd':
            self.move_down()
        elif act == 'l':
            self.move_left()
        elif act == 'r':
            self.move_right()
        elif act != 'h':
            print 'Invalid move Pacman!'
        self.update_score()

    # Find the legal actions from the set of all possible actions regarless of legality (u, d, l, r, h)
    # Also check for walls
    def valid_actions(self):
        # Hold is always valid
        valid = ['h']

        # Check to make sure moves stay on board
        # Also does bonus check for wall
        if self.y - 1 >= 0 and self.board.board[self.y-1][self.x] != 'w' and self.board.board[self.y-1][self.x] != 'g':
            valid.append('u')
        if self.y + 1 <= self.max_y and self.board.board[self.y+1][self.x] != 'w' \
                and self.board.board[self.y+1][self.x] != 'g':
            valid.append('d')
        if self.x - 1 >= 0 and self.board.board[self.y][self.x-1] != 'w' and self.board.board[self.y][self.x-1] != 'g':
            valid.append('l')
        if self.x + 1 <= self.max_x and self.board.board[self.y][self.x+1] != 'w' \
                and self.board.board[self.y][self.x+1] != 'g':
            valid.append('r')

        return valid

    # Check if you found a pill and increment the psuedo-score if you did
    def update_score(self):
        if (self.x, self.y) in self.board.pills:
            self.pills += 1
            self.score = int(self.pills/float(self.board.tot_pills) * 100)

    # Move position back to start (top left)
    def reset(self, board):
        self.board = board
        self.x = 0
        self.y = 0
        self.pills = 0
        self.score = 0

    # Create a GP tree using ramped half and half
    def init_tree(self):
        if random.randint(0, 1) == 0:
            self.tree.data = random.choice(self.functional)
            self.init_full(self.tree, 0)
        else:
            posvals = self.functional + self.terminal
            self.tree.data = random.choice(posvals)
            self.init_grow(self.tree, 0)

    # Create a GP tree using full depth initialization
    def init_full(self, tree, depth):
        if depth < self.max_depth-1:
            nodel = Tree(random.choice(self.functional), depth+1)
            noder = Tree(random.choice(self.functional), depth+1)
            tree.add_child(nodel)
            tree.add_child(noder)
            for i in range(len(tree.children)):
                tree.children[i] = self.init_full(tree.children[i], depth+1)
        else:
            nodel = self.pick_terminal(depth+1)
            noder = self.pick_terminal(depth+1)

            tree.add_child(nodel)
            tree.add_child(noder)

        return tree

    # Create a GP tree using grow initialization
    def init_grow(self, tree, depth):
        if tree.data not in self.terminal:
            if depth < self.max_depth-1:
                if random.randint(0, 1) == 0:
                    nodel = Tree(random.choice(self.functional), depth+1)
                else:
                    nodel = self.pick_terminal(depth+1)
                tree.add_child(nodel)

                if random.randint(0, 1) == 0:
                    noder = Tree(random.choice(self.functional), depth+1)
                else:
                    noder = self.pick_terminal(depth+1)
                tree.add_child(noder)

                for i in range(len(tree.children)):
                    if tree.children[i].data in self.functional:
                        tree.children[i] = self.init_full(tree.children[i], depth+1)
            else:
                nodel = self.pick_terminal(depth+1)
                noder = self.pick_terminal(depth+1)

                tree.add_child(nodel)
                tree.add_child(noder)

        return tree

    # Create a random terminal node
    def pick_terminal(self, depth):
        rand = random.choice(self.terminal)

        if rand != 'float':
            node = Tree(rand, depth)
        else:
            node = Tree(random.random() * 100, depth)

        return node

    # Parse the tree and return the result as a float
    def parse_tree(self, tree, x, y):
        res = 0

        if len(tree.children) > 0:
            if tree.data == 'add':
                res = self.parse_tree(tree.children[0], x, y) + self.parse_tree(tree.children[1], x, y)
            elif tree.data == 'sub':
                res = self.parse_tree(tree.children[0], x, y) - self.parse_tree(tree.children[1], x, y)
            elif tree.data == 'mult':
                res = self.parse_tree(tree.children[0], x, y) * self.parse_tree(tree.children[1], x, y)
            elif tree.data == 'div':
                res = self.parse_tree(tree.children[0], x, y) / self.parse_tree(tree.children[1], x, y)
            else:
                res = random.uniform(self.parse_tree(tree.children[0], x, y), self.parse_tree(tree.children[1], x, y))
        else:
            if tree.data == 'x':
                res = self.pill_dist(x, y)
            elif tree.data == 'y':
                res = self.ghost_dist(x, y)
            else:
                res = float(tree.data)

        return res

    # Return the equation represented by the tree as a string
    def print_tree(self, tree):
        res = ''

        if len(tree.children) > 0:
            if tree.data == 'add':
                res = '(' + self.print_tree(tree.children[0]) + ') + (' + self.print_tree(tree.children[1]) + ')'
            elif tree.data == 'sub':
                res = '(' + self.print_tree(tree.children[0]) + ') - (' + self.print_tree(tree.children[1]) + ')'
            elif tree.data == 'mult':
                res = '(' + self.print_tree(tree.children[0]) + ') * (' + self.print_tree(tree.children[1]) + ')'
            elif tree.data == 'div':
                res = '(' + self.print_tree(tree.children[0]) + ') / (' + self.print_tree(tree.children[1]) + ')'
            else:
                res = 'rand(' + self.print_tree(tree.children[0]) + ', ' + self.print_tree(tree.children[1]) + ')'
        else:
            if tree.data == 'x':
                res = 'pill_dist'
            elif tree.data == 'y':
                res = 'ghost_dist'
            else:
                res = str(tree.data)

        return res

    # The manhattan distance to the nearest pill
    def pill_dist(self, x, y):
        mind = float("inf")
        for i in self.board.pills:
            temp = abs(x - i[0]) + abs(y - i[1])
            if temp < mind:
                mind = temp

        return mind

    # The manhattan distance to the nearest ghost
    def ghost_dist(self, x, y):
        mind = float("inf")
        for i in self.board.ghosts:
            temp = abs(x - i[0]) + abs(y - i[1])
            if temp < mind:
                mind = temp

        return mind

    def mutate(self, tree):
        selected = tree.rand_node()

        if selected in self.functional:
            # Grow at the randomly selected element
            selected.children = []
            self.init_grow(selected, 0)
        return tree


class Ghost(Agent):
    def __init__(self, x, y, max_x, max_y, board, depth):
        Agent.__init__(self, x, y, max_x, max_y, board)

        self.tree = Tree('', 0)
        self.functional = ['add', 'sub', 'mult', 'div', 'rand']
        self.terminal = ['x', 'y', 'float']
        self.max_depth = depth
        self.score = 0

        self.init_tree()

    def __add__(self, other):
        return self.score + other

    def __radd__(self, other):
        return self.score + other

    def __cmp__(self, other):
        if self.score == other:
            return 0
        elif self.score > other:
            return -1
        else:
            return 1

    # Randomly choose an action from the legal moves
    def generate_action(self):
        valid = self.valid_actions()
        act = self.best_act(valid)
        self.take_action(act)

    # Choose the best action
    def best_act(self, valid):
        act = ''
        best_heur = -float("inf")
        for i in valid:
            if i == 'u':
                try:
                    act_heur = self.parse_tree(self.tree, self.x, self.y-1)
                except:
                    act_heur = -500
            elif i == 'd':
                try:
                    act_heur = self.parse_tree(self.tree, self.x, self.y+1)
                except:
                    act_heur = -500
            elif i == 'r':
                try:
                    act_heur = self.parse_tree(self.tree, self.x+1, self.y)
                except:
                    act_heur = -500
            else:
                try:
                    act_heur = self.parse_tree(self.tree, self.x-1, self.y)
                except:
                    act_heur = -500
            if act_heur > best_heur:
                best_heur = act_heur
                act = i

        return act

    # Create a list of legal moves for the ghost, options are u, d, l, r
    # Also does bonus check for wall
    def valid_actions(self):
        # The ghost has to move
        valid = []

        # Check all four possible moves and keep the legal ones
        if self.y - 1 >= 0 and self.board.board[self.y-1][self.x] != 'w':
            valid.append('u')
        if self.y + 1 <= self.max_y and self.board.board[self.y+1][self.x] != 'w':
            valid.append('d')
        if self.x - 1 >= 0 and self.board.board[self.y][self.x-1] != 'w':
            valid.append('l')
        if self.x + 1 <= self.max_x and self.board.board[self.y][self.x+1] != 'w':
            valid.append('r')

        return valid

    # Move back to the starting position (bottom right)
    def reset(self, board):
        self.board = board
        self.x = self.max_x
        self.y = self.max_y
        self.score = 0

        # Create a GP tree using ramped half and half
    def init_tree(self):
        if random.randint(0, 1) == 0:
            self.tree.data = random.choice(self.functional)
            self.init_full(self.tree, 0)
        else:
            posvals = self.functional + self.terminal
            self.tree.data = random.choice(posvals)
            self.init_grow(self.tree, 0)

    # Create a GP tree using full depth initialization
    def init_full(self, tree, depth):
        if depth < self.max_depth-1:
            nodel = Tree(random.choice(self.functional), depth+1)
            noder = Tree(random.choice(self.functional), depth+1)
            tree.add_child(nodel)
            tree.add_child(noder)
            for i in range(len(tree.children)):
                tree.children[i] = self.init_full(tree.children[i], depth+1)
        else:
            nodel = self.pick_terminal(depth+1)
            noder = self.pick_terminal(depth+1)

            tree.add_child(nodel)
            tree.add_child(noder)

        return tree

    # Create a GP tree using grow initialization
    def init_grow(self, tree, depth):
        if tree.data not in self.terminal:
            if depth < self.max_depth-1:
                if random.randint(0, 1) == 0:
                    nodel = Tree(random.choice(self.functional), depth+1)
                else:
                    nodel = self.pick_terminal(depth+1)
                tree.add_child(nodel)

                if random.randint(0, 1) == 0:
                    noder = Tree(random.choice(self.functional), depth+1)
                else:
                    noder = self.pick_terminal(depth+1)
                tree.add_child(noder)

                for i in range(len(tree.children)):
                    if tree.children[i].data in self.functional:
                        tree.children[i] = self.init_full(tree.children[i], depth+1)
            else:
                nodel = self.pick_terminal(depth+1)
                noder = self.pick_terminal(depth+1)

                tree.add_child(nodel)
                tree.add_child(noder)

        return tree

    # Create a random terminal node
    def pick_terminal(self, depth):
        rand = random.choice(self.terminal)

        if rand != 'float':
            node = Tree(rand, depth)
        else:
            node = Tree(random.random() * 100, depth)

        return node

    # Parse the tree and return the result as a float
    def parse_tree(self, tree, x, y):
        res = 0

        if len(tree.children) > 0:
            if tree.data == 'add':
                res = self.parse_tree(tree.children[0], x, y) + self.parse_tree(tree.children[1], x, y)
            elif tree.data == 'sub':
                res = self.parse_tree(tree.children[0], x, y) - self.parse_tree(tree.children[1], x, y)
            elif tree.data == 'mult':
                res = self.parse_tree(tree.children[0], x, y) * self.parse_tree(tree.children[1], x, y)
            elif tree.data == 'div':
                res = self.parse_tree(tree.children[0], x, y) / self.parse_tree(tree.children[1], x, y)
            else:
                res = random.uniform(self.parse_tree(tree.children[0], x, y), self.parse_tree(tree.children[1], x, y))
        else:
            if tree.data == 'x':
                res = self.pac_dist(x, y)
            elif tree.data == 'y':
                res = self.ghost_dist(x, y)
            else:
                res = float(tree.data)

        return res

    # Return the equation represented by the tree as a string
    def print_tree(self, tree):
        res = ''

        if len(tree.children) > 0:
            if tree.data == 'add':
                res = '(' + self.print_tree(tree.children[0]) + ') + (' + self.print_tree(tree.children[1]) + ')'
            elif tree.data == 'sub':
                res = '(' + self.print_tree(tree.children[0]) + ') - (' + self.print_tree(tree.children[1]) + ')'
            elif tree.data == 'mult':
                res = '(' + self.print_tree(tree.children[0]) + ') * (' + self.print_tree(tree.children[1]) + ')'
            elif tree.data == 'div':
                res = '(' + self.print_tree(tree.children[0]) + ') / (' + self.print_tree(tree.children[1]) + ')'
            else:
                res = 'rand(' + self.print_tree(tree.children[0]) + ', ' + self.print_tree(tree.children[1]) + ')'
        else:
            if tree.data == 'x':
                res = 'pac_dist'
            elif tree.data == 'y':
                res = 'ghost_dist'
            else:
                res = str(tree.data)

        return res

    # The manhattan distance to the pacman
    def pac_dist(self, x, y):
        return abs(x - self.board.pacman[0]) + abs(y - self.board.pacman[1])

    # The manhattan distance to the nearest ghost
    def ghost_dist(self, x, y):
        mind = 1
        for i in self.board.ghosts:
            temp = abs(x - i[0]) + abs(y - i[1])
            if mind > temp > 1:
                mind = temp

        return mind

    def mutate(self, tree):
        selected = tree.rand_node()

        if selected in self.functional:
            # Grow at the randomly selected element
            selected.children = []
            self.init_grow(selected, 0)
        return tree


class BoardState:
    def __init__(self, rows, cols, density, wall_density):
        # Create an empty board
        self.rows = rows
        self.cols = cols
        self.board = [['e' for x in range(cols)] for x in range(rows)]

        # Add Ms.Pacman to the board
        self.pacman = (0, 0)
        self.board[0][0] = 'm'

        # Add ghosts to the board
        self.ghosts = list()
        for i in range(3):
            self.ghosts.append((cols-1, rows-1))

        self.board[rows-1][cols-1] = 'g'

        # Bonus add walls
        self.walls = []
        if wall_density > 0:
            for i in range(rows):
                for j in range(cols):
                    if self.board[i][j] != 'm' and self.board[i][j] != 'g' and random.random() <= wall_density/100.0:
                        # Add wall
                        self.board[i][j] = 'w'
                        # Check to make sure there everything is reachable and remove if not
                        b = self.bfs((0, 0))
                        if any(int(sys.maxint) in sublist for sublist in b):
                            self.board[i][j] = 'e'
                        else:
                            self.walls.append((j, i))

        # Add pills to the board
        self.pills = []
        self.num_pills = 0
        for i in range(rows):
            for j in range(cols):
                if self.board[i][j] != 'm' and self.board[i][j] != 'w' and random.random() <= density/100.0:
                    self.board[i][j] = 'p'
                    self.num_pills += 1
                    self.pills.append((j, i))

        self.tot_pills = self.num_pills
        self.board[rows-1][cols-1] = 'g'

    def print_board(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print self.board[i][j],
            print

    # Bonus check for connectivity
    def bfs(self, v):
        # Initialize all distances to infinity
        b = [[int(sys.maxint) for x in range(self.cols)] for x in range(self.rows)]
        # Mark walls as -1 to differentiate from unreachable and wall
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 'w':
                    b[i][j] = -1
        q = Queue()

        b[v[1]][v[0]] = 0
        q.put(v)

        while not q.empty():
            u = q.get()

            for i in [-1, 1]:
                if 0 <= u[1]+i < self.rows:
                    if b[u[1]+i][u[0]] == sys.maxint:
                        b[u[1]+i][u[0]] = b[v[1]][v[0]] + 1
                        q.put((u[0], u[1]+i))
                if 0 <= u[0]+i < self.cols:
                    if b[u[1]][u[0]+i] == sys.maxint:
                        b[u[1]][u[0]+i] = b[v[1]][v[0]] + 1
                        q.put((u[0]+i, u[1]))

        return b


class Game:
    def __init__(self, filename):
        self.rand_seed = int()
        config_params = self.read_config(filename)

        random.seed(self.rand_seed)

        self.rows = config_params['height']
        self.cols = config_params['width']
        self.density = config_params['density']

        # Bonus setup
        if 'wall_density' in config_params.keys():
            self.wall_density = config_params['wall_density']
        else:
            self.wall_density = 0

        # Setup the game
        self.time = 2*self.rows*self.cols
        self.tot_time = self.time
        self.board = BoardState(self.rows, self.cols, self.density, self.wall_density)
        self.game_over = False

        # Setup evolutionary parameters
        self.runs = config_params['runs']
        self.fit_evals = config_params['fit_evals']

        # Log file setup
        self.log = ''
        self.log_file = config_params['log_file']
        self.result_file = config_params['result_file']
        self.sol_file = config_params['sol_file']

        # Evolutionary parameters
        self.pop_size_pacman = config_params['pop_size_pacman']
        self.pop_size_ghost = config_params['pop_size_ghost']
        self.offspring_pacman = config_params['num_offspring_pacman']
        self.offspring_ghost = config_params['num_offspring_ghost']
        self.mutation = config_params['mutation_rate']
        self.tourn_size_pacman = config_params['tourn_size_pacman']
        self.tourn_size_ghost = config_params['tourn_size_ghost']
        self.over_percent = config_params['over_percent']
        self.parent_strat = config_params['parent_strat']
        self.survival_strat = config_params['survival_strat']
        self.parsimony_coef_pacman = config_params['parsimony_coef_pacman']
        self.parsimony_coef_ghost = config_params['parsimony_coef_ghost']
        self.max_depth = config_params['max_tree_depth']
        self.population_pacman = []
        self.population_ghost = []
        self.evals = 0
        self.best_fit = 0

    def read_config(self, filename):
        with open(filename) as data_file:
            config_params = json.load(data_file)

        self.rand_seed = config_params['rand_seed']
        # If there was no rand_seed key use the current time
        if self.rand_seed is None:
            self.rand_seed = int(time.time())

        return config_params

    def init_pop(self):
        self.population_pacman = []
        self.population_ghost = []
        for i in range(self.pop_size_pacman):
            # Create Ms.Pacman
            self.population_pacman.append(Pacman(0, 0, self.cols-1, self.rows-1, self.board, self.max_depth))
        for i in range(self.pop_size_ghost):
            # Create identical ghosts
            self.population_ghost.append(Ghost(self.cols-1, self.rows-1, self.cols-1, self.rows-1, self.board, self.max_depth))

    def fps_parent_selection(self, population, offspring):
        mating_pool = list()

        # Probability distribution of becoming a parent
        tot_fit = sum(population)
        prob_parent = [x.score / float(tot_fit) for x in population]

        # Randomly pick parents using the generated probability distribution
        parents_index = numpy.random.choice(a=range(len(population)), size=offspring, p=prob_parent, replace=True)
        parents_index = parents_index.tolist()

        for i in parents_index:
            mating_pool.append(population[i])

        return mating_pool

    def overselection_parent(self, population, offspring):
        mating_pool = []
        population.sort(key=lambda x: x.score, reverse=True)
        best_pool = population[:int(len(population)*self.over_percent)]
        worst_pool = population[int(len(population)*(1-self.over_percent)):]
        for i in range(int(offspring*self.over_percent)):
            mating_pool += self.fps_parent_selection(best_pool, offspring*self.over_percent)
        for i in range(int(offspring*(1-self.over_percent))):
            mating_pool += self.fps_parent_selection(worst_pool, offspring*(1-self.over_percent))
        return mating_pool

    def evolve(self):
        new_pop_pacman = []
        new_pop_ghosts = []

        # Create a mating pool for the pacman and the ghosts
        if self.parent_strat == 'over-selection':
            parents_pacman = self.overselection_parent(self.population_pacman, self.offspring_pacman)
            parents_ghosts = self.overselection_parent(self.population_ghost, self.offspring_ghost)
        else:
            parents_pacman = self.fps_parent_selection(self.population_pacman, self.offspring_pacman)
            parents_ghosts = self.fps_parent_selection(self.population_ghost, self.offspring_ghost)

        # Evolve pacman
        par_iter = 0
        while len(new_pop_pacman) < self.pop_size_pacman:
            if random.random() < self.mutation:
                new_pop_pacman.append(Pacman(0, 0, self.cols-1, self.rows-1, self.board, self.max_depth))
                new_pop_pacman.append(Pacman(0, 0, self.cols-1, self.rows-1, self.board, self.max_depth))
                new_pop_pacman[-2].tree = parents_pacman[par_iter].mutate(parents_pacman[par_iter].tree)
                new_pop_pacman[-1].tree = parents_pacman[par_iter].mutate(parents_pacman[par_iter].tree)
                par_iter += 2
            else:
                new_pop_pacman.append(Pacman(0, 0, self.cols-1, self.rows-1, self.board, self.max_depth))
                new_pop_pacman.append(Pacman(0, 0, self.cols-1, self.rows-1, self.board, self.max_depth))
                tree1, tree2 = Tree.crossover(parents_pacman[par_iter].tree, parents_pacman[par_iter+1].tree)
                new_pop_pacman[-2].tree = tree1
                new_pop_pacman[-1].tree = tree2
                par_iter += 2

        # Evolve ghosts
        par_iter = 0
        while len(new_pop_ghosts) < self.pop_size_ghost:
            if random.random() < self.mutation:
                new_pop_ghosts.append(Ghost(self.cols-1, self.rows-1, self.cols-1, self.rows-1, self.board, self.max_depth))
                new_pop_ghosts.append(Ghost(self.cols-1, self.rows-1, self.cols-1, self.rows-1, self.board, self.max_depth))
                new_pop_ghosts[-2].tree = parents_ghosts[par_iter].mutate(parents_ghosts[par_iter].tree)
                new_pop_ghosts[-1].tree = parents_ghosts[par_iter].mutate(parents_ghosts[par_iter].tree)
                par_iter += 2
            else:
                new_pop_ghosts.append(Ghost(self.cols-1, self.rows-1, self.cols-1, self.rows-1, self.board, self.max_depth))
                new_pop_ghosts.append(Ghost(self.cols-1, self.rows-1, self.cols-1, self.rows-1, self.board, self.max_depth))
                tree1, tree2 = Tree.crossover(parents_ghosts[par_iter].tree, parents_ghosts[par_iter+1].tree)
                new_pop_ghosts[-2].tree = tree1
                new_pop_ghosts[-1].tree = tree2
                par_iter += 2

        # Combine parents and children into one pool
        total_pool_pacman = self.population_pacman + new_pop_pacman
        total_pool_ghost = self.population_ghost + new_pop_ghosts
        random.shuffle(total_pool_pacman)
        random.shuffle(total_pool_ghost)

        # Evaluate the population and children, number of evaluations is dependent on the different pop sizes
        if len(total_pool_pacman) > len(total_pool_ghost):
            for i in range(len(total_pool_pacman)):
                total_pool_pacman[i].reset(self.board)
                total_pool_ghost[i % len(total_pool_ghost)].reset(self.board)
                current_fit = self.fitness_eval(total_pool_pacman[i], total_pool_ghost[i % len(total_pool_ghost)])
                total_pool_pacman[i].log = self.log
                total_pool_pacman[i].score = current_fit
                total_pool_ghost[i % len(total_pool_ghost)].score = -current_fit
                self.board_reset()
        else:
            for i in range(len(total_pool_ghost)):
                total_pool_pacman[i % len(total_pool_pacman)].reset(self.board)
                total_pool_ghost[i].reset(self.board)
                current_fit = self.fitness_eval(total_pool_pacman[i % len(total_pool_pacman)], total_pool_ghost[i])
                total_pool_pacman[i % len(total_pool_pacman)].log = self.log
                total_pool_pacman[i % len(total_pool_pacman)].score = current_fit
                total_pool_ghost[i].score = -current_fit
                self.board_reset()

        new_pop_pacman = self.survivor_selection(total_pool_pacman, self.pop_size_pacman, self.tourn_size_pacman)
        new_pop_ghosts = self.survivor_selection(total_pool_ghost, self.pop_size_ghost, self.tourn_size_ghost)

        return new_pop_pacman, new_pop_ghosts

    def survivor_selection(self, population, size, tourn_size):
        if self.survival_strat == 'trunc':
            new_pop = self.truncation_survival_selection(population, size)
        else:
            new_pop = self.k_tournament_survivor_selection(population, size, tourn_size)
        return new_pop

    def truncation_survival_selection(self, population, size):
        population.sort(key=lambda x: x.score, reverse=True)
        return population[:size]

    def k_tournament_survivor_selection(self, population, size, tourn_size):
        new_pop = list()

        current_member = 0
        while current_member < size:
            tournament = list()
            # Pick tourn_size_parents randomly
            for i in range(0, tourn_size):
                r = random.choice(population)
                tournament.append(r)

            loser = max(tournament)

            new_pop.append(loser)
            population.remove(loser)
            current_member += 1

        return new_pop

    def run_experiment(self):
        # Open and write headers to log file and result file
        result_file = open(self.result_file, 'w+')
        result_file.write('Result Log\n\n')
        result_file.close()

        self.result_header()

        for i in range(self.runs):
            generation = 1
            self.evals = 0

            # Write the run number to the result log
            result_file = open(self.result_file, 'a')
            result_file.write('Run ' + str(i+1) + '\n')

            # Initialize the population
            self.init_pop()
            for k in range(self.pop_size_pacman):
                self.population_pacman[k].board = self.board
                current_fit = self.fitness_eval(self.population_pacman[k], self.population_ghost[k])
                self.population_pacman[k].log = self.log
                self.population_pacman[k].score = current_fit
                self.population_ghost[k].score = -current_fit
                self.board_reset()
            self.evals += self.pop_size_pacman

            # Run the fitness evals for 1 run
            while self.evals < self.fit_evals:
                self.population_pacman, self.population_ghost = self.evolve()
                if self.pop_size_pacman + self.offspring_pacman > self.pop_size_ghost + self.offspring_ghost:
                    self.evals += self.pop_size_pacman + self.offspring_pacman
                else:
                    self.evals += self.pop_size_ghost + self.offspring_ghost
                current_fit = max(self.population_pacman).score
                result_file = open(self.result_file, 'a')
                avg = sum(self.population_pacman) / self.pop_size_pacman
                result_file.write(str(self.evals) + '\t' + str(avg) + '\t' + str(current_fit) + '\n')
                result_file.close()
                generation += 1
            result_file = open(self.result_file, 'a')
            result_file.write('\n')
            result_file.close()

            self.population_pacman.sort(key=lambda x: x.score, reverse=True)
            self.population_ghost.sort(key=lambda x: x.score, reverse=True)
            if self.best_fit < self.population_pacman[0].score:
                self.best_fit = self.population_pacman[0].score
                log_file = open(self.log_file, 'w+')
                log_file.write(self.population_pacman[0].log)
                log_file.close()
                sol_file = open(self.sol_file, 'w+')
                sol_file.write("Pacman's evaluation function\n" +
                               self.population_pacman[0].print_tree(self.population_pacman[0].tree) + '\n\n')
                sol_file.write("Ghosts' evaluation function\n" +
                               self.population_ghost[0].print_tree(self.population_ghost[0].tree))
                sol_file.close()
            self.board_reset()

    def board_reset(self):
        self.board = BoardState(self.rows, self.cols, self.density, self.wall_density)
        self.game_over = False
        self.time = 2*self.rows*self.cols
        self.log = ''

    def fitness_eval(self, pacman, ghost):
        # Create three identical ghosts
        ghost.board = self.board
        pacman.board = self.board
        ghost.score = 0
        pacman.score = 0
        ghosts = [copy.deepcopy(ghost) for i in range(3)]

        self.init_log(pacman, ghosts)

        # Run turns until the game is over
        while not self.game_over and self.time > 0 and len(self.board.pills) > 0:
            self.turn(pacman, ghosts)
            self.time -= 1
            if len(self.board.pills) == 0:
                pacman.score += int(self.time/float(self.tot_time) * 100)
            self.log_turn(pacman, ghosts)

        pacman.score = max(pacman.score, 1)

        ghost.score = -pacman.score
        ghost.score -= self.parsimony_coef_ghost*Tree.find_depth(ghost.tree)

        # Parsimony pressure
        pacman.score -= self.parsimony_coef_pacman*Tree.find_depth(pacman.tree)
        pacman.score = max(pacman.score, 1)

        return pacman.score

    def turn(self, pacman, ghosts):
        pacman.generate_action()
        for i in ghosts:
            i.generate_action()
        self.update_board(pacman, ghosts)
        # Output board state to terminal
        # self.board.print_board()
        # print

    def update_board(self, pacman, ghosts):
        # Move Ms.Pacman
        pac_pos = (pacman.x, pacman.y)
        self.board.board[self.board.pacman[1]][self.board.pacman[0]] = 'e'
        self.board.pacman = pac_pos
        self.board.board[pacman.y][pacman.x] = 'm'
        if pac_pos in self.board.pills:
            self.board.pills.remove(pac_pos)
            self.board.num_pills -= 1

        # Remove ghosts old positions
        for i in self.board.ghosts:
            if (i[0], i[1]) in self.board.pills:
                self.board.board[i[1]][i[0]] = 'p'
            else:
                self.board.board[i[1]][i[0]] = 'e'
        # Add new ghost positions
        for i in ghosts:
            # Check if the ghost found Ms.Pacman
            if (i.x, i.y) == pac_pos:
                self.end_game()
            self.board.board[i.y][i.x] = 'g'
        # Update ghosts list
        for i in range(len(ghosts)):
            self.board.ghosts[i] = (ghosts[i].x, ghosts[i].y)

    def end_game(self):
        self.game_over = True

    def result_header(self):
        with open(self.result_file, 'w+') as log_file:
            log_file.write('Height: ' + str(self.rows) + '\n')
            log_file.write('Width: ' + str(self.cols) + '\n')
            log_file.write('Pill density: ' + str(self.density) + '\n')
            log_file.write('Wall density: ' + str(self.wall_density) + '\n')
            log_file.write('Random seed: ' + str(self.rand_seed) + '\n')
            log_file.write('Result log: ' + self.log_file + '\n')
            log_file.write('Solution file: ' + self.sol_file + '\n')
            log_file.write('Pacman population size: ' + str(self.pop_size_pacman) + '\n')
            log_file.write('Ghost population size: ' + str(self.pop_size_ghost) + '\n')
            log_file.write('Pacman lambda: ' + str(self.offspring_pacman) + '\n')
            log_file.write('Ghost lambda: ' + str(self.offspring_ghost) + '\n')
            log_file.write('Mutation rate: ' + str(self.mutation) + '\n')
            log_file.write('Parent selection: ' + self.parent_strat + '\n')
            log_file.write('Survival strategy: ' + self.survival_strat + '\n')
            if self.survival_strat == 'k-tourn':
                log_file.write('Pacman tournament size: ' + str(self.tourn_size_pacman) + '\n')
                log_file.write('Ghost tournament size: ' + str(self.tourn_size_ghost) + '\n')
            log_file.write('Parsimony coefficient Pacman: ' + str(self.parsimony_coef_pacman) + '\n')
            log_file.write('Parsimony coefficient Ghosts: ' + str(self.parsimony_coef_ghost) + '\n')
            log_file.write('Max tree depth: ' + str(self.max_depth) + '\n')
            log_file.write('\n')
            log_file.close()

    def init_log(self, pacman, ghosts):
        # Write initial info for log file
        self.log = str(self.cols) + '\n' + str(self.rows) + '\n'
        self.log += 'm ' + str(pacman.x) + ' ' + str(pacman.y) + '\n'
        for i in range(len(ghosts)):
            self.log += str(i) + ' ' + str(ghosts[i].x) + ' ' + str(ghosts[i].y) + '\n'
        for i in self.board.pills:
            self.log += 'p ' + str(i[0]) + ' ' + str(i[1]) + '\n'
        for i in self.board.walls:
            self.log += 'w ' + str(i[0]) + ' ' + str(i[1]) + '\n'

        self.log += 't ' + str(self.time) + ' ' + str(pacman.score) + '\n'

    def log_turn(self, pacman, ghosts):
        # Write info after a turn
        self.log += 'm ' + str(pacman.x) + ' ' + str(pacman.y) + '\n'
        for i in range(len(ghosts)):
            self.log += str(i) + ' ' + str(ghosts[i].x) + ' ' + str(ghosts[i].y) + '\n'
        self.log += 't ' + str(self.time) + ' ' + str(pacman.score) + '\n'


def main():
    if len(sys.argv) == 1:
        f = 'config/default.cfg'
    else:
        f = sys.argv[1]

    g = Game(filename=f)
    g.run_experiment()

if __name__ == "__main__":
    main()
