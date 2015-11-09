__author__ = 'matt'
__eamil__ = 'mia2n4@mst.edu'

import random
import time
import json
import sys
from Queue import Queue


class Agent:
    def __init__(self, x, y, max_x, max_y, board):
        self.x = x
        self.y = y
        self.max_x = max_x
        self.max_y = max_y
        self.board = board

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

    def move_up(self):
        if self.y - 1 >= 0:
            self.y -= 1

    def move_down(self):
        if self.y + 1 <= self.max_y:
            self.y += 1

    def move_right(self):
        if self.x + 1 <= self.max_x:
            self.x += 1

    def move_left(self):
        if self.x - 1 >= 0:
            self.x -= 1


class Pacman(Agent):
    def __init__(self, x, y, max_x, max_y, board):
        Agent.__init__(self, x, y, max_x, max_y, board)
        self.score = 0

    def generate_action(self):
        valid = self.valid_actions()
        act = valid[random.randint(0, len(valid)-1)]
        self.take_action(act)

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
            print 'Invalid move!'
        self.update_score()

    def valid_actions(self):
        # Hold is always valid
        valid = ['h']

        # Check to make sure moves stay on board
        # Also does bonus check for wall
        if self.y - 1 >= 0 and self.board.board[self.y-1][self.x] != 'w':
            valid.append('u')
        if self.y + 1 <= self.max_y and self.board.board[self.y+1][self.x] != 'w':
            valid.append('d')
        if self.x - 1 >= 0 and self.board.board[self.y][self.x-1] != 'w':
            valid.append('l')
        if self.x + 1 <= self.max_x and self.board.board[self.y][self.x+1] != 'w':
            valid.append('r')

        return valid

    def update_score(self):
        if (self.x, self.y) in self.board.pills:
            self.score += 1

    def reset(self, board):
        self.board = board
        self.x = 0
        self.y = 0
        self.score = 0


class Ghost(Agent):
    def __init__(self, x, y, max_x, max_y, board):
        Agent.__init__(self, x, y, max_x, max_y, board)

    def generate_action(self):
        valid = self.valid_actions()
        act = valid[random.randint(0, len(valid)-1)]
        self.take_action(act)

    # Also does bonus check for wall
    def valid_actions(self):
        # Hold is always valid
        valid = []

        if self.y - 1 >= 0 and self.board.board[self.y-1][self.x] != 'w':
            valid.append('u')
        if self.y + 1 <= self.max_y and self.board.board[self.y+1][self.x] != 'w':
            valid.append('d')
        if self.x - 1 >= 0 and self.board.board[self.y][self.x-1] != 'w':
            valid.append('l')
        if self.x + 1 <= self.max_x and self.board.board[self.y][self.x+1] != 'w':
            valid.append('r')

        return valid

    def reset(self, board):
        self.board = board
        self.x = self.max_x
        self.y = self.max_y


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


        # Add pills to the board
        self.pills = []
        self.num_pills = 0
        for i in range(rows):
            for j in range(cols):
                if self.board[i][j] != 'm' and self.board[i][j] != 'w' and random.random() <= density/100.0:
                    self.board[i][j] = 'p'
                    self.num_pills += 1
                    self.pills.append((j, i))

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
        self.board = BoardState(self.rows, self.cols, self.density, self.wall_density)
        self.game_over = False

        # Create Ms.Pacman
        self.pacman = Pacman(0, 0, self.cols-1, self.rows-1, self.board)

        # Create ghosts
        self.ghosts = list()
        self.ghosts = [Ghost(self.cols-1, self.rows-1, self.cols-1, self.rows-1, self.board) for i in range(3)]

        # Setup evolutionary parameters
        self.runs = config_params['runs']
        self.fit_evals = config_params['fit_evals']

        # Log file setup
        self.log = ''
        self.log_file = config_params['log_file']
        self.result_file = config_params['result_file']

    def read_config(self, filename):
        with open(filename) as data_file:
            config_params = json.load(data_file)

        self.rand_seed = config_params['rand_seed']
        # If there was no rand_seed key use the current time
        if self.rand_seed is None:
            self.rand_seed = int(time.time())

        return config_params

    def run_experiment(self):
        # Open and write headers to log file and result file
        result_file = open(self.result_file, 'w+')
        result_file.write('Result Log\n\n')
        result_file.close()

        self.log_header()

        for i in range(self.runs):
            best_log = ''
            best_fit = 0

            # Write the run number to the result log
            result_file = open(self.result_file, 'a')
            result_file.write('Run ' + str(i+1) + '\n')

            # Run the fitness evals for 1 run
            for j in range(self.fit_evals):
                current_fit = self.fitness_eval()
                if current_fit > best_fit:
                    best_fit = current_fit
                    best_log = self.log
                    result_file.write(str(j) + '\t' + str(best_fit) + '\n')
                self.board_reset()
            log_file = open(self.log_file, 'a')
            log_file.write(best_log + '\n')
            log_file.close()
            result_file.write('\n')
            result_file.close()

    def board_reset(self):
        self.board = BoardState(self.rows, self.cols, self.density, self.wall_density)
        self.pacman.reset(self.board)
        for i in range(len(self.ghosts)):
            self.ghosts[i].reset(self.board)
        self.game_over = False
        self.time = 2*self.rows*self.cols
        self.log = ''

    def fitness_eval(self):
        self.init_log()

        # Run turns until the game is over
        while not self.game_over and self.time > 0 and len(self.board.pills) > 0:
            self.turn()
            self.time -= 1
            self.log_turn()

        return self.pacman.score

    def turn(self):
        self.pacman.generate_action()
        for i in self.ghosts:
            i.generate_action()
        self.update_board()
        # Output board state to terminal
        # self.board.print_board()
        # print

    def update_board(self):
        # Move Ms.Pacman
        pac_pos = (self.pacman.x, self.pacman.y)
        self.board.board[self.board.pacman[1]][self.board.pacman[0]] = 'e'
        self.board.pacman = pac_pos
        self.board.board[self.pacman.y][self.pacman.x] = 'm'
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
        for i in self.ghosts:
            # Check if the ghost found Ms.Pacman
            if (i.x, i.y) == pac_pos:
                self.end_game()
            self.board.board[i.y][i.x] = 'g'
        # Update ghosts list
        for i in range(len(self.ghosts)):
            self.board.ghosts[i] = (self.ghosts[i].x, self.ghosts[i].y)

    def end_game(self):
        self.game_over = True

    def log_header(self):
        with open(self.log_file, 'w+') as log_file:
            log_file.write('Height: ' + str(self.rows) + '\n')
            log_file.write('Width: ' + str(self.cols) + '\n')
            log_file.write('Pill density: ' + str(self.density) + '\n')
            log_file.write('Random seed: ' + str(self.rand_seed) + '\n')
            log_file.write('Result log: ' + self.log_file + '\n')
            log_file.write('\n')
            log_file.close()

    def init_log(self):
        # Write initial info for log file
        self.log = str(self.cols) + '\n' + str(self.rows) + '\n'
        self.log += 'm ' + str(self.pacman.x) + ' ' + str(self.pacman.y) + '\n'
        for i in range(len(self.ghosts)):
            self.log += str(i) + ' ' + str(self.ghosts[i].x) + ' ' + str(self.ghosts[i].y) + '\n'
        for i in self.board.pills:
            self.log += 'p ' + str(i[0]) + ' ' + str(i[1]) + '\n'
        self.log += 't ' + str(self.time) + ' ' + str(self.pacman.score) + '\n'

    def log_turn(self):
        # Write info after a turn
        self.log += 'm ' + str(self.pacman.x) + ' ' + str(self.pacman.y) + '\n'
        for i in range(len(self.ghosts)):
            self.log += str(i) + ' ' + str(self.ghosts[i].x) + ' ' + str(self.ghosts[i].y) + '\n'
        self.log += 't ' + str(self.time) + ' ' + str(self.pacman.score) + '\n'


def main():
    if len(sys.argv) == 1:
        f = 'config/default.cfg'
    else:
        f = sys.argv[1]

    g = Game(filename=f)
    g.run_experiment()

if __name__ == "__main__":
    main()
