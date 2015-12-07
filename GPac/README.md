The script can be run with a default configuration using:
    python2 pacman.py
    
The script will create a solution file at output/out.res and a log file at output/out.log.

If you wish to change the config file used simply add an argument when running the script:
    python2 pacman.py <path to config>
    
The config file should use a json format with key values:
height: height of board
width: width of board
density: percentage of cells to be pills (b/w 0 and 100)
rand_seed: the number to seed random with, if it is null the time will be used
runs: number of runs for in the experiment
fit_evals: number of fitness evaluations in each run
log_file: path to the log file
result_file: path to the result file

Bonus Info:
The bonus is outdated and unused