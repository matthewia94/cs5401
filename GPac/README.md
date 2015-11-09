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
The key wall_density can be added to walls to the file it works the same as pill density but with walls.  If it is not
found in the file it will be set to 0 and there will be no walls.

The bonus can be run using the configBonus.cfg file (python2 pacman.py configBonus.cfg).  In the source code the added
code is marked with a comment that has the word bonus in it.