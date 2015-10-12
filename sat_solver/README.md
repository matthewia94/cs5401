The script can be run with a default configuration using:
    python sat_solver.py

The script will create a solution file at output/out.log, a log file at output/out.sol and a fitness graph file at 
output/out.png.

If you wish to change the config file used simply add an argument when running the script:
    python sat_solver <path to config>
    
The config file should use a json format.
parent_sel can have the values "k-tournament", "fps", or "uniform-random"
survival_sel can have the values "k-tournament", "fps", "truncation", or "uniform-random"
mutation is the mutation rate
r is the r-elistist number of solutions to keep
survival strategy is either "," or "+"
if seeded is true you should specify a seed_file