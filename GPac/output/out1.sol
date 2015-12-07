Pacman's evaluation function
((((78.4634043285) - (38.0129465172)) - ((35.4203527185) / (ghost_dist))) - ((rand((pill_dist) * (35.766950229), ((pill_dist) * (35.766950229)) + (ghost_dist))) - (((78.4634043285) - (38.0129465172)) - ((35.4203527185) / (ghost_dist))))) - (((35.4203527185) / (ghost_dist)) + (38.9828841944))

Ghosts' evaluation function
(pac_dist) + ((ghost_dist) + (rand((ghost_dist) / ((ghost_dist) - (pac_dist)), ((pac_dist) / (ghost_dist)) + (rand(pac_dist, pac_dist)))))