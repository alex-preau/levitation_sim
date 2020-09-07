# levitation_sim
Python code for thermophoretic levitation

The current situation is not ideal, since so many functions rely on global variables which are calcualted at runtime. Then the run_sim module has alot of functions that would preferably be in src/, but unless a couple major changes are made this is gonnna stay the case. It all runs well. Modifying the config file alone should hopefully propegate all the needed changes to adjust the number of particles 
