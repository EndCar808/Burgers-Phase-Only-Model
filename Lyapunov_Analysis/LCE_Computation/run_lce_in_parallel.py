from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re

# Values to run for
n = [64, 128, 256, 512]

# a = np.append(np.append(np.arange(0.0, 1.0, 0.05), np.arange(1.0, 2.0, 0.025)), np.arange(2.0, 2.5, 0.05))
a = np.arange(0.0, 2.5, 0.05)
print(a)
print(a.shape)

beta = 0.0;

######################
##	Create command list
######################
cmdList = [['./bin/main ' + str(i) + ' ' + str(j) + ' ' + str(beta)] for i in n for j in a]



######################
##	Run commands in parallel
######################
# Set the limit of subprocesses / threads to spawn at any one time	
procLimit = 25

# Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True) for cmd in cmdList)] * procLimit 

# Loop of grouped iterable
for processes in zip_longest(*groups): 
	for proc in filter(None, processes): # filters out 'None' fill values if procLimit does not divide evenly into cmdList
		[runCodeOutput, runCodeErr] = proc.communicate()
		print(runCodeOutput)
		print(runCodeErr)
		proc.wait()