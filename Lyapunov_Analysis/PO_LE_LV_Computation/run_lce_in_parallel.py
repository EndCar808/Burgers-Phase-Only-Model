from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re

# Values to run for
# n = [64, 128, 256, 512]
n = [64, 128, 256]
k0 = [1]

# a = np.append(np.append(np.arange(0.0, 1.0, 0.05), np.arange(1.0, 2.0, 0.025)), np.arange(2.0, 2.5, 0.05))
a = np.arange(0.0, 3.5, 0.05)
# a = np.arange(0.0, 3.5, 0.05)
# a = [30, 300]
print(a)

# beta = [0.0 , 1.0];
beta = [0.0];

u0 = ["RANDOM"]

######################
##	Create command list
######################
cmdList = [['./bin/main ' + str(i) + ' ' + str(k) + ' ' + str(j) + ' ' + str(b) + ' ' + str(u)] for i in n for k in k0 for j in a for b in beta for u in u0]


print(cmdList)
print(len(cmdList))
######################
##	Run commands in parallel
######################
# Set the limit of subprocesses / threads to spawn at any one time	
procLimit = 35

# Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True) for cmd in cmdList)] * procLimit 

# Loop of grouped iterable
for processes in zip_longest(*groups): 
	for proc in filter(None, processes): # filters out 'None' fill values if procLimit does not divide evenly into cmdList
		[runCodeOutput, runCodeErr] = proc.communicate()
		print(runCodeOutput)
		print(runCodeErr)
		proc.wait()
