from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re

# Values to run for
k0    = 1
N     = [64, 128, 256, 512]
alpha = np.arange(1.25, 2.5, 0.25)
beta  = [0.0, 1.0]
iters = 400000

######################
##	Create command list
######################
cmdList = [['python3 plot_lce_triaddynamics_snapsvideo.py' + ' ' + str(k0) + ' ' + str(a) + ' ' + str(b) + ' ' + str(iters) + ' ' + str(n)] for a in alpha for b in beta for n in N]

print(cmdList[0])
######################
##	Run commands in parallel
######################
## Set the limit of subprocesses / threads to spawn at any one time	
procLimit = 10

## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True) for cmd in cmdList)] * procLimit 

## Loop of grouped iterable
for processes in zip_longest(*groups): 
	for proc in filter(None, processes): # filters out 'None' fill values if procLimit does not divide evenly into cmdList
		[runCodeOutput, runCodeErr] = proc.communicate()
		print(runCodeOutput)
		print(runCodeErr)
		proc.wait()