from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np


###############################
##	Generate Command List
###############################
N     = [2**i for i in range(6, 14)]
alpha = np.arange(0.0, 2.5, 0.05)

cmdList = [['python3 jacobian.py' + ' ' + str(n) + ' ' + str(a)] for n in N for a in alpha]

print(cmdList)
print(len(cmdList))

###############################
##	Run commands in parallel
###############################
# Set the limit of subprocesses / threads to spawn at any one time	
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
