from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re

if __name__ == '__main__':
	######################
	##	Dataspace
	######################
	k0 = 1
	# n = [2**10, 2**11, 2**12, 2**13, 
	n = [4096, 8192, 16384]
	print(n)
	# a = [0.0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
	# a = np.arange(0.0, 2.5, 0.05)
	a = [1.25]
	print(a)
	beta = [0.0];
	print(beta)
	u0  = "ALIGNED"

	######################
	##	Create command list
	######################
	cmdList = [['./bin/main ' + str(i) + ' ' + str(k0) + ' ' + str(j) + ' ' + str(b) + ' ' + str(u0)] for i in n for j in a for b in beta]

	print(cmdList)
	print(len(cmdList))

	#########################
	##	Run commands in parallel
	#########################
	## Set the limit of subprocesses / threads to spawn at any one time	
	procLimit = 3

	## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
	groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True) for cmd in cmdList)] * procLimit 

	## Loop of grouped iterable
	for processes in zip_longest(*groups): 
		for proc in filter(None, processes): # filters out 'None' fill values if procLimit does not divide evenly into cmdList
			[runCodeOutput, runCodeErr] = proc.communicate()
			print(runCodeOutput)
			print(runCodeErr)
			proc.wait()