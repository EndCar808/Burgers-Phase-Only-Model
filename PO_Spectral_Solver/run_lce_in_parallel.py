from subprocess import Popen, PIPE
import numpy as np
import re

# Values to run for
n = [64, 128, 256, 512]
a = np.arange(0.0, 2.55, 0.05)

print(a)

beta = 0.0

######################
##	Create command list
######################
for j in a:
	cmdList = [['./bin/lce_main ' + str(i) + ' ' + str(j) + ' ' + str(beta)] for i in n]

	print(cmdList)

	######################
	##	Run commands in parallel
	######################
	procsList = [Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True) for cmd in cmdList]

	for proc in procsList:
		[runCodeOutput, runCodeErr] = proc.communicate()
		print(runCodeOutput)
		print(runCodeErr)
		proc.wait()