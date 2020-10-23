from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re
from datetime import datetime

if __name__ == '__main__':
	######################
	##	Dataspace
	######################
	k0 = 1
	# n = [2**10, 2**11, 2**12, 2**13, 
	n = [256]
	print(n)
	a = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.45]) 
	# a = np.arange(0.0, 2.5, 0.05)
	# a = [1.25]
	print(a)
	beta = [0.0];
	print(beta)
	u0  = "RANDOM"
	iters = 10000000

	########################
	##	Create command list
	########################
	cmdList = [['./bin/main {} {} {:0.3f} {:0.3f} {} {}'.format(i, k0, j, b, u0, iters)] for i in n for j in a for b in beta]

	print(cmdList)
	print(len(cmdList))

	#############################
	##	Run commands in parallel
	#############################
	## Set the limit of subprocesses / threads to spawn at any one time	
	procLimit = 9

	# Create output objects to store process error and output
	output = []
	error  = []

	## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
	groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmdList)] * procLimit 

	## Loop of grouped iterable
	for processes in zip_longest(*groups): 
		for proc in filter(None, processes): # filters out 'None' fill values if procLimit does not divide evenly into cmdList
			
			# Communicate with process to retrive output and error
			[runCodeOutput, runCodeErr] = proc.communicate()

			# Append to output and error objects and print both to screen
			output.append(runCodeOutput)
			error.append(runCodeErr)
			print(runCodeOutput)
			print(runCodeErr)
			proc.wait()


	######################################
	##	Write Output and Error to file
	######################################
	# Get data and time
	now = datetime.now()
	d_t = now.strftime("%d%b%Y_%H:%M:%S")

	# Write to files
	with open("./output/par_run_output_{}.txt".format(d_t), "w") as file:
		for item in output:
			file.write("%s\n" % item)

	with open("./output/par_run_error_{}.txt".format(d_t), "w") as file:
		for i, item in enumerate(error):
			file.write("%s\n" % cmdList[i])
			file.write("%s\n" % item)