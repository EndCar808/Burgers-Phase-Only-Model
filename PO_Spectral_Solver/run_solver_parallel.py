from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re
from datetime import datetime

if __name__ == '__main__':
	######################
	##	Dataspace
	######################
	N     = [512, 1024, 2048] #[256, 512, 1024, 2048, 4096, 8192, 16384]
	k0    = 1
	# alpha = np.array([0.0, 0.5, 1.0, 1.2, 1.5, 1.7, 2.0, 2.2, 2.5]) 
	# alpha = np.array([0.00, 0.50, 1.00, 1.25, 1.5, 1.75, 2.00, 2.5]) 
	alpha = [1.25] #np.arange(0.0, 2.51, 0.05)[~np.isin(np.arange(0.0, 2.51, 0.05), np.arange(0.0, 2.51, 0.1))] #np.arange(0.0, 2.51, 0.05)
	beta  = [0.0]	
	u0    = "RANDOM"
	iters = int(4e5)
	save_steps = 1
	comp_steps = 1

	########################
	##	Create command list
	########################
	cmdList = [['./bin/main -n {} -k {} -a {:0.3f} -b {:0.3f} -u {} -t {} -s {} -c {}'.format(n, k0, a, b, u0, iters, save_steps, comp_steps)] for n in N for a in alpha for b in beta]

	print(cmdList)
	print(len(cmdList))

	#############################
	##	Run commands in parallel
	#############################
	## Set the limit of subprocesses / threads to spawn at any one time	
	procLimit = np.maximum(len(N), len(alpha))
	print("Process Created = {}".format(procLimit))

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