from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re
from datetime import datetime

######################################
##	Define dataspace
######################################
# n  = [64, 128, 256, 512]
n     = [256]
k0    = [1]
a     = np.array([0.05, 0.55, 1.05, 1.5, 2.05, 2.5, 3.05, 3.45]) 
# a     = np.arange(0.0, 3.5, 0.05)
# a     = [30, 300]
# beta  = [0.0 , 1.0];
beta  = [0.0];
u0    = ["RANDOM"]
m_end = [1000000] #(400000, 40000, 20000, 12500, 10000, 8000)
m_itr = [1] #(1, 10, 20, 32, 40, 50)

######################################
##	Create command list
######################################
cmdList = [['./bin/main ' + str(i) + ' ' + str(k) + ' ' + str(j) + ' ' + str(b) + ' ' + str(u) + ' ' + str(m_e) + ' ' + str(m_i)] for i in n for k in k0 for j in a for b in beta for u in u0 for m_i, m_e in zip(m_itr, m_end)]

# print(cmdList)
print("Commands to perform: {}".format(len(cmdList)))



######################################
##	Run commands in parallel
######################################
# Set the limit of subprocesses / threads to spawn at any one time	
procLimit = 8

# Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmdList)] * procLimit 

# Create output objects to store process error and output
output = []
error  = []


# Loop of grouped iterable
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