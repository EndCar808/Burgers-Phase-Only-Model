from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re
from datetime import datetime

######################################
##	Define dataspace
######################################
n   = [128, 256, 512, 1024]
# n      = [1024]
k0     = [1]
a = np.arange(0.0, 2.51, 0.05)
# a      = np.array([1.3, 2.00])  #np.array([0.00, 0.50, 1.00, 1.25, 1.3, 1.4, 1.5, 1.75, 2.00, 2.5]) 
# a      = np.delete(np.arange(0.0, 2.51, 0.01), range(0, np.arange(0.0, 2.51, 0.01).shape[0], 5)) 
# a      = np.array([0.02, 0.06, 0.17, 0.23, 0.26, 0.29, 0.31, 0.33, 0.34, 0.36, 0.37, 0.38, 0.39, 0.41, 0.71, 0.76, 0.82, 0.87, 0.89, 0.98, 1.02, 1.03, 1.04, 1.06, 1.16, 1.22, 1.24, 1.32, 1.39, 1.41, 1.42, 1.43, 1.44, 1.46, 1.56, 1.61, 1.64, 1.66, 1.84, 1.88, 1.91, 1.92, 1.94, 2.02, 2.08, 2.17, 2.18, 2.23, 2.29, 2.33, 2.38])
# beta   = [0.0 , 1.0]
beta   = [0.0]
u0     = ["RANDOM"]
m_end  = [8000] #(400000, 40000, 20000, 12500, 10000, 8000)
m_itr  = [50] #(1, 10, 20, 32, 40, 50)
# numLEs = [ for nn in n]



######################################
##	Create command list
######################################
cmdList = [['./bin/main ' + str(i) + ' ' + str(k) + ' ' + str(j) + ' ' + str(b) + ' ' + str(u) + ' ' + str(m_e) + ' ' + str(m_i) + ' ' + str(int(i / 2 - k0[0]))] for i in n for k in k0 for j in a for b in beta for u in u0 for m_i, m_e in zip(m_itr, m_end)]

print(cmdList)
print("Commands to perform: {}".format(len(cmdList)))



#####################################
#	Run commands in parallel
#####################################
## Set the limit of subprocesses / threads to spawn at any one time	
procLimit = 26

## Create grouped iterable of subprocess calls to Popen() - see grouper recipe in itertools
groups = [(Popen(cmd, shell = True, stdout = PIPE, stdin = PIPE, stderr = PIPE, universal_newlines = True) for cmd in cmdList)] * procLimit 

## Create output objects to store process error and output
output = []
error  = []


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
## Get data and time
now = datetime.now()
d_t = now.strftime("%d%b%Y_%H:%M:%S")

## Write to files
with open("./output/par_run_output_{}.txt".format(d_t), "w") as file:
	for item in output:
		file.write("%s\n" % item)

with open("./output/par_run_error_{}.txt".format(d_t), "w") as file:
	for i, item in enumerate(error):
		file.write("%s\n" % cmdList[i])
		file.write("%s\n" % item)