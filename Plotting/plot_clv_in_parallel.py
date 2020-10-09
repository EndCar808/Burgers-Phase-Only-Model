from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import sys
import os
import re

############################
##	Parameter space
############################
k0     = 1
N      = [256]
alpha  = np.arange(0.00, 3.5, 0.05)
beta   = 0.0
iters  = 4000000
trans  = 1000
m_end  = 80000
m_iter = 50
u0     = "RANDOM"




############################
##	Create command list
############################
input_dir  = "/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data/RESULTS"

cmdList = []
for a in alpha:
    for n in N:
        if os.path.exists(input_dir + "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/CLVData_ITERS[{},{},{}]_TRANS[{}].h5".format(n, k0, a, beta, u0, iters, m_end, m_iter, trans)):
            cmdList.append(['python3 plot_clv_over_alpha.py' + ' ' + str(k0) + ' ' + str(a) + ' ' + str(beta) + ' ' + str(iters) + ' ' + str(trans) + ' ' + str(n) + ' ' + u0 + ' ' + str(m_end) + ' ' + str(m_iter)])
        elif os.path.exists(input_dir + "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/CLVData_ITERS[{},{},{}]_TRANS[{}].h5".format(n, k0, a, beta, u0, iters, m_end, m_iter, 100)):
            cmdList.append(['python3 plot_clv_over_alpha.py' + ' ' + str(k0) + ' ' + str(a) + ' ' + str(beta) + ' ' + str(iters) + ' ' + str(100) + ' ' + str(n) + ' ' + u0 + ' ' + str(m_end) + ' ' + str(m_iter)])
        elif os.path.exists(input_dir + "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/CLVData_ITERS[{},{},{}]_TRANS[{}].h5".format(n, k0, a, beta, u0, iters, m_end, m_iter, 10000)):
            cmdList.append(['python3 plot_clv_over_alpha.py' + ' ' + str(k0) + ' ' + str(a) + ' ' + str(beta) + ' ' + str(iters) + ' ' + str(10000) + ' ' + str(n) + ' ' + u0 + ' ' + str(m_end) + ' ' + str(m_iter)])
        elif os.path.exists(input_dir + "/RESULTS_N[{}]_k0[{}]_ALPHA[{:0.3f}]_BETA[{:0.3f}]_u0[{}]/CLVData_ITERS[{},{},{}]_TRANS[{}].h5".format(n, k0, a, beta, u0, iters, m_end, m_iter, 100000)):
            cmdList.append(['python3 plot_clv_over_alpha.py' + ' ' + str(k0) + ' ' + str(a) + ' ' + str(beta) + ' ' + str(iters) + ' ' + str(100000) + ' ' + str(n) + ' ' + u0 + ' ' + str(m_end) + ' ' + str(m_iter)])
        else:
            print("No file: a = {}".format(a))




#######################################################
##	Check cmdlist creation & Run commands in parallel
#######################################################
if len(cmdList) != alpha.shape[0]:
	print("Length of command list: {}/{}".format(len(cmdList), alpha.shape[0]))
	print("Some files missing...Exiting")
	sys.exit()
else:	
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