from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re

# Values to run for
k0    = 1
N     = [64, 128, 256]
alpha = np.arange(1.05, 2.45, 0.05)
# alpha = [2.0, 2.35]
beta  = [0.0]
iters = 400000
trans = 0
u0    = "RANDOM"


######################
##	Run commands
######################
for a in alpha:
    for b in beta:
        for n in N:
            cmd = 'python3 plot_lce_triaddynamics_snapsvideo_parallel_new.py' + ' ' + str(k0) + ' ' + str(a) + ' ' + str(b) + ' ' + str(iters) + ' ' + str(trans) + ' ' + str(n) + ' ' + u0

            print(cmd)
            # Create a subprocess class using Popen in the shell - store this in runCode
            runCode = Popen([cmd], shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)

            # Use communicate method to interact with the subprocess to send code and error ouptput 
            [runCodeOutput, runCodeErr] = runCode.communicate()
            print(runCodeOutput)
            print(runCodeErr)
            runCode.wait()
