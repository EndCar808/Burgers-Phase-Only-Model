from subprocess import Popen, PIPE
from itertools import zip_longest
import numpy as np
import re

# Values to run for
# Values to run for
k0    = 1
N     = [64, 128, 512]
alpha = 1.4
beta  = 0.0
trans = 0
iters = 400000
u0    = "D"


######################
##	Run commands
######################
for n in N:
    cmd = './PO_Spectral_Solver/bin/main' + ' ' + str(n) + ' ' + str(k0) + ' ' + str(alpha) + ' ' + str(beta) + ' ' + u0

    print(cmd)
    # Create a subprocess class using Popen in the shell - store this in runCode
    runCode = Popen([cmd], shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)

    # Use communicate method to interact with the subprocess to send code and error ouptput 
    [runCodeOutput, runCodeErr] = runCode.communicate()
    print(runCodeOutput)
    print(runCodeErr)
    runCode.wait()

    cmd = 'python3 ./Plotting/plot_lce_triaddynamics_snapsvideo_parallel_new.py' + ' ' + str(k0) + ' ' + str(alpha) + ' ' + str(beta) + ' ' + str(iters) + ' ' + str(trans)  + ' ' + str(n) + ' ' + u0

    print(cmd)
    # Create a subprocess class using Popen in the shell - store this in runCode
    runCode = Popen([cmd], shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)

    # Use communicate method to interact with the subprocess to send code and error ouptput 
    [runCodeOutput, runCodeErr] = runCode.communicate()
    print(runCodeOutput)
    print(runCodeErr)
    runCode.wait()
