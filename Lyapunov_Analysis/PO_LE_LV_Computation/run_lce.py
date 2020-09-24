# Enda Carroll
# June 2020
# Python code to call the lce algorithm

from subprocess import Popen, PIPE
import numpy as np
import re

# Values to run for
n = [64]
a = np.arange(0.0, 2.55, 0.05)

print(a)


# Loop over values, run code and output to file
for i in n:
	for j in a:
		# The comand to run plus its agruments n and m
		cmdRun = './bin/lce_main ' + str(i) + ' ' + str(j) 

		# Create a subprocess class using Popen in the shell - store this in runCode
		runCode = Popen([cmdRun], shell = True, stdout = PIPE, stdin = PIPE, universal_newlines = True)

		# Use communicate method to interact with the subprocess to send code and error ouptput 
		[runCodeOutput, runCodeErr] = runCode.communicate()
		print(runCodeOutput)
		print(runCodeErr)
		runCode.wait()