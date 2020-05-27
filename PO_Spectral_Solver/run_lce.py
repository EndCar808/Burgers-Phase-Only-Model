# Enda Carroll
# June 2019
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
		cmdRun = './bin/main_lce ' + str(i) + ' ' + str(j) 

		# Create a subprocess class using Popen in the shell - store this in runCode
		runCode = Popen([cmdRun], shell = True, stdout = PIPE, stdin = PIPE)

		# Use communicate method to interact with the subprocess to send code and error ouptput 
		[runCodeOutput, runCodeErr] = runCode.communicate()
		print(runCodeOutput)
		runCode.wait()