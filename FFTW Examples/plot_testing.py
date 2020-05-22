#!/usr/bin/env python    
# line above specifies which program should be called to run this script - called shebang
# the way it is called above (and not #!/user/bin/python) ensures portability amongst Unix distros

######################
##	Library Imports
######################
import matplotlib
matplotlib.use('TkAgg') # Use this backend for displaying plots in window
# matplotlib.use('Agg') # Use this backend for writing plots to file

import h5py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt




######################
##	Read input file
######################
# Check if program was given directory for input file
if (len(sys.argv) == 1):
	print("No Input file specified, Error.\n")
	sys.exit()
else :
	inpt_dir_data = str(sys.argv[1])


# print directory of input file to screen
print("\n\tData Directory: %s\n" % inpt_dir_data)#




# Open input file in given directory for reading in data
if (os.path.isfile(inpt_dir_data + "/Test_Data.h5")):
	HDFfileData = h5py.File(inpt_dir_data + "/Test_Data.h5", 'r')
else:
	print("Cannot open %s data file, Error.\n" % "/Test_Data.h5")
	sys.exit()



	

# Size parameters
num_r_modes = HDFfileData['FTSinewave'].shape[0]

u = HDFfileData["Sinewave"]
u_z = HDFfileData["FTSinewave"]
ift_u = HDFfileData["IFTSinewave"]



dx = 2*np.pi / num_r_modes;
x  = np.arange(0, 2*np.pi, dx)

plt.plot(x, np.fft.ifft(np.fft.fft(u)))
plt.show()


plt.plot(x, np.fft.ifft(u_z), '-*', x, u, '-o', x, np.sin(x))
plt.legend(("FT", "Sin", "np.sin"))
plt.show()