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
plt.style.use('classic')



######################
##	Read input file
######################
if (len(sys.argv) == 1):
	print("No Input file Directory specified, Error.\n")
	sys.exit()
else:
	HDFfileData = h5py.File(str(sys.argv[1]))
	
# Check if program was given directory for input file
# if (len(sys.argv) == 1):
# 	print("No Input file Directory specified, Error.\n")
# 	sys.exit()
# else:
# 	inpt_dir_data = str(sys.argv[1])
	

# # print directory of input file to screen
# print("\n\tData Directory: %s\n" % inpt_dir_data)


# # Open input file in given directory for reading in data
# if (os.path.isfile(inpt_dir_data + "/*.h5")):
# 	HDFfileData = h5py.File(inpt_dir_data + "/*.h5", 'r')
# else:
# 	print("Cannot open %s data file, Error.\n" % "/Runtime_Data.h5")
# 	sys.exit()

# if (os.path.isfile(inpt_dir_data + "/Matlab_data.h5")):
# 	HDFfileDataMatlab = h5py.File(inpt_dir_data + "/Matlab_data.h5", 'r')
# else:
# 	print("Cannot open %s data file, Error. Continuing regardless! \n" % "/Runtime_Data.h5")



# print input file name to screen
print("\n\nData File: %s\n" % str(sys.argv[1]))


phases = HDFfileData['Phases']
triads = HDFfileData['Triads']
time   = HDFfileData['Time']
amps   = HDFfileData['Amps']
R      = HDFfileData['PhaseOrderR']
Phi    = HDFfileData['PhaseOrderPhi']
lce    = HDFfileData['LCE']

# Reshape triads
tdims     = triads.attrs['Triad_Dims']
triadsnew = np.reshape(triads, np.append(triads.shape[0], tdims[0, :]))



print(phases.shape)



print("\n\n")
print(triads.shape)
print("\n\n")


print("\n\n")
print(amps.shape)
print("\n\n")

print("\n\n")
print(time.shape)

print("\n\n")

print("\n\n")
print(lce.shape)

print("\n\n")


print("\n\n")
print(amps[:])
print("\n\n")

print(time[:, 0])

triad_2k = np.zeros((len(time), len(phases[0, :]) - 2))
for k in range(2, len(phases[0, :]) - 2):
	triad_2k[:, k] = phases[:, 2] + phases[:, k] - phases[:, k + 2]

plt.figure()
plt.plot(time[:, 0], R[:, 0], 'b-', time[:, 0], Phi[:, 0], 'g-')
plt.legend(np.arange(2, len(phases[0, :])))
plt.title(r'Kuramoto')
plt.grid(True)


plt.figure()
for i in range(2, len(phases[0, :])):
	plt.plot(time[:, 0], phases[:, i], '-')
plt.legend(np.arange(2, len(phases[0, :])))
plt.title(r'Phases Tseries')
plt.grid(True)

print(np.extract(triads[:, :] != -10, triads))
plt.figure()
plt.hist(np.extract(triads[:, :] != -10, triads).flatten(), bins = 1000)
plt.xlim(-0.1, 2*np.pi + 0.1)



plt.figure()
for i in range(2, len(triadsnew[0, :, 0])):
	plt.plot(time[:, 0], triad_2k[:, i], '-')
plt.grid(True)
plt.title(r'Triads')


plt.show()




######################
##	Output Directory
######################
# Get current working directory
out_dir_data = os.getcwd()


# create output directory if doesn't exist
# if (os.path.isdir(out_dir_data + "/TestPlots")):
# 	out_dir_data += "/TestPlots" # add new dir to path
# 	os.mkdir(out_dir_data)       # make the new dir
# 	if (os.path.isdir(out_dir_data)): 
# 		print("Output directory: %s" % out_dir_data)
# 	else:
# 		print("Failed to create output directory, Error! \n")
# 		sys.exit()
# else:
# 	out_dir_data += "/TestPlots"

# check if output directory exist
# if (os.path.isdir(out_dir_data + "/TestPlots")):
# 	out_dir_data += "/TestPlots"
# else:
# 	print("Failed to detect output directory, Error! \n")
# 	sys.exit()

# print("\t Output Directory: %s\n\n" % out_dir_data)




# ######################
# ##	Open datasets
# ######################
# num_r_modes  = HDFfileData['ComplexModesFull'].shape[1]

# # num_r_modes = (num_c_modes-1)*2
# num_tsteps  = HDFfileData['ComplexModesFull'].shape[0]

# # read in MATLAB data to test against
# if (HDFfileDataMatlab):
# 	k1 = np.array(np.transpose(HDFfileDataMatlab['k1']), dtype = complex)
# 	k2 = np.array(np.transpose(HDFfileDataMatlab['k2']), dtype = complex)
# 	k3 = np.array(np.transpose(HDFfileDataMatlab['k3']), dtype = complex)
# 	k4 = np.array(np.transpose(HDFfileDataMatlab['k4']), dtype = complex)
# 	U_Z_real = HDFfileDataMatlab['uhatreal']
# 	U_Z_imag = HDFfileDataMatlab['uhatimag']
# 	# recreate complex modes from real and imaginary data 
# 	U_Z = np.array(np.transpose(U_Z_real[:, :]), dtype = complex) # matlab stores in column major so need to transpose
# 	U_Z.imag = np.transpose(U_Z_imag[:, :])


# print("\nNumber of tsteps: %s\n" % num_tsteps)


# # data
# u_z = HDFfileData['ComplexModesFull']
# u   = HDFfileData['RealModesFull']




# ######################
# ##	Prelim data calc
# ######################
# dx = 2*np.pi / num_r_modes;
# x  = np.arange(0, 2*np.pi, dx)



# print(U_Z.shape)
# print(u_z.shape)
# print(u.shape)

# print(k1[0, :])

# print("\n\n")
# print(u_z[2, :])

# print("\n\n")



# plt.plot(x, u[1, :], '*-', x, k1[0, :])
# # plt.plot(x, u[1, :] - k1[0, :])
# # plt.plot(x, np.fft.ifft(u_z[1, :]), '*-', x, np.fft.ifft(k1[0, :]))
# plt.legend(('Solver main', 'Matlab'))
# plt.show()

######################
##	Plotting
######################
# for t in range(num_tsteps):
# 	print("Snapshot Indx = %d\n" % t)
# 	# plt.clf()

# 	plt.plot(x, np.fft.ifft(u_z[t, :]).real, x, np.fft.ifft(U_Z[t, :]).real)
# 	plt.legend(('C', 'Matlab'))
	
# 	plt.title('Real Space Solution')
# 	plt.xlabel(r'$x$')
# 	plt.ylabel(r'$u$')
# 	plt.grid(which='major', axis='both')
# 	plt.savefig(out_dir_data +'/Snap_{:05d}.png'.format(t), format='png', dpi = 800)	
# 	plt.close()
	



