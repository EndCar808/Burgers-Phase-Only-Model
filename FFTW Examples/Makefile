jac:
	gcc -I/usr/include/hdf5/serial/ -I/usr/include/gsl/ jacobian.c -o jaco -lfftw3 -lm

conv:
	gcc -I/usr/include/hdf5/serial/ -I/usr/include/gsl/ fft_convolution.c -o convo -lfftw3 -lm

rhs:
	gcc -I/usr/include/hdf5/serial/ -I/usr/include/gsl/ po_rhs.c -o porhs -fopenmp -lpthread -lgsl -lgslcblas -lfftw3_omp -lfftw3 -lm