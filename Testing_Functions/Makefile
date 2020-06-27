jac:
	gcc -I/usr/include/hdf5/serial/ -I/usr/include/gsl/ jacobian.c -o jaco -lfftw3 -lm

conv:
	gcc -I/usr/include/hdf5/serial/ -I/usr/include/gsl/ fft_convolution.c -o convo -lfftw3 -lm

rhs:
	gcc -I/usr/include/hdf5/serial/ -I/usr/include/gsl/  -I../Libraries/lapack-3.9.0/CBLAS/include/ -I../Libraries/lapack-3.9.0/LAPACKE/include/ po_rhs.c -L../Libraries/lapack-3.9.0/ -o porhs -fopenmp -lpthread -lgsl -lgslcblas -llapacke -llapack -lblas -lgfortran -lfftw3_omp -lfftw3 -lm