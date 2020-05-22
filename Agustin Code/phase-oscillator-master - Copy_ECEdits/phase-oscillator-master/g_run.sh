source /data.ttf/arguedas/install/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
#cd ./source
#ifort -ipo -O3 -no-prec-div -fp-model fast=2 -xHost -heap-arrays -parallel -qopenmp -I${MKLROOT}/include/fftw -L${MKLROOT}/include -mkl mod_kind.f90 mod_utilities.f90 mod_functions.f90 mod_oscillators.f90 mod_statistics.f90 main.f90 -o main.exe
#cp main.exe ../
#rm *.mod *.exe
#cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKLROOT}/lib/intel64/
qsub g_sub.sh
