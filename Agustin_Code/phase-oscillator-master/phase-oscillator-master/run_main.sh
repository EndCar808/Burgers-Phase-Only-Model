source /scratch.local/arguedas/installs/intel/compilers_and_libraries/linux/bin/compilervars.{sh,csh} -arch intel64 -platform linux

cd ./source
ifort -ipo -O3 -no-prec-div -fp-model fast=2 -xHost -heap-arrays -parallel -qopenmp -I${MKLROOT}/include/fftw -L${MKLROOT}/include -mkl mod_kind.f90 mod_utilities.f90 mod_functions.f90 mod_oscillators.f90 mod_statistics.f90 main.f90 -o main.exe
cp main.exe ../
rm *.exe
rm *.mod
cd ..
chmod +x main.exe
#for i in {1..48}
#do
#  ./main.exe $i
#done
./main.exe 43
rm main.exe
