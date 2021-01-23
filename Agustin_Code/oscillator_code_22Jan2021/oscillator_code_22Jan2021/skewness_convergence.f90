PROGRAM main
USE mod_kind, ONLY: ki, kr, Pi, output_unit
USE omp_lib
IMPLICIT NONE

INTEGER(KIND=ki) :: N, k0, write_unit, i
REAL(KIND=kr) :: alpha, beta, r, skewness

alpha = 0.0_kr
beta  = 0.0_kr
k0    = 1_ki

OPEN(NEWUNIT=write_unit, FILE='statistics/skewness.dat')
DO i = 8_ki,20_ki
	N = 2_ki**i
	k0=1_ki!2_ki**(i-8)
	r = Pi/REAL(N,KIND=kr)
	skewness = get_skewness(N, k0, alpha, beta, r)
	WRITE(write_unit,*) N, skewness, skewness/N
	WRITE(output_unit,*) N, skewness, skewness/N
END DO
CLOSE(write_unit)
STOP

CONTAINS

REAL(KIND=kr) FUNCTION get_skewness(N, k0, alpha, beta, r) RESULT(skewness)
	IMPLICIT NONE
	INTEGER(KIND=ki), INTENT(IN) :: N, k0
	REAL(KIND=kr), INTENT(IN) :: alpha, beta, r
	INTEGER(KIND=ki) :: i, j
	REAL(KIND=kr) :: k_cutoff, a_k(1:N), k(1:N), second_moment

	k_cutoff  = 0.5_kr*REAL(N,KIND=kr)
	k(1:N)    = [(REAL(i), i=1, N)]
	a_k       = k**(-alpha)*EXP(-beta*(k/k_cutoff)**2)
	a_k(1:k0) = 0.0_kr

	CALL OMP_SET_NUM_THREADS(8_ki)

	second_moment=0.0_kr
	!$OMP PARALLEL
	!$OMP DO REDUCTION(+:second_moment) PRIVATE(i)
	DO i = k0+1_ki,N
		second_moment = second_moment + a_k(i)**2*(1.0_kr-COS(r*i)) 
	END DO
	!$OMP END DO
	!$OMP END PARALLEL
	
	skewness = 0.0_kr
	!$OMP PARALLEL
	!$OMP DO REDUCTION(+:skewness) PRIVATE(i,j)
	DO i = 2_ki*k0+2_ki,N
		DO j = k0+1_ki,i/2_ki
			skewness = skewness + 3.0_kr*a_k(i)*a_k(j)*a_k(i-j)*(SIN(r*i)-SIN(r*j)-SIN(r*(i-j)))
		END DO
	END DO
	!$OMP END DO
	!$OMP END PARALLEL
	skewness = skewness / SQRT(second_moment)**3_ki
END FUNCTION get_skewness

END PROGRAM main
