PROGRAM main
	USE mod_kind, ONLY: ki, kr
	USE mod_oscillators, ONLY: initialize_oscillators, finalize_oscillators, integrate, resize_oscillators
	IMPLICIT NONE

	INTEGER(KIND=ki), PARAMETER :: n_max=8, steps=5*4*10**7,transient=10**5,binning=10**2,  k0=1, n_bins=8000
	REAL(KIND=kr), PARAMETER :: beta = 0.0_kr, alpha = 1.0_kr, x_lim=40.0
	INTEGER(KIND=ki) :: input_index, save_unit, N

	input_index = 1

	OPEN(NEWUNIT=save_unit,FILE='parameters/alpha_list.dat')
	WRITE(save_unit,*) alpha
	CLOSE(save_unit)

!	N=5_ki
!	CALL initialize_oscillators(N, k0, alpha, beta, n_max, n_bins, x_lim, input_index)
!	CALL integrate(transient)
!
!	N=7_ki
!	CALL resize_oscillators(N, k0)
!	CALL integrate(transient)
!
!	N=9_ki
!	CALL resize_oscillators(N, k0)
!	CALL integrate(transient)

	N=11_ki
        CALL initialize_oscillators(N, k0, alpha, beta, n_max, n_bins, x_lim, input_index)
!        CALL resize_oscillators(N,k0)
 !       CALL integrate(transient)
	CALL integrate(steps,binning)
	CALL finalize_oscillators()
STOP
END PROGRAM main
