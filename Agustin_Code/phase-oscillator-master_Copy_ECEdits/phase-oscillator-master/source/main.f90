PROGRAM main
	USE mod_kind, ONLY: ki, kr, output_unit
	USE mod_utilities, ONLY: linspace
	USE mod_oscillators, ONLY: initialize_oscillators, finalize_oscillators, integrate, resize_oscillators
	IMPLICIT NONE

	INTEGER(KIND=ki), PARAMETER :: n_max=8, steps=10**6,transient=10**6,binning=10**3, alpha_resolution=16
	INTEGER(KIND=ki), PARAMETER :: k0_list(3) = (/ 0, 5, 7 /)
	REAL(KIND=kr) :: alpha, beta=1.0, x_lim=40.0, a, b
	INTEGER(KIND=ki) :: alpha_index, k0_index, num_args, save_unit, k0, n_bins=8000, N
	REAL(KIND=kr), DIMENSION(alpha_resolution) :: alpha_list
	CHARACTER(len=16) :: file_name

	num_args = COMMAND_ARGUMENT_COUNT()
	N=8_ki
	k0=1_ki
	a = 1.5_ki
	b = 2.65321_ki

	CALL initialize_oscillators(N, k0, a, b, n_max, n_bins, x_lim, alpha_index)
	CALL integrate(steps,binning)

! 	IF (num_args==1) THEN
! 		CALL GET_COMMAND_ARGUMENT(1,file_name)

! 		READ(file_name,*) alpha_index
! 		k0_index=(alpha_index-1)/alpha_resolution+1
! 		alpha_index=MOD(alpha_index-1,alpha_resolution)+1
! 		k0=k0_list(k0_index)

! 		IF (alpha_index>0 .AND. alpha_index<=alpha_resolution) THEN

! 			alpha_list(1:alpha_resolution/8)=linspace(REAL(0.0,KIND=kr),REAL(0.5,KIND=kr),alpha_resolution/8,.FALSE.)
! 			alpha_list(1+alpha_resolution/8:5*alpha_resolution/8)=linspace(REAL(0.5,KIND=kr),REAL(1.0,KIND=kr),alpha_resolution/2,.FALSE.)
! 			alpha_list(1+5*alpha_resolution/8:alpha_resolution)=linspace(REAL(1.0,KIND=kr),REAL(2.5,KIND=kr),3*alpha_resolution/8,.TRUE.)

! 			alpha=alpha_list(alpha_index)

! !			OPEN(NEWUNIT=save_unit,FILE='parameters/alpha_list.dat')
! !			WRITE(save_unit,*) alpha_list
! !			CLOSE(save_unit)
! !
! !			N=5_ki
! !			k0=0_ki
! !			CALL initialize_oscillators(N, k0, alpha, beta, n_max, n_bins, x_lim, alpha_index)
! !			CALL integrate(transient)
! !
! !			N=7_ki
! !			k0=0_ki
! !			CALL resize_oscillators(N, k0)
! !			CALL integrate(transient)
! !
! !			N=9_ki
! !			k0=0_ki
! !			CALL resize_oscillators(N, k0)
! !			CALL integrate(transient)
! !
! !			N=11_ki
! !			k0=0_ki
! !			CALL resize_oscillators(N, k0)
! !			CALL integrate(transient)
! !
! !			N=13_ki
! !			k0=0_ki
! !			CALL resize_oscillators(N, k0)
! !			CALL integrate(transient)
! !
! !			N=14_ki
! !			k0=k0_list(k0_index)
! !			CALL resize_oscillators(N, k0)
! !			CALL integrate(transient)
! !
! 			N=15_ki
! 			k0=k0_list(k0_index)
! 			CALL initialize_oscillators(N, k0, alpha, beta, n_max, n_bins, x_lim, alpha_index)
! !			CALL resize_oscillators(N,k0)
! !			CALL integrate(transient)
! 			CALL integrate(steps,binning)
! 			CALL finalize_oscillators()

! 		ELSE
! 			WRITE(output_unit,*) 'Wrong input index! It should be <= ',alpha_resolution
! 		END IF
! 	ELSE
! 		WRITE(output_unit,*) 'Wrong number of arguments. Expecting one integer.'
! 	END IF

	CONTAINS


END PROGRAM main
