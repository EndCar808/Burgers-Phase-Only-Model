MODULE mod_oscillators
	USE mod_kind, ONLY: ki, kr, output_unit
	IMPLICIT NONE

	PRIVATE
	PUBLIC :: initialize_oscillators, finalize_oscillators, integrate, resize_oscillators

	INTEGER(KIND=ki) :: N, N_FFTW, n_exp, k0, n_max, n_bins, dur_norm, alpha_index, small_scale_stats_norm
	INTEGER(KIND=ki) :: progress_unit
	REAL(KIND=kr) :: alpha, beta
	REAL(KIND=kr), ALLOCATABLE, DIMENSION(:) :: k, a_k, phi, omega_k
	REAL(KIND=kr) :: k_cutoff
	REAL(KIND=kr) :: dt, omega_dt(2), x_lim
	REAL(KIND=kr) :: start, finish
	COMPLEX(KIND=kr), ALLOCATABLE, DIMENSION(:) :: a_k_Grad
	CHARACTER(len=16) :: file_name
	LOGICAL :: file_exists

	INTERFACE
		MODULE SUBROUTINE initialize_stats()
		END SUBROUTINE

		MODULE SUBROUTINE finalize_stats()
		END SUBROUTINE

		MODULE SUBROUTINE calculate_stats()
		END SUBROUTINE

		MODULE SUBROUTINE save_stats()
		END SUBROUTINE
		
		MODULE SUBROUTINE triad_alloc()
		END SUBROUTINE

		MODULE SUBROUTINE triad_dealloc()
		END SUBROUTINE
		
		MODULE SUBROUTINE trig_moment()
		END SUBROUTINE
		
		MODULE SUBROUTINE trig_moment_save()
		END SUBROUTINE
	END INTERFACE

	INTERFACE integrate
		PROCEDURE integrate_no_save
		PROCEDURE integrate_save
	END INTERFACE

	CONTAINS

	SUBROUTINE initialize_oscillators(a_n_exp, a_k0, a_alpha, a_beta, a_n_max, a_n_bins, a_x_lim, a_alpha_index)
		USE mod_kind, ONLY: PI2
		USE mod_functions, ONLY: initialize_fft, omega_max_min, set_timescale, test_fftw
		USE omp_lib, ONLY: omp_get_wtime
        IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: a_n_exp, a_k0, a_n_max, a_n_bins, a_alpha_index
		REAL(KIND=kr), INTENT(IN) :: a_alpha, a_beta, a_x_lim
		INTEGER(KIND=ki) :: i

		start=omp_get_wtime()

		alpha_index=a_alpha_index
		file_name= get_file_name(a_n_exp,a_k0,alpha_index)

		INQUIRE(FILE='parameters/Parameters_'// TRIM(file_name) //'.dat', EXIST=file_exists)
		IF (file_exists) THEN
			CALL load_parameters()
			ALLOCATE(phi(1:N))
			CALL load_phi('end_phi_'// TRIM(file_name))
			WRITE(output_unit,*) 'Loaded pre-existing phi.'

		ELSE
			n_exp	=a_n_exp
			N       =2_ki**(n_exp-1)-1
			N_FFTW	=2_ki**n_exp
			k0      =a_k0
			alpha   =a_alpha
			beta    =a_beta
			n_max   =a_n_max
			n_bins  =a_n_bins
			dur_norm=0
			small_scale_stats_norm=0
			x_lim   =a_x_lim
			ALLOCATE(phi(1:N))
			CALL RANDOM_NUMBER(phi)
			phi=phi*PI2
			phi(1:k0)=0.0
		END IF

		OPEN(NEWUNIT=progress_unit,FILE='progress/progress_'// TRIM(file_name) //'.dat')
		90  FORMAT ('Doing n_exp=',I8,', N=',I8,', N_FFTW=',I8,', k0=',I5,', alpha=',f7.5,', beta=',f7.5,', n_max=',I5 &
& ,', n_bins=',I5,', dur_norm=',I9,', x_lim=',f10.5,', small_scale_stats_norm=',I9,', dur_norm=',I9)
		WRITE(progress_unit, 90 ) n_exp,N,N_FFTW,k0,alpha,beta, n_max, n_bins, dur_norm, x_lim, small_scale_stats_norm,dur_norm
		WRITE( output_unit , 90 ) n_exp,N,N_FFTW,k0,alpha,beta, n_max, n_bins, dur_norm, x_lim, small_scale_stats_norm,dur_norm

		ALLOCATE(       k(1:N))
		ALLOCATE(     a_k(1:N))
		ALLOCATE( omega_k(1:N))
		ALLOCATE(a_k_Grad(1:N))

		k_cutoff=0.5*REAL(N,KIND=kr)
		k(1:N)=[(REAL(i), i=1, N)]
		a_k=k**(-alpha)*exp(-beta*(k/k_cutoff)**2)
		omega_k=-k/a_k
		k(1:k0)=0.0
		a_k(1:k0)=0.0
		a_k_Grad=a_k*k*CMPLX(0.0,1.0)
		omega_k(1:k0)=0.0
!		dt=set_timescale(a_k,N)

		omega_dt=1.0/omega_max_min(N,k0,alpha,beta)
		dt = omega_dt(1) * 0.5_kr
		WRITE(progress_unit,*) 'Fastest timescale is ', omega_dt(1), ', slowest timescale is ', omega_dt(2),'.'		
		WRITE( output_unit ,*) 'Fastest timescale is ', omega_dt(1), ', slowest timescale is ', omega_dt(2),'.'
		WRITE(progress_unit,*) 'Timestep set to ', dt, '.'
		WRITE( output_unit ,*) 'Timestep set to ', dt, '.'
		CALL initialize_fft(N_FFTW)
		CALL test_fftw(N,a_k)

	END SUBROUTINE initialize_oscillators

	SUBROUTINE finalize_oscillators()
		USE mod_functions, ONLY: destroy_fft
		USE omp_lib, ONLY: omp_get_wtime
		IMPLICIT NONE
        DEALLOCATE(k, a_k, phi, omega_k, a_k_Grad)
		CALL destroy_fft()
		finish=omp_get_wtime()
		WRITE( output_unit ,'("Time = ",f10.3," seconds.")') finish-start
		WRITE(progress_unit,'("Time = ",f10.3," seconds.")') finish-start
		CLOSE(progress_unit)
	END SUBROUTINE finalize_oscillators

	SUBROUTINE resize_oscillators(N_new,k0_new)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N_new, k0_new
		REAL(KIND=kr), ALLOCATABLE, DIMENSION(:) :: phi_old
		ALLOCATE(phi_old(1:SIZE(phi)))
		phi_old=phi
		CALL finalize_oscillators()
		CALL initialize_oscillators(N_new, k0_new, alpha, beta, n_max, n_bins, x_lim, alpha_index)
		phi(1:k0)=0.0_kr
		phi(k0+1:SIZE(phi_old))=phi_old(k0+1:SIZE(phi_old))
		DEALLOCATE(phi_old)
	END SUBROUTINE resize_oscillators

	SUBROUTINE save_parameters()
		IMPLICIT NONE
		INTEGER(KIND=ki) :: write_unit
		OPEN(NEWUNIT=write_unit,FILE='parameters/Parameters_'// TRIM(file_name) //'.dat')
		WRITE(write_unit,*) n_exp
		WRITE(write_unit,*) N
		WRITE(write_unit,*) N_FFTW
		WRITE(write_unit,*) k0
		WRITE(write_unit,*) alpha
		WRITE(write_unit,*) beta
		WRITE(write_unit,*) n_max
		WRITE(write_unit,*) n_bins
		WRITE(write_unit,*) x_lim
		WRITE(write_unit,*) dur_norm
		WRITE(write_unit,*) small_scale_stats_norm
		CLOSE(write_unit)
	END SUBROUTINE save_parameters

	SUBROUTINE load_parameters()
		IMPLICIT NONE
		INTEGER(KIND=ki) :: read_unit
		OPEN(NEWUNIT=read_unit,FILE='parameters/Parameters_'// TRIM(file_name) //'.dat')
		READ(read_unit,*) n_exp
		READ(read_unit,*) N
		READ(read_unit,*) N_FFTW
		READ(read_unit,*) k0
		READ(read_unit,*) alpha
		READ(read_unit,*) beta
		READ(read_unit,*) n_max
		READ(read_unit,*) n_bins
		READ(read_unit,*) x_lim
		READ(read_unit,*) dur_norm
		READ(read_unit,*) small_scale_stats_norm
		CLOSE(read_unit)
	END SUBROUTINE load_parameters

	SUBROUTINE save_phi(file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		INTEGER(KIND=ki) :: write_unit
		OPEN(NEWUNIT=write_unit, FILE='instances/'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		WRITE(write_unit) phi
		CLOSE(write_unit)
	END SUBROUTINE save_phi

	SUBROUTINE load_phi(file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		INTEGER(KIND=ki) :: read_unit
		OPEN(NEWUNIT=read_unit, FILE='instances/'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		READ(read_unit) phi
		CLOSE(read_unit)
	END SUBROUTINE load_phi

	SUBROUTINE integrate_no_save(steps)
		USE mod_functions, ONLY: rk_4_integrate_fft
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: steps
		INTEGER(KIND=ki) :: i, progress
		progress=steps/100
		IF (progress==0) THEN
			progress=1
		END IF
		WRITE(progress_unit,*) 'Doing a no-save integration of length', steps, '.'
		WRITE( output_unit ,*) 'Doing a no-save integration of length', steps, '.'
		DO i=1_ki,steps
			IF (MOD(i,progress)==0) THEN
				WRITE( output_unit ,*) INT(REAL(i)/REAL(steps)*100.0), '%'
				WRITE(progress_unit,*) INT(REAL(i)/REAL(steps)*100.0), '%'
			END IF
			phi=rk_4_integrate_fft(phi,a_k,omega_k,N,dt)
		END DO
	END SUBROUTINE integrate_no_save

	SUBROUTINE integrate_save(steps,binning)
		USE mod_functions, ONLY: rk_4_integrate_fft
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: steps,binning
		INTEGER(KIND=ki) :: i, progress,control
		CALL save_phi('start_phi_'// TRIM(file_name))
		control=1
		progress=steps/100
		IF (progress==0) THEN
			progress=1
		END IF
		CALL triad_alloc()
		CALL initialize_stats()
		WRITE(progress_unit,*) 'Doing a save integration of length', steps, '.'
		WRITE( output_unit ,*) 'Doing a save integration of length', steps, '.'
		DO i=1_ki,steps
			IF (MOD(i,progress)==0) THEN
				WRITE( output_unit ,*) INT(REAL(i)/REAL(steps)*100.0), '%'
				WRITE(progress_unit,*) INT(REAL(i)/REAL(steps)*100.0), '%'
			END IF
			IF (control==binning) THEN
				control=0
				CALL calculate_stats()
				CALL save_stats()
				CALL save_parameters()
				CALL save_phi('instant_phi_'// TRIM(file_name))
				CALL trig_moment()
                               CALL trig_moment_save()
			END IF
			phi=rk_4_integrate_fft(phi,a_k,omega_k,N,dt)
			control=control+1
		END DO
	CALL trig_moment()
        CALL trig_moment_save()
        CALL triad_dealloc()
	CALL save_phi('end_phi_'// TRIM(file_name))
	CALL save_parameters()
	CALL finalize_stats()
	END SUBROUTINE integrate_save

	FUNCTION get_file_name(N,k0,alpha_index)
		INTEGER(KIND=ki), INTENT(IN) :: N, k0, alpha_index
		CHARACTER(len=16) :: file_number, file_N, file_k0, get_file_name
			WRITE(file_N,'(I3)') N
			WRITE(file_k0,'(I3)') k0
			WRITE(file_number,'(I3)') alpha_index
			get_file_name= 'N' // TRIM(ADJUSTL(file_N)) // '_k' // TRIM(ADJUSTL(file_k0)) // '_' // TRIM(ADJUSTL(file_number))
	END FUNCTION get_file_name

END MODULE mod_oscillators
