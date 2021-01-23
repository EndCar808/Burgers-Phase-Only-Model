SUBMODULE(mod_oscillators) mod_statistics
	USE mod_kind, ONLY: ki, kr, Pi, Pi2
	USE, INTRINSIC :: ISO_C_BINDING
	USE omp_lib
	IMPLICIT NONE
	INCLUDE 'fftw3.f03'

	REAL(KIND=kr), DIMENSION(:,:), ALLOCATABLE :: du_r
	REAL(KIND=kr), DIMENSION(:), ALLOCATABLE :: bins_u, bins_du, bins_d2u, bins_d4u, bins_d8u, bins_Grad
	REAL(KIND=kr), DIMENSION(:), ALLOCATABLE :: small_scale_stats
	INTEGER(KIND=kr), DIMENSION(:), ALLOCATABLE :: Hist_u, Hist_du, Hist_d2u, Hist_d4u, Hist_d8u, Hist_Grad
	INTEGER(KIND=kr), ALLOCATABLE :: omp_indices(:,:)
	REAL(KIND=kr) :: dx, u_norm, norm_Grad
	INTEGER(KIND=ki) :: write_unit_RS, write_unit_Grad, phi_unit, IS_unit, small_scale_stats_unit

	CONTAINS

	MODULE SUBROUTINE initialize_stats()
		USE mod_functions, ONLY: theoretical_energy
		USE mod_utilities, ONLY: linspace, tile_indices
		IMPLICIT NONE
		INTEGER(KIND=ki) :: write_unit, i, indices(2)

		OPEN(NEWUNIT=write_unit_RS, FILE='realspace/RealSpace_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		OPEN(NEWUNIT=write_unit_Grad, FILE='realspace/Gradient_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		OPEN(NEWUNIT=phi_unit, FILE='instances/phi_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
!		OPEN(NEWUNIT=IS_unit, FILE='statistics/IS_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')

		u_norm=SQRT(theoretical_energy(a_k,N))
		norm_Grad=SQRT(theoretical_energy(a_k*k,N))
		
		IF (file_exists) THEN
			CALL load_histogram(bins_u,Hist_u,'u_'  // file_name)
			CALL load_histogram(bins_du,Hist_du,'du_' // file_name)
			CALL load_histogram(bins_d2u,Hist_d2u,'d2u_' // file_name)
			CALL load_histogram(bins_d4u,Hist_d4u,'d4u_' // file_name)
			CALL load_histogram(bins_d8u,Hist_d8u,'d8u_' // file_name)
			CALL load_histogram(bins_Grad,Hist_Grad,'Grad_' // file_name)
			CALL load_statistics(du_r,file_name)
			CALL load_small_scale_stats(small_scale_stats,file_name)
		ELSE
			ALLOCATE(Hist_u(n_bins))
			ALLOCATE(Hist_du(n_bins))
			ALLOCATE(Hist_d2u(n_bins))
			ALLOCATE(Hist_d4u(n_bins))
			ALLOCATE(Hist_d8u(n_bins))
			ALLOCATE(Hist_Grad(n_bins))
			ALLOCATE(du_r(n_max,N_FFTW/2))
			ALLOCATE(small_scale_stats(n_max))
			Hist_u   =0
			Hist_du  =0
			Hist_d2u =0
			Hist_d4u =0
			Hist_d8u =0
			Hist_Grad=0
			du_r=0.0
			small_scale_stats=0.0

			ALLOCATE(bins_u(n_bins+1))
			ALLOCATE(bins_du(n_bins+1))
			ALLOCATE(bins_d2u(n_bins+1))
			ALLOCATE(bins_d4u(n_bins+1))
			ALLOCATE(bins_d8u(n_bins+1))
			ALLOCATE(bins_Grad(n_bins+1))
			bins_du=set_du_bins()!set_bins(phi,a_k,INT(1,KIND=ki))
			bins_d2u=set_du_bins()!set_bins(phi,a_k,INT(2,KIND=ki))
			bins_d4u=set_du_bins()!set_bins(phi,a_k,INT(4,KIND=ki))
			bins_d8u=set_du_bins()!set_bins(phi,a_k,INT(8,KIND=ki))
			bins_u=set_u_bins()!set_bins(phi,a_k,N)
			bins_Grad=set_Grad_bins()!linspace(-x_lim,x_lim,n_bins+1,.TRUE.)*norm_Grad*Pi/REAL(N,KIND=kr)
			CALL save_bins(bins_u,     'u_'  // file_name)
			CALL save_bins(bins_du,    'du_' // file_name)
			CALL save_bins(bins_d2u,    'd2u_' // file_name)
			CALL save_bins(bins_d4u,    'd4u_' // file_name)
			CALL save_bins(bins_d8u,    'd8u_' // file_name)
			CALL save_bins(bins_Grad,'Grad_' // file_name)

			CALL save_stats()

			OPEN(NEWUNIT=write_unit, FILE='statistics/control_statistics_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
			WRITE(write_unit) second_moment(N,k0,k_cutoff,alpha,beta,k)
			CLOSE(write_unit)

			OPEN(NEWUNIT=write_unit, FILE='realspace/RealSpaceCoordinates_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
			dx=Pi2/REAL(N_FFTW,KIND=kr)
			WRITE(write_unit) dx*[(REAL(i), i=0, N_FFTW-1)]
			CLOSE(write_unit)
		END IF
		!$OMP PARALLEL PRIVATE(indices), SHARED(omp_indices)
		IF (OMP_GET_THREAD_NUM() == 0) THEN
			ALLOCATE(omp_indices(OMP_GET_NUM_THREADS(),2))
		END IF
		!$OMP BARRIER
		indices=tile_indices(N_FFTW/2)
		!$OMP CRITICAL
		omp_indices(OMP_GET_THREAD_NUM()+1,:) = indices(:)
		!$OMP END CRITICAL
		!$OMP END PARALLEL
END SUBROUTINE initialize_stats

	MODULE SUBROUTINE finalize_stats()
		IMPLICIT NONE
		CLOSE(write_unit_RS)
		CLOSE(write_unit_Grad)
		CLOSE(phi_unit)
!		CLOSE(IS_unit)
		DEALLOCATE(Hist_u,Hist_du,Hist_d2u,Hist_d4u,Hist_d8u,Hist_Grad)
		DEALLOCATE(bins_u,bins_du,bins_Grad)
		DEALLOCATE(du_r)
		DEALLOCATE(small_scale_stats)
		DEALLOCATE(omp_indices)
	END SUBROUTINE finalize_stats
	
	MODULE SUBROUTINE calculate_stats()
		USE mod_functions, ONLY: RealSpace
		USE mod_utilities, ONLY: histogram
		IMPLICIT NONE
		REAL(KIND=kr), DIMENSION(N_FFTW) :: RS, Grad
		REAL(KIND=kr), DIMENSION(n_max,N_FFTW/2) :: IS
		REAL(KIND=kr), DIMENSION(n_max) :: SS_stats
		RS=RealSpace(a_k,phi,N)
		Grad=RealSpace(a_k_Grad,phi,N)

		IS=increment_statistics(RS,N_FFTW,n_max)
		du_r=du_r+IS
		dur_norm=dur_norm+1

		small_scale_stats = small_scale_stats + get_small_scale_stats(RS,N_FFTW,n_max)
		small_scale_stats_norm = small_scale_stats_norm+1

		WRITE(write_unit_RS) RS/u_norm
		WRITE(write_unit_Grad) Grad/norm_Grad
		WRITE(phi_unit) phi
!		WRITE(IS_unit) IS

		Hist_u= Hist_u +histogram(CSHIFT(RS,N)-RS, bins_u )
		Hist_du=Hist_du+histogram(CSHIFT(RS,1)-RS, bins_du)
		Hist_d2u=Hist_d2u+histogram(CSHIFT(RS,2)-RS, bins_du)
		Hist_d4u=Hist_d4u+histogram(CSHIFT(RS,4)-RS, bins_du)
		Hist_d8u=Hist_d8u+histogram(CSHIFT(RS,8)-RS, bins_du)
		Hist_Grad=Hist_Grad+histogram(Grad*Pi2/REAL(N_FFTW,KIND=kr), bins_Grad)
	END SUBROUTINE calculate_stats

	MODULE SUBROUTINE save_stats()
		IMPLICIT NONE
		CALL save_histogram(Hist_u   , 'u_'  // file_name)
		CALL save_histogram(Hist_du  ,'du_' // file_name)
		CALL save_histogram(Hist_d2u  ,'d2u_' // file_name)
		CALL save_histogram(Hist_d4u  ,'d4u_' // file_name)
		CALL save_histogram(Hist_d8u  ,'d8u_' // file_name)
		CALL save_histogram(Hist_Grad,'Grad_' // file_name)
		CALL save_statistics(du_r,file_name)
		CALL save_small_scale_stats(small_scale_stats,file_name)
	END SUBROUTINE save_stats

	PURE FUNCTION get_small_scale_stats(u,N_FFTW,n_max)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N_FFTW, n_max
		INTEGER(KIND=ki) :: loop_index
		REAL(KIND=kr), INTENT(IN), DIMENSION(N_FFTW) :: u
		REAL(KIND=kr), DIMENSION(N_FFTW) :: u_r
		REAL(KIND=kr), DIMENSION(n_max) :: get_small_scale_stats
		REAL(KIND=kr) :: dx
		dx=1.0_kr/REAL(N_FFTW,KIND=kr)
		get_small_scale_stats(:) = 0.0_kr
		u_r=CSHIFT(u,1)-u
		DO loop_index=1,n_max
			get_small_scale_stats(loop_index)=SUM(u_r**loop_index)*dx
		END DO
	END FUNCTION get_small_scale_stats

	FUNCTION increment_statistics(u,N_FFTW,n_max)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N_FFTW, n_max
		INTEGER(KIND=ki) :: fast_loop, slow_loop
		REAL(KIND=kr), INTENT(IN), DIMENSION(N_FFTW) :: u
		REAL(KIND=kr), DIMENSION(2*N) :: u_r
		REAL(KIND=kr), DIMENSION(n_max,N_FFTW/2) :: increment_statistics
		REAL(KIND=kr), ALLOCATABLE :: increment_statistics_l(:,:)
		REAL(KIND=kr) :: dx
		dx=1.0/REAL(N_FFTW,KIND=kr)
		increment_statistics(:,:) = 0.0_kr
		!$OMP PARALLEL PRIVATE(increment_statistics_l,fast_loop,slow_loop,u_r), SHARED(increment_statistics,omp_indices,u,dx)
		ALLOCATE(increment_statistics_l(n_max,omp_indices(OMP_GET_THREAD_NUM()+1,1):omp_indices(OMP_GET_THREAD_NUM()+1,2)))
		DO slow_loop=omp_indices(OMP_GET_THREAD_NUM()+1,1),omp_indices(OMP_GET_THREAD_NUM()+1,2)
			u_r=CSHIFT(u,slow_loop)
			DO fast_loop=1,n_max
				increment_statistics_l(fast_loop,slow_loop)=SUM((u_r-u)**fast_loop)*dx
			END DO
		END DO
		!$OMP CRITICAL
		increment_statistics(:,omp_indices(OMP_GET_THREAD_NUM()+1,1):omp_indices(OMP_GET_THREAD_NUM()+1,2)) = increment_statistics_l(:,omp_indices(OMP_GET_THREAD_NUM()+1,1):omp_indices(OMP_GET_THREAD_NUM()+1,2))
		DEALLOCATE(increment_statistics_l)
		!$OMP END CRITICAL
		!$OMP END PARALLEL
	END FUNCTION increment_statistics
! Set bins procedures
	FUNCTION set_bins(phi,a_k,r)
		USE mod_functions, ONLY: RealSpace
		USE mod_utilities, ONLY: linspace
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: r
		REAL(KIND=kr), DIMENSION(N) , INTENT(IN):: phi,a_k
		REAL(KIND=kr), DIMENSION(2*N) :: RS, du_r
		REAL(KIND=kr) :: norm
		REAL(KIND=kr), DIMENSION(n_bins+1) :: set_bins
		RS=RealSpace(a_k,phi,N)
		du_r=CSHIFT(RS,r)-RS
		norm=SQRT(SUM(du_r**2)/REAL(N,KIND=kr)*0.5)!Normalizing by 2Pi lead to removing a Pi and inserting a 0.5
		set_bins=linspace(-x_lim,x_lim,n_bins,.TRUE.)*norm
	END FUNCTION set_bins

	FUNCTION set_u_bins()
		USE mod_utilities, ONLY: linspace
		IMPLICIT NONE
		REAL(KIND=kr), DIMENSION(n_bins+1) :: set_u_bins
		set_u_bins=linspace(-1.0_kr,1.0_kr,n_bins+1,.TRUE.)*2.0_kr*SUM(a_k)/SQRT(REAL(N_FFTW,KIND=kr))
	END FUNCTION set_u_bins

	FUNCTION set_du_bins()
		USE mod_utilities, ONLY: linspace
		IMPLICIT NONE
		REAL(KIND=kr), DIMENSION(n_bins+1) :: set_du_bins
		set_du_bins=linspace(-1.0_kr,1.0_kr,n_bins+1,.TRUE.)*4.0_kr*SUM(a_k)/SQRT(REAL(N_FFTW,KIND=kr))
	END FUNCTION set_du_bins

	FUNCTION set_Grad_bins()
		USE mod_utilities, ONLY: linspace
		IMPLICIT NONE
		REAL(KIND=kr), DIMENSION(n_bins+1) :: set_Grad_bins
		set_Grad_bins=linspace(-1.0_kr,1.0_kr,n_bins+1,.TRUE.)*2.0_kr*SUM(a_k*k)/SQRT(REAL(N_FFTW,KIND=kr))*Pi2/REAL(N_FFTW,KIND=kr)
	END FUNCTION set_Grad_bins
! Save second moment
	PURE FUNCTION second_moment(N,ZeroOffset,k0,alpha,beta,k)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N,ZeroOffset
		REAL(KIND=kr), INTENT(IN) :: alpha, beta, k0, k(N)
		INTEGER(KIND=ki) :: i
		REAL(KIND=kr) :: dx, second_moment_private
		REAL(KIND=kr), DIMENSION(N_FFTW/2) :: second_moment
		REAL(KIND=kr), DIMENSION(N) :: a_k
		a_k=k**(-alpha)*EXP(-beta*(k/k0)**2)
		a_k=a_k*a_k*4.0/REAL(N_FFTW,KIND=kr)! Normalize by 2*Pi
		a_k(1:ZeroOffset)=0.0
		second_moment=0.0
		dx=Pi2/REAL(N_FFTW,KIND=kr)

		DO i=1,N_FFTW/2
			second_moment(i)=SUM(a_k*(1.0-COS(i*dx*k)))
		END DO
	END FUNCTION second_moment
! IO small scale stats
	SUBROUTINE save_small_scale_stats(small_scale_stats,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		REAL(KIND=kr), INTENT(IN), DIMENSION(n_max) :: small_scale_stats
		INTEGER(KIND=ki) :: write_unit
		OPEN(NEWUNIT=write_unit, FILE='statistics/SS_stats_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		WRITE(write_unit) small_scale_stats/REAL(small_scale_stats_norm,KIND=kr)
		CLOSE(write_unit)
	END SUBROUTINE save_small_scale_stats

	SUBROUTINE load_small_scale_stats(small_scale_stats,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		REAL(KIND=kr), INTENT(OUT), DIMENSION(:), ALLOCATABLE :: small_scale_stats
		INTEGER(KIND=ki) :: read_unit
		ALLOCATE(small_scale_stats(n_max))
		OPEN(NEWUNIT=read_unit, FILE='statistics/SS_stats_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		READ(read_unit) small_scale_stats
		CLOSE(read_unit)
		small_scale_stats=small_scale_stats*REAL(small_scale_stats_norm,KIND=kr)
	END SUBROUTINE load_small_scale_stats
! IO increment statistics
	SUBROUTINE save_statistics(du_r,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		REAL(KIND=kr), INTENT(IN), DIMENSION(n_max,N_FFTW/2) :: du_r
		INTEGER(KIND=ki) :: write_unit
		OPEN(NEWUNIT=write_unit, FILE='statistics/increment_statistics_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		WRITE(write_unit) du_r/REAL(dur_norm,KIND=kr)
		CLOSE(write_unit)
	END SUBROUTINE save_statistics

	SUBROUTINE load_statistics(du_r,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		REAL(KIND=kr), INTENT(OUT), DIMENSION(:,:), ALLOCATABLE :: du_r
		INTEGER(KIND=ki) :: read_unit
		ALLOCATE(du_r(n_max,N_FFTW/2))
		OPEN(NEWUNIT=read_unit, FILE='statistics/increment_statistics_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		READ(read_unit) du_r
		CLOSE(read_unit)
		du_r=du_r*REAL(dur_norm,KIND=kr)
	END SUBROUTINE load_statistics
! IO histograms
	SUBROUTINE save_histogram(Hist,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		INTEGER(KIND=kr), INTENT(IN), DIMENSION(n_bins) :: Hist
		INTEGER(KIND=ki) :: write_unit
		OPEN(NEWUNIT=write_unit, FILE='histograms/Hist_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		WRITE(write_unit) Hist
		CLOSE(write_unit)
	END SUBROUTINE save_histogram

	SUBROUTINE save_bins(bins,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		REAL(KIND=kr), INTENT(IN), DIMENSION(n_bins+1) :: bins
		INTEGER(KIND=ki) :: write_unit
		OPEN(NEWUNIT=write_unit, FILE='histograms/bins_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		WRITE(write_unit) bins
		CLOSE(write_unit)
	END SUBROUTINE save_bins

	SUBROUTINE load_histogram(bins,Hist,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		REAL(KIND=ki), INTENT(OUT), DIMENSION(:), ALLOCATABLE :: bins
		INTEGER(KIND=ki), INTENT(OUT), DIMENSION(:), ALLOCATABLE :: Hist
		INTEGER(KIND=ki) :: read_unit
		ALLOCATE(Hist(n_bins))
		ALLOCATE(bins(n_bins+1))
		OPEN(NEWUNIT=read_unit, FILE='histograms/Hist_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		READ(read_unit) Hist
		CLOSE(read_unit)
		OPEN(NEWUNIT=read_unit, FILE='histograms/bins_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		READ(read_unit) bins
		CLOSE(read_unit)
	END SUBROUTINE load_histogram
END SUBMODULE mod_statistics
