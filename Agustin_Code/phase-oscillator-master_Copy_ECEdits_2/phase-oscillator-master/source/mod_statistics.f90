SUBMODULE(mod_oscillators) mod_statistics
	USE mod_kind, ONLY: ki, kr, Pi, Pi2
	IMPLICIT NONE

	REAL(KIND=kr), DIMENSION(:,:), ALLOCATABLE :: du_r
	REAL(KIND=kr), DIMENSION(:), ALLOCATABLE :: bins_u, bins_du, bins_Grad
	INTEGER(KIND=kr), DIMENSION(:), ALLOCATABLE :: Hist_u, Hist_du, Hist_Grad
	REAL(KIND=kr) :: dx, u_norm, norm_Grad
	INTEGER(KIND=ki) :: write_unit_RS, write_unit_Grad, phi_unit, IS_unit

	CONTAINS

	MODULE SUBROUTINE initialize_stats()
		USE mod_functions, ONLY: theoretical_energy
		USE mod_utilities, ONLY: linspace
		IMPLICIT NONE
		INTEGER(KIND=ki) :: write_unit, i

		OPEN(NEWUNIT=write_unit_RS, FILE='realspace/RealSpace_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		OPEN(NEWUNIT=write_unit_Grad, FILE='realspace/Gradient_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		OPEN(NEWUNIT=phi_unit, FILE='instances/phi_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		OPEN(NEWUNIT=IS_unit, FILE='statistics/IS_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')

		u_norm=SQRT(theoretical_energy(a_k,N))
		norm_Grad=SQRT(theoretical_energy(a_k*k,N))
		
		IF (file_exists) THEN
			CALL load_histogram(bins_u,Hist_u,'u_'  // file_name)
			CALL load_histogram(bins_du,Hist_du,'du_' // file_name)
			CALL load_histogram(bins_Grad,Hist_Grad,'Grad_' // file_name)
			CALL load_statistics(du_r,file_name)
		ELSE
			ALLOCATE(Hist_u(n_bins))
			ALLOCATE(Hist_du(n_bins))
			ALLOCATE(Hist_Grad(n_bins))
			ALLOCATE(du_r(n_max,N))
			Hist_u   =0
			Hist_du  =0
			Hist_Grad=0
			du_r=0.0

			ALLOCATE(bins_u(n_bins+1))
			ALLOCATE(bins_du(n_bins+1))
			ALLOCATE(bins_Grad(n_bins+1))
			bins_du=set_bins(phi,a_k,INT(1,KIND=ki))
			bins_u=set_bins(phi,a_k,N)
			bins_Grad=linspace(-x_lim,x_lim,n_bins+1,.TRUE.)*norm_Grad*Pi/REAL(N,KIND=kr)
			CALL save_bins(bins_u,     'u_'  // file_name)
			CALL save_bins(bins_du,    'du_' // file_name)
			CALL save_bins(bins_Grad,'Grad_' // file_name)

			! DO i = 1,n_bins + 1
			! 	print*, "bins_0[", i -1 , "]", bins_du(i)
			! END DO
			! print*, ""
			! DO i = 1,n_bins + 1
			! 	print*, "bin_1[", i - 1, "]", bins_u(i)
			! END DO
			! print*, ""
			! DO i = 1,n_bins + 1
			! 	print*, "bins_grad[", i -1 , "]", bins_Grad(i)
			! END DO
			! print*, ""


			CALL save_stats()

			OPEN(NEWUNIT=write_unit, FILE='statistics/control_statistics_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
			WRITE(write_unit) second_moment(N,k0,k_cutoff,alpha,beta,k)
			CLOSE(write_unit)

			OPEN(NEWUNIT=write_unit, FILE='realspace/RealSpaceCoordinates_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
			dx=Pi2/REAL(2*N,KIND=kr)
			WRITE(write_unit) dx*[(REAL(i), i=0, 2*N-1)]
			CLOSE(write_unit)
		END IF

	END SUBROUTINE initialize_stats

	MODULE SUBROUTINE finalize_stats()
		IMPLICIT NONE
		CLOSE(write_unit_RS)
		CLOSE(write_unit_Grad)
		CLOSE(phi_unit)
		CLOSE(IS_unit)
		DEALLOCATE(Hist_u,Hist_du,Hist_Grad)
		DEALLOCATE(bins_u,bins_du,bins_Grad)
		DEALLOCATE(du_r)
	END SUBROUTINE finalize_stats
	
	MODULE SUBROUTINE calculate_stats()
		USE mod_functions, ONLY: RealSpace
		USE mod_utilities, ONLY: histogram
		IMPLICIT NONE
		REAL(KIND=kr), DIMENSION(2*N) :: RS, Grad, small, large
		REAL(KIND=kr), DIMENSION(n_max,N) :: IS
		INTEGER(KIND=ki) :: i
		RS=RealSpace(a_k,phi,N)
		Grad=RealSpace(a_k_Grad,phi,N)

		! DO i = 1,2*N
		! 	print*, "RS[", i -1 , "]", RS(i), "Grad[", i -1 , "]", Grad(i) 
		! END DO
		! print*, ""

		IS=increment_statistics(RS,N,n_max)
		du_r=du_r+IS
		dur_norm=dur_norm+1

		WRITE(write_unit_RS) RS/u_norm
		WRITE(write_unit_Grad) Grad/norm_Grad
		WRITE(phi_unit) phi
		WRITE(IS_unit) IS

		small = CSHIFT(RS,1)-RS
		large = CSHIFT(RS,N)-RS
		! DO i = 1,2*N
		! 	print*, "small[", i -1 , "]", small(i), "large[", i -1 , "]", large(i) 
		! END DO
		! print*, ""
		Hist_u= Hist_u +histogram(CSHIFT(RS,N)-RS, bins_u )
		Hist_du=Hist_du+histogram(CSHIFT(RS,1)-RS, bins_du)
		Hist_Grad=Hist_Grad+histogram(Grad*Pi/REAL(N,KIND=kr), bins_Grad)

		! DO i = 1,n_bins
		! 	print*, "small[", i -1 , "]", Hist_du(i)
		! END DO
		! print*, ""
		! DO i = 1,n_bins
		! 	print*, "large[", i - 1, "]", Hist_u(i)
		! END DO
		! print*, ""
		! DO i = 1,n_bins
		! 	print*, "grad[", i -1 , "]", Hist_Grad(i)
		! END DO
		! print*, ""
	END SUBROUTINE calculate_stats

	MODULE SUBROUTINE save_stats()
		IMPLICIT NONE
		CALL save_histogram(Hist_u   , 'u_'  // file_name)
		CALL save_histogram(Hist_du  ,'du_' // file_name)
		CALL save_histogram(Hist_Grad,'Grad_' // file_name)
		CALL save_statistics(du_r,file_name)
	END SUBROUTINE save_stats


	PURE FUNCTION increment_statistics(u,N,n_max)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N, n_max
		INTEGER(KIND=ki) :: fast_loop, slow_loop
		REAL(KIND=kr), INTENT(IN), DIMENSION(2*N) :: u
		REAL(KIND=kr), DIMENSION(2*N) :: u_r
		REAL(KIND=kr), DIMENSION(n_max,N) :: increment_statistics
		REAL(KIND=kr) :: dx
		dx=0.5/REAL(N,KIND=kr)!Normalize by 2*Pi
		DO CONCURRENT (slow_loop=1:N)
			u_r=CSHIFT(u,slow_loop)
			DO CONCURRENT (fast_loop=1:n_max)
				increment_statistics(fast_loop,slow_loop)=SUM((u_r-u)**fast_loop)*dx
			END DO
		END DO
	END FUNCTION increment_statistics

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
		set_bins=linspace(-x_lim,x_lim,n_bins + 1,.TRUE.)*norm
	END FUNCTION set_bins

	PURE FUNCTION second_moment(N,ZeroOffset,k0,alpha,beta,k)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N,ZeroOffset
		REAL(KIND=kr), INTENT(IN) :: alpha, beta, k0, k(N)
		INTEGER(KIND=ki) :: i
		REAL(KIND=kr) :: dx, second_moment_private
		REAL(KIND=kr), DIMENSION(N) :: second_moment, a_k
		a_k=k**(-alpha)*EXP(-beta*(k/k0)**2)
		a_k=a_k**2*2.0/REAL(N,KIND=kr)! Normalize by 2*Pi
		a_k(1:ZeroOffset)=0.0
		second_moment=0.0
		dx=Pi/REAL(N,KIND=kr)

		DO CONCURRENT (i=1:N)
			second_moment(i)=SUM(a_k*(1.0-COS(i*dx*k)))
		END DO
	END FUNCTION second_moment

	SUBROUTINE save_statistics(du_r,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		REAL(KIND=kr), INTENT(IN), DIMENSION(n_max,N) :: du_r
		INTEGER(KIND=ki) :: write_unit, i
		OPEN(NEWUNIT=write_unit, FILE='statistics/increment_statistics_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		WRITE(write_unit) du_r/REAL(dur_norm,KIND=kr)
		CLOSE(write_unit)
	END SUBROUTINE save_statistics

	SUBROUTINE load_statistics(du_r,file_name)
		IMPLICIT NONE
		CHARACTER(len=*), INTENT(IN) :: file_name
		REAL(KIND=kr), INTENT(OUT), DIMENSION(:,:), ALLOCATABLE :: du_r
		INTEGER(KIND=ki) :: read_unit, i
		ALLOCATE(du_r(n_max,N))
		OPEN(NEWUNIT=read_unit, FILE='statistics/increment_statistics_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		READ(read_unit) du_r
		CLOSE(read_unit)
		du_r=du_r*REAL(dur_norm,KIND=kr)
	END SUBROUTINE load_statistics

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
