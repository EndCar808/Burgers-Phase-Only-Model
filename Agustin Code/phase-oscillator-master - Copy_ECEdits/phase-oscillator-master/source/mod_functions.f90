MODULE mod_functions
	USE mod_kind, ONLY: ki, kr
	USE, INTRINSIC :: ISO_C_BINDING
	USE omp_lib
	IMPLICIT NONE
	INCLUDE 'fftw3.f03'
	
	PRIVATE
	PUBLIC :: rk_4_integrate, rk_4_integrate_fft,initialize_fft,destroy_fft, &
&	set_timescale,convolve,convolve_fft,RealSpace, theoretical_energy, omega_max_min


	INTERFACE RealSpace
		PROCEDURE RealSpace_r
		PROCEDURE RealSpace_c
	END INTERFACE

	! Size of 1D transform
	INTEGER(KIND=ki) :: N_fftw_convolve
	INTEGER(KIND=ki) :: N_fftw_c2r

	! Execution status
	INTEGER(KIND=ki) :: STATUS = 0

	! Data arrays
	COMPLEX(KIND=kr), ALLOCATABLE :: x_cmplx_c2r(:)
	REAL(KIND=kr), ALLOCATABLE :: x_real_c2r(:)
	COMPLEX(KIND=kr), ALLOCATABLE :: x_cmplx_convolve(:)
	REAL(KIND=kr), ALLOCATABLE :: x_real_convolve(:)

	! FFTW plan handle
	INTEGER(KIND=ki) :: plan_r2c_convolve = 0
	INTEGER(KIND=ki) :: plan_c2r_convolve = 0
	INTEGER(KIND=ki) :: plan_c2r = 0

	!Normalization of FFT
	REAL(KIND=kr) :: Norm_FFT_r2c

	CONTAINS

	PURE FUNCTION omega_max_min(N,k0,alpha,beta)
		INTEGER(KIND=ki), INTENT(IN) :: N, k0
		REAL(KIND=kr), INTENT(IN) :: alpha, beta
		REAL(KIND=kr), DIMENSION(-N:N) :: a_k
		REAL(KIND=kr), DIMENSION(k0+1:N) :: omega
		REAL(KIND=kr) :: k_cutoff, k_r
		INTEGER(KIND=ki) :: k,p
		REAL(KIND=kr) :: omega_max_min(2)

		k_cutoff=0.5_kr*REAL(N,KIND=kr)
		a_k=0.0_kr
		DO CONCURRENT (k=k0+1:N)
			k_r=REAL(k,KIND=kr)
			a_k(k)=k_r**(-alpha)*EXP(-beta*(k_r/k_cutoff)**2)
			a_k(-k)=a_k(k)
		END DO

		omega=0.0_kr

		DO CONCURRENT (k=k0+1:N)
			DO CONCURRENT (p=k-N:N)
				omega(k)=omega(k)+a_k(p)*a_k(k-p)

			END DO
			omega(k)=omega(k)*REAL(k,KIND=kr)/a_k(k)
		END DO
		omega_max_min(1)=MAXVAL(omega)
		omega_max_min(2)=MINVAL(omega)
	END FUNCTION omega_max_min

	PURE REAL(KIND=kr) FUNCTION set_timescale(a_k,N) RESULT(dt)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
        	REAL(KIND=kr), INTENT(IN), DIMENSION(N) :: a_k
		INTEGER(KIND=ki) :: i
		dt=0.0
		DO CONCURRENT (i=1:N-1)
			dt = dt + a_k(i)*a_k(N-i)
		END DO
		dt=dt*REAL(N)/a_k(N)
		dt=1.0/dt
	END FUNCTION set_timescale

	SUBROUTINE initialize_fft(N)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
		INTEGER(KIND=C_INT) :: stat,threads
		threads=16
		CALL OMP_SET_NUM_THREADS(threads)
		N_fftw_convolve=4*N
		N_fftw_c2r=2*N
		Norm_FFT_r2c=SQRT(REAL(N_fftw_c2r,KIND=kr))
		stat = fftw_init_threads()
		IF (stat .EQ. 0) STOP 'Error initializing threads'
		CALL dfftw_plan_with_nthreads(threads)
		ALLOCATE(x_cmplx_convolve(N_fftw_convolve/2+1))
		ALLOCATE(x_real_convolve(N_fftw_convolve))
		ALLOCATE(x_cmplx_c2r(N_fftw_c2r/2+1))
		ALLOCATE(x_real_c2r(N_fftw_c2r))
		CALL dfftw_plan_dft_r2c_1d(plan_r2c_convolve, N_fftw_convolve, x_real_convolve, x_cmplx_convolve, FFTW_MEASURE)
		CALL dfftw_plan_dft_c2r_1d(plan_c2r_convolve, N_fftw_convolve, x_cmplx_convolve, x_real_convolve, FFTW_MEASURE)	
		CALL dfftw_plan_dft_c2r_1d(plan_c2r, N_fftw_c2r, x_cmplx_c2r, x_real_c2r, FFTW_MEASURE)
	END SUBROUTINE initialize_fft

	SUBROUTINE destroy_fft()
		CALL dfftw_destroy_plan(plan_r2c_convolve)
		CALL dfftw_destroy_plan(plan_c2r_convolve)
		CALL dfftw_destroy_plan(plan_c2r)
		CALL fftw_cleanup_threads()
		DEALLOCATE(x_real_convolve, x_cmplx_convolve, x_real_c2r, x_cmplx_c2r)
	END SUBROUTINE destroy_fft

	FUNCTION RealSpace_r(a_k,phi,N)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
		REAL(KIND=kr), DIMENSION(N), INTENT(IN) :: phi, a_k
		REAL(KIND=kr), DIMENSION(N_fftw_c2r) :: RealSpace_r
		RealSpace_r=RealSpace_c(a_k*CMPLX(1.0,0.0),phi,N)
	END FUNCTION RealSpace_r

	FUNCTION RealSpace_c(a_k,phi,N)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
		REAL(KIND=kr), DIMENSION(N), INTENT(IN) :: phi
		COMPLEX(KIND=kr), DIMENSION(N), INTENT(IN) :: a_k
		REAL(KIND=kr), DIMENSION(N_fftw_c2r) :: RealSpace_c
		x_cmplx_c2r(1)=CMPLX(0.0,0.0)
		x_cmplx_c2r(2:N+1)=a_k*EXP(CMPLX(0.0,1.0)*phi)
		CALL dfftw_execute_dft_c2r(plan_c2r, x_cmplx_c2r, x_real_c2r)
		RealSpace_c=x_real_c2r/Norm_FFT_r2c!You should actually normalize here by sqrt of this!!
	END FUNCTION RealSpace_c

	FUNCTION convolve_fft(a_e_phi,N)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
		COMPLEX(KIND=kr), DIMENSION(N), INTENT(IN) :: a_e_phi
		COMPLEX(KIND=kr) :: convolve_fft(N)

		x_cmplx_convolve=CMPLX(0.0,0.0)
		x_cmplx_convolve(2:N+1)=a_e_phi(1:N)
		
		CALL dfftw_execute_dft_c2r(plan_c2r_convolve, x_cmplx_convolve, x_real_convolve)
		x_real_convolve=x_real_convolve**2
		CALL dfftw_execute_dft_r2c(plan_r2c_convolve, x_real_convolve, x_cmplx_convolve)
		
		convolve_fft(1:N)=x_cmplx_convolve(2:N+1)/REAL(N_fftw_convolve,KIND=kr)!Correctly normalized, check against normal convolve
	END FUNCTION convolve_fft

	FUNCTION ODE_fft(phi,a_k,omega_k,N)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
		REAL(KIND=kr), INTENT(IN), DIMENSION(N) :: phi, a_k,omega_k
		COMPLEX(KIND=kr), DIMENSION(N) :: Conv, a_e_phi, ODE_fft

		a_e_phi=a_k*EXP(CMPLX(0.0,1.0)*phi)
		Conv=convolve_fft(a_e_phi,N)
		ODE_fft=omega_k*REAL(Conv*EXP(CMPLX(0.0,-1.0)*phi))
	END FUNCTION ODE_fft

	FUNCTION rk_4_integrate_fft(phi,a_k,omega_k,N,dt)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
		REAL(KIND=kr), INTENT(IN), DIMENSION(N) :: phi, a_k,omega_k
		REAL(KIND=kr), DIMENSION(N) :: rk_4_integrate_fft
		REAL(KIND=kr), DIMENSION(N) :: k1,k2,k3,k4
		REAL(KIND=kr), INTENT(IN) :: dt

		k1=ODE_fft(phi,a_k,omega_k,N)
		k2=ODE_fft(phi+dt*k1*0.5,a_k,omega_k,N)
		k3=ODE_fft(phi+dt*k2*0.5,a_k,omega_k,N)
		k4=ODE_fft(phi+dt*k3,a_k,omega_k,N)
		rk_4_integrate_fft=phi+dt/6.0*(k1+2.0*k2+2.0*k3+k4)
	END FUNCTION rk_4_integrate_fft

	PURE FUNCTION convolve(u,v,N)
		IMPLICIT NONE
		INTEGER(KIND=ki) :: k_loop,p_loop
		INTEGER(KIND=ki), INTENT(IN) :: N
		COMPLEX(KIND=kr), INTENT(IN), DIMENSION(-N:N) :: u, v
		COMPLEX(KIND=kr) :: convolve(N)

		DO CONCURRENT (k_loop=1:N)

		    convolve(k_loop)=CMPLX(0.0,0.0)

		    DO p_loop =-N+k_loop, N

				convolve(k_loop)=convolve(k_loop)+u(p_loop)*v(k_loop-p_loop)
		    END DO
		END DO

	END FUNCTION convolve

	PURE FUNCTION ODE(phi,a_k,omega_k,N)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
		REAL(KIND=kr), INTENT(IN), DIMENSION(N) :: phi, a_k,omega_k
		INTEGER(KIND=ki) :: k_loop,p_loop

		COMPLEX(KIND=kr), DIMENSION(-N:N) :: u_k
		COMPLEX(KIND=kr) :: Conv(N)
		REAL(KIND=kr) :: x,y

		REAL(KIND=kr) :: ODE(N)

		u_k(0)=(0.0,0.0)

		DO CONCURRENT (k_loop=1:N)
			x=a_k(k_loop)*cos(phi(k_loop))
			y=a_k(k_loop)*sin(phi(k_loop))
			u_k( k_loop)=cmplx(x,y)
			u_k(-k_loop)=cmplx(x,-y)
		END DO

		Conv=convolve(u_k,u_k,N)
		ODE=omega_k*REAL(Conv*EXP(CMPLX(0.0,-1.0)*phi))

	END FUNCTION ODE

	PURE FUNCTION rk_4_integrate(phi,a_k,omega_k,N,dt)
		IMPLICIT NONE
		INTEGER(KIND=ki), INTENT(IN) :: N
		INTEGER(KIND=ki) :: i
		REAL(KIND=kr), INTENT(IN), DIMENSION(1:N) :: phi, a_k,omega_k
		REAL(KIND=kr), DIMENSION(N) :: rk_4_integrate
		REAL(KIND=kr), DIMENSION(N) :: k1,k2,k3,k4
		REAL(KIND=kr), INTENT(IN) :: dt

		k1=ODE(phi,a_k,omega_k,N)
		k2=ODE(phi+dt*k1*0.5,a_k,omega_k,N)
		k3=ODE(phi+dt*k2*0.5,a_k,omega_k,N)
		k4=ODE(phi+dt*k3,a_k,omega_k,N)
		rk_4_integrate=phi+dt/6.0*(k1+2.0*k2+2.0*k3+k4)
	END FUNCTION rk_4_integrate

	PURE REAL(KIND=kr) FUNCTION theoretical_energy(a_k,N)
		INTEGER(KIND=ki), INTENT(IN) :: N
		REAL(KIND=kr), INTENT(IN), DIMENSION(N) :: a_k
		theoretical_energy=2.0*SUM(a_k**2)/REAL(N,KIND=kr)!Normalize over 2.0*Pi
	END FUNCTION theoretical_energy


END MODULE mod_functions
