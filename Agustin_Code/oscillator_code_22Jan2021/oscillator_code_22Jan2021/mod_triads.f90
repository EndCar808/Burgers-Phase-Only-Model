SUBMODULE(mod_oscillators) mod_triads
	USE mod_kind, ONLY: ki, kr, Pi
	IMPLICIT NONE

	COMPLEX(KIND=kr), ALLOCATABLE :: exp1(:)
	INTEGER(KIND=ki) :: total_triads, normalization

	CONTAINS

	MODULE SUBROUTINE triad_alloc()
		IMPLICIT NONE
		INTEGER(KIND=ki) :: ratio, read_unit,k,p
	        LOGICAL :: file_exists
		ratio=N/k0
		total_triads=k0**2*(ratio**2/4-ratio+1)
		total_triads = 0
		DO p=k0+1,N/2
			DO k=2*p,N
				total_triads =  total_triads +1
			END DO
		END DO
		ALLOCATE(exp1(total_triads))
		exp1=CMPLX(0.0_kr,0.0_kr)
		normalization=0.0_kr
        INQUIRE(FILE='statistics/triads/trig_moment_'// TRIM(file_name) //'.dat',EXIST=file_exists)
        IF (file_exists) THEN
            OPEN(NEWUNIT=read_unit, FILE='statistics/triads/trig_moment_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
            READ(read_unit) exp1
            CLOSE(read_unit)
            OPEN(NEWUNIT=read_unit, FILE='statistics/triads/normalization_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
            READ(read_unit) normalization
            CLOSE(read_unit)
        END IF
        END SUBROUTINE triad_alloc

	MODULE SUBROUTINE triad_dealloc()
		IMPLICIT NONE
		DEALLOCATE(exp1)
	END SUBROUTINE triad_dealloc

	PURE SUBROUTINE update_exp(phi,exp1,normalization)
		IMPLICIT NONE
	        REAL(KIND=kr), INTENT(IN), DIMENSION(N) :: phi
		INTEGER(KIND=ki) :: p,k,control
		COMPLEX(KIND=kr), DIMENSION(N), INTENT(INOUT) :: exp1
		INTEGER(KIND=ki), INTENT(INOUT) :: normalization
		control=1
		normalization=normalization+1_ki
		DO p=k0+1,N/2
			DO k=2*p,N
				exp1(control)=exp1(control)+EXP(CMPLX(0.0_kr,phi(p)+phi(k-p)-phi(k)))
				control=control+1
			END DO
		END DO
	END SUBROUTINE update_exp

	MODULE SUBROUTINE trig_moment()
		IMPLICIT NONE
		CALL update_exp(phi,exp1,normalization)
	END SUBROUTINE trig_moment

	PURE FUNCTION triad_skewness()
		IMPLICIT NONE
		INTEGER(KIND=ki) :: p,j,control, r
		REAL(KIND=kr) :: dx
		REAL(KIND=kr) :: triad_skewness(N), second_moment(N)
		
		dx = Pi / REAL(N,KIND=kr)
		
		DO CONCURRENT (p=1:N)
			second_moment(p)=SUM(a_k**2*(1.0-COS(p*dx*k)))
		END DO
		
		triad_skewness(1:N)=0.0_kr
		DO r=1,N
    		control=1
    		DO p=k0+1,N/2
    			DO j=2*p,N
                    triad_skewness(r)=triad_skewness(r)+3.0_kr*a_k(j)*a_k(j-p)*a_k(p)*(SIN(r*j*dx)-SIN(r*p*dx)-SIN(r*(j-p)*dx))*ABS(exp1(control))
                    control=control+1
    			END DO
    		END DO
    	END DO
    	triad_skewness=triad_skewness/SQRT(second_moment)**3/normalization
	END FUNCTION triad_skewness

	MODULE SUBROUTINE trig_moment_save()
		IMPLICIT NONE
		INTEGER(KIND=ki) :: write_unit
		OPEN(NEWUNIT=write_unit, FILE='statistics/triads/trig_moment_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		WRITE(write_unit) exp1
		CLOSE(write_unit)
		OPEN(NEWUNIT=write_unit, FILE='statistics/triads/normalization_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
		WRITE(write_unit) normalization
		CLOSE(write_unit)
!		OPEN(NEWUNIT=write_unit, FILE='statistics/triads/triad_skewness_'// TRIM(file_name) //'.dat',FORM='UNFORMATTED')
!		WRITE(write_unit) triad_skewness()
!		CLOSE(write_unit)
    END SUBROUTINE trig_moment_save

END SUBMODULE mod_triads
