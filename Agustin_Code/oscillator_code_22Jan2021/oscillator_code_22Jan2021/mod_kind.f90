MODULE mod_kind
	! Module for standard word sizes

	USE, INTRINSIC :: iso_fortran_env, ONLY: ki=>int64, kr=>real64,output_unit
	IMPLICIT NONE
	PRIVATE
	PUBLIC :: ki,kr,output_unit, Pi2_inv ,Pi_inv, Pi2, Pi
	REAL(KIND=kr), PARAMETER :: Pi2=8.0*atan(1.0)
	REAL(KIND=kr), PARAMETER :: Pi=4.0*atan(1.0)
	REAL(KIND=kr), PARAMETER :: Pi2_inv=1.0/Pi2
	REAL(KIND=kr), PARAMETER :: Pi_inv=1.0/Pi
	
END MODULE mod_kind
