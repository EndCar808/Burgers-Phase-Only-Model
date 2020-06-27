!10**6 !for N=2^16, steps=10^5 takes 4118 seconds


MODULE mod_utilities
	USE mod_kind, ONLY: ki, kr
	IMPLICIT NONE

	PRIVATE
	PUBLIC :: linspace, histogram

	CONTAINS

	PURE FUNCTION linspace(r_start,r_end,i_num,endpoint)
		IMPLICIT NONE
		REAL(KIND=kr), INTENT(IN) :: r_start,r_end
		INTEGER(KIND=ki), INTENT(IN) :: i_num
		LOGICAL, INTENT(IN) :: endpoint
		REAL(KIND=kr), DIMENSION(i_num) :: linspace
		INTEGER(KIND=ki) :: i
		IF (endpoint .EQV. .TRUE.) THEN
			DO CONCURRENT (i=1:i_num)
				linspace(i)=r_start+(r_end-r_start)*REAL(i-1,KIND=kr)/REAL(i_num-1,KIND=kr)
			END DO
		ELSE
			DO CONCURRENT (i=1:i_num)
				linspace(i)=r_start+(r_end-r_start)*REAL(i-1,KIND=kr)/REAL(i_num,KIND=kr)
			END DO			
		END IF
	END FUNCTION linspace

	FUNCTION histogram(data_in, bins)
		IMPLICIT NONE
		REAL(KIND=kr), DIMENSION(:), INTENT(IN) :: bins
		REAL(KIND=kr), DIMENSION(:) :: data_in
		INTEGER(KIND=ki), DIMENSION(:), ALLOCATABLE :: histogram
		INTEGER(KIND=ki) :: loop_outer, loop_inner, data_size, num_bins
		REAL(KIND=kr) :: x_0, x_1, sample
		data_size=SIZE(data_in)
		num_bins=SIZE(bins)-1
		ALLOCATE(histogram(num_bins))
		histogram=0

		DO loop_outer=1,data_size
			sample=data_in(loop_outer)
			DO loop_inner=1,num_bins
				x_0=bins(loop_inner)
				x_1=bins(loop_inner+1)
				IF (x_0 <= sample .AND. sample < x_1) THEN
					histogram(loop_inner)=histogram(loop_inner)+1
					EXIT
				END IF
			END DO
		END DO

	END FUNCTION histogram
	
END MODULE mod_utilities
