MODULE mod_utilities
	USE mod_kind, ONLY: ki, kr
	USE, INTRINSIC :: ISO_C_BINDING
	USE omp_lib
	IMPLICIT NONE

	PRIVATE
	PUBLIC :: linspace, histogram, tile_indices

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
		REAL(KIND=kr), DIMENSION(:), INTENT(IN) :: data_in
		INTEGER(KIND=ki), DIMENSION(:), ALLOCATABLE :: histogram, histogram_l
		INTEGER(KIND=ki) :: loop_outer, loop_inner, data_size, num_bins, omp_indices(2)
		REAL(KIND=kr), DIMENSION(:), ALLOCATABLE :: data_l
		REAL(KIND=kr) :: x_0, x_1, sample
		data_size=SIZE(data_in)
		num_bins=SIZE(bins)-1
		ALLOCATE(histogram(num_bins))
		histogram=0
		!$OMP PARALLEL PRIVATE(histogram_l,data_l,sample,loop_outer,loop_inner,omp_indices) SHARED(bins,histogram,data_in)
		ALLOCATE(histogram_l(num_bins))
		histogram_l=0
		omp_indices=tile_indices(data_size)
		ALLOCATE(data_l(omp_indices(1):omp_indices(2)))
		data_l(omp_indices(1):omp_indices(2)) = data_in(omp_indices(1):omp_indices(2))
		DO loop_outer=omp_indices(1),omp_indices(2)
			sample=data_l(loop_outer)
			IF (sample<bins(1) .OR. bins(SIZE(bins))<sample) THEN
				WRITE(*,*) 'Sample outside binning area!'
				70  FORMAT ('bins_1=',f7.5,', bins_-1=',f7.5,', sample=',f7.5)
				WRITE(*,70) bins(1), bins(SIZE(bins)),sample
			END IF
			DO loop_inner=1,num_bins
				x_0=bins(loop_inner)
				x_1=bins(loop_inner+1)
				IF (x_0 <= sample .AND. sample < x_1) THEN
					histogram(loop_inner)=histogram(loop_inner)+1
					EXIT
				END IF
			END DO
		END DO
		DEALLOCATE(data_l)
		!$OMP CRITICAL
		histogram = histogram + histogram_l
		DEALLOCATE(histogram_l)
		!$OMP END CRITICAL
		!$OMP END PARALLEL
	END FUNCTION histogram

	FUNCTION tile_indices(nums)
		INTEGER(KIND=ki) :: ims, rest, im, m
		INTEGER(KIND=ki), INTENT(IN) :: nums
		INTEGER(KIND=ki), DIMENSION(2) :: tile_indices

		IF (OMP_IN_PARALLEL()) THEN
			ims = OMP_GET_NUM_THREADS()
			im = OMP_GET_THREAD_NUM()+1
			IF (ims>nums) THEN
				WRITE(*,*) 'Error: more images than data sets.'
				STOP
			END IF

			rest=MOD(nums,ims)
			m=(nums/ims)


			IF (im<=rest) THEN
			tile_indices(1)=im+m*(im-1)
			tile_indices(2)=(m+1)*im
			ELSE
			tile_indices(1)=(im-1)*m+rest+1
			tile_indices(2)=rest+m*im     
			END IF
		ELSE 
			WRITE(*,*) 'Warning: Called tile_indices outside a parallel region.'
			tile_indices= (/ 1_ki,nums /)
		END IF
	END FUNCTION tile_indices

END MODULE mod_utilities
