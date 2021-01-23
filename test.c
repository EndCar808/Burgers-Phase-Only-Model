#include <stdlib.h>
#include <stdio.h>
#include <math.h>


void amp_normalize(double* norm, double* amp, int num_osc, int k0) {

	// Initialize variables
	int k1;
	int N = num_osc - 1;

	// Compute the sum for each k
	for (int kk = 0; kk <= N; ++kk) {
		if (kk <= k0) {
			norm[kk] = 0.0;
		}
		else {
			for (int k_1 = 0; k_1 <= 2 * N; ++k_1) {
				// Adjust for the correct k1 value
				if (k_1 <= N) {     
					k1 = -N + k_1;
				}
				else  {
					k1 = k_1 - N;
				}
				
				// Compute the convolution
				if (k1 >= - N + kk) {
					norm[kk] +=  amp[abs(k1)] * amp[abs(kk - k1)];
				}
			}
		}
		
	}
}


int main(int argc, char** argv) {

	// Collocation points
	int N = atoi(argv[1]);

	int n  = N / 2 + 1;
	int k0 = 1;

	double a = 1.0;

	double* amp = (double*) malloc(sizeof(double) * n);
	double* amp_norm = (double*) malloc(sizeof(double) * n);

	for (int i = 0; i < n; ++i)	{
		if (i <= k0) {
			amp[i] = 0.0;
		}
		else {
			amp[i] = pow((double) i, -a);
		}
	}

	for (int i = 0; i < n; ++i)	{
		printf("a_k[%d]: %6.16lf\n", i, amp[i]);
	}
	printf("\n\n");


	amp_normalize(amp_norm, amp, n, k0);


	// for (int i = 0; i < n; ++i)	{
	// 	printf("amp_n[%d]: %6.16lf\ta_k[%d]: %6.16lf\n", i, amp_norm[i], i, amp[i]);
	// }
	// printf("\n\n");

	free(amp);
	free(amp_norm);
	

	return 0;
}