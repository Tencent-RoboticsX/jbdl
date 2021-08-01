#include<iostream>
#include<vector>
#include <stdlib.h>
#include "osqp.h"

using namespace std;


void gpu_lcp(const c_float *H, const c_float *q, const c_float *F, const c_float *l,
	const c_float *u, const c_int n, const c_int m, c_float *primal, c_float *dual) {

	c_int *P_p;
	P_p = (c_int*)malloc((n + 1) * sizeof(c_int));

	P_p[0] = 0;

	c_int *row_nozeronum_P;
	row_nozeronum_P = (c_int*)malloc(n * sizeof(c_int));

	c_float * datatmp_P;
	datatmp_P = (c_float*)malloc((n*n) * sizeof(c_float));

	c_int * indicestmp_P;
	indicestmp_P = (c_int*)malloc((n*n) * sizeof(c_int));

	
	c_int NNZ_P = 0;

	for (c_int i = 0; i < n; i++) {
		c_int row_nozeros_P = 0;
		for (c_int j = 0; j < n; j++) {
			if (abs(H[n*j + i]) > 1e-6) {
				datatmp_P[NNZ_P] = H[n*j + i];
				indicestmp_P[NNZ_P] = j;
				NNZ_P++;
				row_nozeros_P++;
			}
		}

		row_nozeronum_P[i] = row_nozeros_P;

		if (i > 0) {
			P_p[i] = P_p[i - 1] + row_nozeronum_P[i - 1];
		}
	}
	P_p[n] = NNZ_P;


	c_float *P_x;
	c_int *P_i;

	P_x = (c_float*)malloc(NNZ_P * sizeof(c_float));
	P_i = (c_int*)malloc(NNZ_P * sizeof(c_int));


	for (c_int k = 0; k < NNZ_P; k++) {
		P_x[k] = datatmp_P[k];
		P_i[k] = indicestmp_P[k];

		printf("P_i[%d]:%d\n", k, P_i[k]);
		printf("P_x[%d]:%f\n", k, P_x[k]);
	}


	c_int *A_p;
	A_p = (c_int*)malloc((n + 1) * sizeof(c_int));
	A_p[0] = 0;

	c_int *row_nozeronum_A;
	row_nozeronum_A = (c_int*)malloc(n * sizeof(c_int));

	c_float * datatmp_A;
	datatmp_A = (c_float*)malloc((n*m) * sizeof(c_float));

	c_int * indicestmp_A;
	indicestmp_A = (c_int*)malloc((n*m) * sizeof(c_int));

	c_int NNZ_A = 0;

	for (c_int i = 0; i < n; i++) {
		c_int row_nozeros_A = 0;
		for (c_int j = 0; j < m; j++) {
			if (abs(F[n*j + i]) > 1e-6) {
				datatmp_A[NNZ_A] = F[n*j + i];
				indicestmp_A[NNZ_A] = j;
				
				NNZ_A++;
				row_nozeros_A++;
			}
		}

		row_nozeronum_A[i] = row_nozeros_A;

		if (i > 0) {
			A_p[i] = A_p[i - 1] + row_nozeronum_A[i - 1];
		}
	}
	A_p[n] = NNZ_A;


	c_float *A_x;
	c_int *A_i;
	A_x = (c_float*)malloc(NNZ_A * sizeof(c_float));
	A_i = (c_int*)malloc(NNZ_A * sizeof(c_int));

	for (c_int k = 0; k < NNZ_A; k++) {
		A_x[k] = datatmp_A[k];
		A_i[k] = indicestmp_A[k];

		printf("A_i[%d]:%d\n", k, A_i[k]);
		printf("A_x[%d]:%f\n", k, A_x[k]);
	}

	for (int k = 0; k < (n + 1); k++) {
		printf("A_p[%d]:%d\n", k, A_p[k]);
		printf("P_p[%d]:%d\n", k, P_p[k]);
	}

	free(row_nozeronum_P);
	free(datatmp_P);
	free(indicestmp_P);

	free(row_nozeronum_A);
	free(datatmp_A);
	free(indicestmp_A);

	c_int exitflag = 0;

	OSQPSolver   *solver;
	OSQPSettings *settings;
	csc *P_csc = (csc *)malloc(sizeof(csc));
	csc *A_csc = (csc *)malloc(sizeof(csc));

	/* Populate matrices */
	csc_set_data(A_csc, m, n, NNZ_A, const_cast<c_float *>(A_x), const_cast<c_int *>(A_i), const_cast<c_int *>(A_p));
	csc_set_data(P_csc, n, n, NNZ_P, const_cast<c_float *>(P_x), const_cast<c_int *>(P_i), const_cast<c_int *>(P_p));

	settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
	if (settings) osqp_set_default_settings(settings);
	settings->polish = 1;

	/* Setup workspace */
	exitflag = osqp_setup(&solver, P_csc, q, A_csc, l, u, m, n, settings);

	/* Solve Problem */
	osqp_solve(solver);


	for (int i = 0; i < n; i++) {
		primal[i] = solver->solution->x[i];
	}

	for (int j = 0; j < m; j++) {
		dual[j] = solver->solution->y[j];
	}


}

int main() {
	c_float H[4] = { 4.0, 1.0, 0.0, 2.0 };
	c_float q[2] = { 1.0, 1.0, };
	c_float F[6] = { 1.0, 1.0, 1.0, 0.0, 0.0, 1.0 };
	c_float l[3] = { 1.0, 0.0, 0.0 };
	c_float u[3] = { 1.0, 0.7, 0.7 };
	c_int n = 2;
	c_int m = 3;

	c_float primal[2];
	c_float dual[3];

	gpu_lcp(H, q, F, l, u, n, m, primal, dual);

	return 0;
}
