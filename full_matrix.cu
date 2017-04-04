#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <cuda.h>

char *cp_to_device(char *from, size_t size) {
	char *tmp;
	cudaMalloc((void**)&tmp, size);
	cudaMemcpy(tmp, from, size, cudaMemcpyHostToDevice);
	return tmp;
}

void cp_to_host(char *to, char*from, size_t size) {
	cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
  cudaFree(from);
}

__global__ void cc_loop1(const double * __restrict rho, const double * __restrict  Vf, const double * __restrict  V, double * __restrict rho_ave, int sizex, int sizey, int Nmats) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= sizex || j >= sizey) return;
	double ave = 0.0;
	for (int mat = 0; mat < Nmats; mat++) {
		// Optimisation:
		if (Vf[(i+sizex*j)*Nmats+mat] > 0.0)
			ave += rho[(i+sizex*j)*Nmats+mat]*Vf[(i+sizex*j)*Nmats+mat];
	}
	rho_ave[i+sizex*j] = ave/V[i+sizex*j];
}

__global__ void mc_loop1(const double * __restrict rho, const double * __restrict  Vf, const double * __restrict  V, double * __restrict rho_ave, int sizex, int sizey, int Nmats) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= sizex || j >= sizey) return;
	int ncells = sizex*sizey;

	for (int mat = 0; mat < Nmats; mat++) {
		if (Vf[ncells*mat + i+sizex*j] > 0.0)
			rho_ave[i+sizex*j] += rho[ncells*mat + i+sizex*j] * Vf[ncells*mat + i+sizex*j];
	}
	rho_ave[i+sizex*j] /= V[i+sizex*j];
}

__global__ void cc_loop2(const double * __restrict rho, const double * __restrict  Vf, const double * __restrict t, const double * __restrict n, double * __restrict p, int sizex, int sizey, int Nmats) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= sizex || j >= sizey) return;
	for (int mat = 0; mat < Nmats; mat++) {
		if (Vf[(i+sizex*j)*Nmats+mat] > 0.0) {
			double nm = n[mat];
			p[(i+sizex*j)*Nmats+mat] = (nm * rho[(i+sizex*j)*Nmats+mat] * t[(i+sizex*j)*Nmats+mat]) / Vf[(i+sizex*j)*Nmats+mat];
		}
		else {
			p[(i+sizex*j)*Nmats+mat] = 0.0;
		}
	}
}

__global__ void mc_loop2(const double * __restrict rho, const double * __restrict  Vf, const double * __restrict t, const double * __restrict n, double * __restrict p, int sizex, int sizey, int Nmats) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= sizex || j >= sizey) return;
	int ncells = sizex*sizey;

	for (int mat = 0; mat < Nmats; mat++) {
		double nm = n[mat];
		if (Vf[ncells*mat + i+sizex*j] > 0.0) {
			p[ncells*mat + i+sizex*j] = (nm * rho[ncells*mat + i+sizex*j] * t[ncells*mat + i+sizex*j]) / Vf[ncells*mat + i+sizex*j];
		}
		else {
			p[ncells*mat + i+sizex*j] = 0.0;
		}
	}
}

__global__ void cc_loop3(const double * __restrict rho, double *__restrict rho_mat_ave, const double * __restrict  Vf,
						const double * __restrict x, const double * __restrict y, int sizex, int sizey, int Nmats) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= sizex-1 || j >= sizey-1 || i < 1 || j < 1) return;

	// o: outer
	double xo = x[i+sizex*j];
	double yo = y[i+sizex*j];

	// There are at most 9 neighbours in 2D case.
	double dsqr[9];

	for (int nj = -1; nj <= 1; nj++) {
		for (int ni = -1; ni <= 1; ni++) {

			dsqr[(nj+1)*3 + (ni+1)] = 0.0;

			// i: inner
			double xi = x[(i+ni)+sizex*(j+nj)];
			double yi = y[(i+ni)+sizex*(j+nj)];

			dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
			dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
		}
	}

	for (int mat = 0; mat < Nmats; mat++) {
		if (Vf[(i+sizex*j)*Nmats+mat] > 0.0) {
			double rho_sum = 0.0;
			int Nn = 0;

			for (int nj = -1; nj <= 1; nj++) {
				for (int ni = -1; ni <= 1; ni++) {

					if (Vf[((i+ni)+sizex*(j+nj))*Nmats+mat] > 0.0) {
						rho_sum += rho[((i+ni)+sizex*(j+nj))*Nmats+mat] / dsqr[(nj+1)*3 + (ni+1)];
						Nn += 1;
					}
				}
			}
			rho_mat_ave[(i+sizex*j)*Nmats+mat] = rho_sum / Nn;
		}
		else {
			rho_mat_ave[(i+sizex*j)*Nmats+mat] = 0.0;
		}
	}
}

__global__ void mc_loop3(const double * __restrict rho, double * __restrict rho_mat_ave, const double * __restrict  Vf,
						const double * __restrict x, const double * __restrict y, int sizex, int sizey, int Nmats) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= sizex-1 || j >= sizey-1 || i < 1 || j < 1) return;
	int ncells = sizex*sizey;

	for (int mat = 0; mat < Nmats; mat++) {
		if (Vf[ncells*mat + i+sizex*j] > 0.0) {
					// o: outer
			double xo = x[i+sizex*j];
			double yo = y[i+sizex*j];

			double rho_sum = 0.0;
			int Nn = 0;

			for (int nj = -1; nj <= 1; nj++) {
				for (int ni = -1; ni <= 1; ni++) {
					if (Vf[ncells*mat + (i+ni)+sizex*(j+nj)] > 0.0) {
						double dsqr = 0.0;

						// i: inner
						double xi = x[(i+ni)+sizex*(j+nj)];
						double yi = y[(i+ni)+sizex*(j+nj)];

						dsqr += (xo - xi) * (xo - xi);
						dsqr += (yo - yi) * (yo - yi);

						rho_sum += rho[ncells*mat + i+sizex*j] / dsqr;
						Nn += 1;
					}
				}
			}

			rho_mat_ave[ncells*mat + i+sizex*j] = rho_sum / Nn;
		}
		else {
			rho_mat_ave[ncells*mat + i+sizex*j] = 0.0;
		}
	}
}

void full_matrix_cell_centric(int sizex, int sizey, int Nmats,
	double *rho, double *rho_mat_ave, double *p, double *Vf, double *t,
	double *V, double *x, double *y,
	double *n, double *rho_ave)
{
	double *d_rho = (double *)cp_to_device((char*)rho, sizex*sizey*Nmats*sizeof(double));
	double *d_rho_mat_ave = (double *)cp_to_device((char*)rho_mat_ave, sizex*sizey*Nmats*sizeof(double));
	double *d_p = (double *)cp_to_device((char*)p, sizex*sizey*Nmats*sizeof(double));
	double *d_t = (double *)cp_to_device((char*)t, sizex*sizey*Nmats*sizeof(double));
	double *d_Vf = (double *)cp_to_device((char*)Vf, sizex*sizey*Nmats*sizeof(double));
	double *d_V = (double *)cp_to_device((char*)V, sizex*sizey*sizeof(double));
	double *d_x = (double *)cp_to_device((char*)x, sizex*sizey*sizeof(double));
	double *d_y = (double *)cp_to_device((char*)y, sizex*sizey*sizeof(double));
	double *d_n = (double *)cp_to_device((char*)n, Nmats*sizeof(double));
	double *d_rho_ave = (double *)cp_to_device((char*)rho_ave, sizex*sizey*sizeof(double));

	int thx = 32;
	int thy = 4;
	dim3 threads(thx,thy,1);
	dim3 blocks((sizex-1)/thx+1, (sizey-1)/thy+1, 1);


	// Cell-centric algorithms
	// Computational loop 1 - average density in cell
  double t1 = omp_get_wtime();
  cc_loop1<<<blocks, threads>>>(d_rho, d_Vf, d_V, d_rho_ave, sizex, sizey, Nmats);
  cudaDeviceSynchronize();
  printf("Full matrix, cell centric, alg 1: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 2 - Pressure for each cell and each material
  t1 = omp_get_wtime();
  cc_loop2<<<blocks, threads>>>(d_rho, d_Vf, d_t, d_n, d_p, sizex, sizey, Nmats);
  cudaDeviceSynchronize();
  printf("Full matrix, cell centric, alg 2: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 3 - Average density of each material over neighborhood of each cell
  t1 = omp_get_wtime();
  cc_loop3<<<blocks, threads>>>(d_rho, d_rho_mat_ave,  d_Vf, d_x, d_y, sizex, sizey, Nmats);
  cudaDeviceSynchronize();
  printf("Full matrix, cell centric, alg 3: %g sec\n", omp_get_wtime()-t1);

  cp_to_host((char*)rho, (char*)d_rho, sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)rho_mat_ave, (char*)d_rho_mat_ave, sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)p,   (char*)d_p,   sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)t,   (char*)d_t,   sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)Vf,  (char*)d_Vf,  sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)V,   (char*)d_V,   sizex*sizey*sizeof(double));
  cp_to_host((char*)x,   (char*)d_x,   sizex*sizey*sizeof(double));
  cp_to_host((char*)y,   (char*)d_y,   sizex*sizey*sizeof(double));
  cp_to_host((char*)n,   (char*)d_n,   Nmats*sizeof(double));
  cp_to_host((char*)rho_ave, (char*)d_rho_ave, sizex*sizey*sizeof(double));
}

void full_matrix_material_centric(int sizex, int sizey, int Nmats,
	double *rho, double *rho_mat_ave, double *p, double *Vf, double *t,
	double *V, double *x, double *y,
	double *n, double *rho_ave)
{
	double *d_rho = (double *)cp_to_device((char*)rho, sizex*sizey*Nmats*sizeof(double));
	double *d_p = (double *)cp_to_device((char*)p, sizex*sizey*Nmats*sizeof(double));
	double *d_t = (double *)cp_to_device((char*)t, sizex*sizey*Nmats*sizeof(double));
	double *d_Vf = (double *)cp_to_device((char*)Vf, sizex*sizey*Nmats*sizeof(double));
	double *d_V = (double *)cp_to_device((char*)V, sizex*sizey*sizeof(double));
	double *d_x = (double *)cp_to_device((char*)x, sizex*sizey*sizeof(double));
	double *d_y = (double *)cp_to_device((char*)y, sizex*sizey*sizeof(double));
	double *d_n = (double *)cp_to_device((char*)n, Nmats*sizeof(double));
	double *d_rho_ave = (double *)cp_to_device((char*)rho_ave, sizex*sizey*sizeof(double));
	double *d_rho_mat_ave = (double *)cp_to_device((char*)rho_mat_ave, sizex*sizey*Nmats*sizeof(double));

	int thx = 32;
	int thy = 4;
	dim3 threads(thx,thy,1);
	dim3 blocks((sizex-1)/thx+1, (sizey-1)/thy+1, 1);



	// Material-centric algorithms
	// Computational loop 1 - average density in cell
  double t1 = omp_get_wtime();
  mc_loop1<<<blocks, threads>>>(d_rho, d_Vf, d_V, d_rho_ave, sizex, sizey, Nmats);
  cudaDeviceSynchronize();
  printf("Full matrix, material centric, alg 1: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 2 - Pressure for each cell and each material
  t1 = omp_get_wtime();
  mc_loop2<<<blocks, threads>>>(d_rho, d_Vf, d_t, d_n, d_p, sizex, sizey, Nmats);
  cudaDeviceSynchronize();
  printf("Full matrix, material centric, alg 2: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 3 - Average density of each material over neighborhood of each cell
  t1 = omp_get_wtime();
  mc_loop3<<<blocks, threads>>>(d_rho, d_rho_mat_ave, d_Vf, d_x, d_y, sizex, sizey, Nmats);
  cudaDeviceSynchronize();
  printf("Full matrix, material centric, alg 2: %g sec\n", omp_get_wtime()-t1);

  cp_to_host((char*)rho, (char*)d_rho, sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)p,   (char*)d_p,   sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)t,   (char*)d_t,   sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)Vf,  (char*)d_Vf,  sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)V,   (char*)d_V,   sizex*sizey*sizeof(double));
  cp_to_host((char*)x,   (char*)d_x,   sizex*sizey*sizeof(double));
  cp_to_host((char*)y,   (char*)d_y,   sizex*sizey*sizeof(double));
  cp_to_host((char*)n,   (char*)d_n,   Nmats*sizeof(double));
  cp_to_host((char*)rho_ave, (char*)d_rho_ave, sizex*sizey*sizeof(double));
  cp_to_host((char*)rho_mat_ave, (char*)d_rho_mat_ave, sizex*sizey*Nmats*sizeof(double));
}

bool full_matrix_check_results(int sizex, int sizey, int Nmats,
	double *rho_ave, double *rho_ave_mat, double *p, double *p_mat,
	double *rho, double *rho_mat, double *rho_mat_ave, double *rho_mat_ave_mat)
{
	int ncells = sizex * sizey;
	printf("Checking results of full matrix representation... ");

	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			if (abs(rho_ave[i+sizex*j] - rho_ave_mat[i+sizex*j]) > 0.0001) {
				printf("1. cell-centric and material-centric values are not equal! (%f, %f, %d, %d)\n",
					rho_ave[i+sizex*j], rho_ave_mat[i+sizex*j], i, j);
				return false;
			}

			for (int mat = 0; mat < Nmats; mat++) {
				if (abs(p[(i+sizex*j)*Nmats+mat] - p_mat[ncells*mat + i+sizex*j]) > 0.0001) {
					printf("2. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						p[(i+sizex*j)*Nmats+mat], p_mat[ncells*mat + i+sizex*j], i, j, mat);
					return false;
				}

				if (abs(rho_mat_ave[(i+sizex*j)*Nmats+mat] - rho_mat_ave_mat[ncells*mat + i+sizex*j]) > 0.0001) {
					printf("3. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						rho_mat_ave[(i+sizex*j)*Nmats+mat], rho_mat_ave_mat[ncells*mat + i+sizex*j], i, j, mat);
					return false;
				}
			}
		}
	}

	printf("All tests passed!\n");
	return true;
}
