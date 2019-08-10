#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <cuda.h>


struct full_data
{
	int sizex;
	int sizey;
	int Nmats;
	double * __restrict__ rho;
	double * __restrict__ rho_mat_ave;
	double * __restrict__ p;
	double * __restrict__ Vf;
	double * __restrict__ t;
	double * __restrict__ V;
	double * __restrict__ x;
	double * __restrict__ y;
	double * __restrict__ n;
	double * __restrict__ rho_ave;
};



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

void full_matrix_cell_centric(full_data cc)
{
	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;
	double *d_rho = (double *)cp_to_device((char*)cc.rho, sizex*sizey*Nmats*sizeof(double));
	double *d_rho_mat_ave = (double *)cp_to_device((char*)cc.rho_mat_ave, sizex*sizey*Nmats*sizeof(double));
	double *d_p = (double *)cp_to_device((char*)cc.p, sizex*sizey*Nmats*sizeof(double));
	double *d_t = (double *)cp_to_device((char*)cc.t, sizex*sizey*Nmats*sizeof(double));
	double *d_Vf = (double *)cp_to_device((char*)cc.Vf, sizex*sizey*Nmats*sizeof(double));
	double *d_V = (double *)cp_to_device((char*)cc.V, sizex*sizey*sizeof(double));
	double *d_x = (double *)cp_to_device((char*)cc.x, sizex*sizey*sizeof(double));
	double *d_y = (double *)cp_to_device((char*)cc.y, sizex*sizey*sizeof(double));
	double *d_n = (double *)cp_to_device((char*)cc.n, Nmats*sizeof(double));
	double *d_rho_ave = (double *)cp_to_device((char*)cc.rho_ave, sizex*sizey*sizeof(double));

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

  cp_to_host((char*)cc.rho, (char*)d_rho, sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)cc.rho_mat_ave, (char*)d_rho_mat_ave, sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)cc.p,   (char*)d_p,   sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)cc.t,   (char*)d_t,   sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)cc.Vf,  (char*)d_Vf,  sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)cc.V,   (char*)d_V,   sizex*sizey*sizeof(double));
  cp_to_host((char*)cc.x,   (char*)d_x,   sizex*sizey*sizeof(double));
  cp_to_host((char*)cc.y,   (char*)d_y,   sizex*sizey*sizeof(double));
  cp_to_host((char*)cc.n,   (char*)d_n,   Nmats*sizeof(double));
  cp_to_host((char*)cc.rho_ave, (char*)d_rho_ave, sizex*sizey*sizeof(double));
}

void full_matrix_material_centric(full_data cc, full_data mc)
{
	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;
	double *d_rho = (double *)cp_to_device((char*)mc.rho, sizex*sizey*Nmats*sizeof(double));
	double *d_p = (double *)cp_to_device((char*)mc.p, sizex*sizey*Nmats*sizeof(double));
	double *d_t = (double *)cp_to_device((char*)mc.t, sizex*sizey*Nmats*sizeof(double));
	double *d_Vf = (double *)cp_to_device((char*)mc.Vf, sizex*sizey*Nmats*sizeof(double));
	double *d_V = (double *)cp_to_device((char*)mc.V, sizex*sizey*sizeof(double));
	double *d_x = (double *)cp_to_device((char*)mc.x, sizex*sizey*sizeof(double));
	double *d_y = (double *)cp_to_device((char*)mc.y, sizex*sizey*sizeof(double));
	double *d_n = (double *)cp_to_device((char*)mc.n, Nmats*sizeof(double));
	double *d_rho_ave = (double *)cp_to_device((char*)mc.rho_ave, sizex*sizey*sizeof(double));
	double *d_rho_mat_ave = (double *)cp_to_device((char*)mc.rho_mat_ave, sizex*sizey*Nmats*sizeof(double));

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

  cp_to_host((char*)mc.rho, (char*)d_rho, sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)mc.p,   (char*)d_p,   sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)mc.t,   (char*)d_t,   sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)mc.Vf,  (char*)d_Vf,  sizex*sizey*Nmats*sizeof(double));
  cp_to_host((char*)mc.V,   (char*)d_V,   sizex*sizey*sizeof(double));
  cp_to_host((char*)mc.x,   (char*)d_x,   sizex*sizey*sizeof(double));
  cp_to_host((char*)mc.y,   (char*)d_y,   sizex*sizey*sizeof(double));
  cp_to_host((char*)mc.n,   (char*)d_n,   Nmats*sizeof(double));
  cp_to_host((char*)mc.rho_ave, (char*)d_rho_ave, sizex*sizey*sizeof(double));
  cp_to_host((char*)mc.rho_mat_ave, (char*)d_rho_mat_ave, sizex*sizey*Nmats*sizeof(double));
}

bool full_matrix_check_results(full_data cc, full_data mc)
{
	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;
	int ncells = sizex * sizey;
	printf("Checking results of full matrix representation... ");

	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			if (fabs(cc.rho_ave[i+sizex*j] - mc.rho_ave[i+sizex*j]) > 0.0001) {
				printf("1. cell-centric and material-centric values are not equal! (%f, %f, %d, %d)\n",
					cc.rho_ave[i+sizex*j], mc.rho_ave[i+sizex*j], i, j);
				return false;
			}

			for (int mat = 0; mat < Nmats; mat++) {
				if (fabs(cc.p[(i+sizex*j)*Nmats+mat] - mc.p[ncells*mat + i+sizex*j]) > 0.0001) {
					printf("2. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						cc.p[(i+sizex*j)*Nmats+mat], mc.p[ncells*mat + i+sizex*j], i, j, mat);
					return false;
				}

				if (fabs(cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] - mc.rho_mat_ave[ncells*mat + i+sizex*j]) > 0.0001) {
					printf("3. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						cc.rho_mat_ave[(i+sizex*j)*Nmats+mat], mc.rho_mat_ave[ncells*mat + i+sizex*j], i, j, mat);
					return false;
				}
			}
		}
	}

	printf("All tests passed!\n");
	return true;
}
