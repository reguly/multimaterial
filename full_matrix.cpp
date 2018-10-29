#include <math.h>
#include <stdio.h>
#include <omp.h>



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

void full_matrix_cell_centric(full_data cc)
{
	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;

#if defined(ACC)
#pragma acc data copy(cc.rho[0:sizex*sizey*Nmats], cc.p[0:sizex*sizey*Nmats], cc.t[0:sizex*sizey*Nmats], cc.Vf[0:sizex*sizey*Nmats]) \
  copy(cc.V[0:sizex*sizey],cc.x[0:sizex*sizey],cc.y[0:sizex*sizey],cc.n[0:Nmats],cc.rho_ave[0:sizex*sizey]) \
  copy(rho_mat_ave[0:sizex*sizey*Nmats])
#endif
{
	// Cell-centric algorithms
	// Computational loop 1 - average density in cell
  double t1 = omp_get_wtime();
  #if defined(OMP)
  #pragma omp parallel for collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int j = 0; j < sizey; j++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
		for (int i = 0; i < sizex; i++){
			double ave = 0.0;
//#pragma omp simd reduction(+:ave)
			for (int mat = 0; mat < Nmats; mat++) {
				// Optimisation:
				if (cc.Vf[(i+sizex*j)*Nmats+mat] > 0.0)
					ave += cc.rho[(i+sizex*j)*Nmats+mat]*cc.Vf[(i+sizex*j)*Nmats+mat];
			}
			cc.rho_ave[i+sizex*j] = ave/cc.V[i+sizex*j];
		}
	}
  printf("Full matrix, cell centric, alg 1: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 2 - Pressure for each cell and each material
  t1 = omp_get_wtime();
  #if defined(OMP)
  #pragma omp parallel for collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int j = 0; j < sizey; j++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
		for (int i = 0; i < sizex; i++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
//#pragma omp simd
			for (int mat = 0; mat < Nmats; mat++) {
				if (cc.Vf[(i+sizex*j)*Nmats+mat] > 0.0) {
					double nm = cc.n[mat];
					cc.p[(i+sizex*j)*Nmats+mat] = (nm * cc.rho[(i+sizex*j)*Nmats+mat] * cc.t[(i+sizex*j)*Nmats+mat]) / cc.Vf[(i+sizex*j)*Nmats+mat];
				}
				else {
					cc.p[(i+sizex*j)*Nmats+mat] = 0.0;
				}
			}
		}
	}
  printf("Full matrix, cell centric, alg 2: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 3 - Average density of each material over neighborhood of each cell
  t1 = omp_get_wtime();
  #if defined(OMP)
  #pragma omp parallel for collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int j = 1; j < sizey-1; j++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
		for (int i = 1; i < sizex-1; i++) {
			// o: outer
			double xo = cc.x[i+sizex*j];
			double yo = cc.y[i+sizex*j];

			// There are at most 9 neighbours in 2D case.
			double dsqr[9];

			for (int nj = -1; nj <= 1; nj++) {
				for (int ni = -1; ni <= 1; ni++) {

					dsqr[(nj+1)*3 + (ni+1)] = 0.0;

					// i: inner
					double xi = cc.x[(i+ni)+sizex*(j+nj)];
					double yi = cc.y[(i+ni)+sizex*(j+nj)];

					dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
					dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
				}
			}
//#pragma omp simd
			for (int mat = 0; mat < Nmats; mat++) {
				if (cc.Vf[(i+sizex*j)*Nmats+mat] > 0.0) {
					double rho_sum = 0.0;
					int Nn = 0;

					for (int nj = -1; nj <= 1; nj++) {
						if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
							continue;

						for (int ni = -1; ni <= 1; ni++) {
							if ((i + ni < 0) || (i + ni >= sizex)) // TODO: better way?
								continue;

							if (cc.Vf[((i+ni)+sizex*(j+nj))*Nmats+mat] > 0.0) {
								rho_sum += cc.rho[((i+ni)+sizex*(j+nj))*Nmats+mat] / dsqr[(nj+1)*3 + (ni+1)];
								Nn += 1;
							}
						}
					}
					cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] = rho_sum / Nn;
				}
				else {
					cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] = 0.0;
				}
			}
		}
	}
  printf("Full matrix, cell centric, alg 3: %g sec\n", omp_get_wtime()-t1);
}
}

void full_matrix_material_centric(full_data cc, full_data mc)
{
	int sizex = mc.sizex;
	int sizey = mc.sizey;
	int Nmats = mc.Nmats;
	int ncells = sizex * sizey;
#if defined(ACC)
#pragma acc data copy(mc.rho[0:sizex*sizey*Nmats], mc.p[0:sizex*sizey*Nmats], mc.t[0:sizex*sizey*Nmats], mc.Vf[0:sizex*sizey*Nmats]) \
  copy(mc.V[0:sizex*sizey],mc.x[0:sizex*sizey],mc.y[0:sizex*sizey],mc.n[0:Nmats],mc.rho_ave[0:sizex*sizey]) \
  copy(mc.rho_mat_ave[0:sizex*sizey*Nmats])
#endif
  {
	// Material-centric algorithms
	// Computational loop 1 - average density in cell
  double t1 = omp_get_wtime();
  #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int j = 0; j < sizey; j++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
//#pragma omp simd
		for (int i = 0; i < sizex; i++) {
			mc.rho_ave[i+sizex*j] = 0.0;
		}
	}

	for (int mat = 0; mat < Nmats; mat++) {
    #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
		for (int j = 0; j < sizey; j++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
//#pragma omp simd
			for (int i = 0; i < sizex; i++) {
				// Optimisation:
				if (mc.Vf[ncells*mat + i+sizex*j] > 0.0)
					mc.rho_ave[i+sizex*j] += mc.rho[ncells*mat + i+sizex*j] * mc.Vf[ncells*mat + i+sizex*j];
			}
		}
	}

  #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int j = 0; j < sizey; j++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
//#pragma omp simd
		for (int i = 0; i < sizex; i++) {
			mc.rho_ave[i+sizex*j] /= mc.V[i+sizex*j];
		}
	}
  printf("Full matrix, material centric, alg 1: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 2 - Pressure for each cell and each material
  t1 = omp_get_wtime();
  #if defined(OMP)
  #pragma omp parallel for collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int mat = 0; mat < Nmats; mat++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
		for (int j = 0; j < sizey; j++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
//#pragma omp simd
			for (int i = 0; i < sizex; i++) {
        double nm = mc.n[mat];
				if (mc.Vf[ncells*mat + i+sizex*j] > 0.0) {
					mc.p[ncells*mat + i+sizex*j] = (nm * mc.rho[ncells*mat + i+sizex*j] * mc.t[ncells*mat + i+sizex*j]) / mc.Vf[ncells*mat + i+sizex*j];
				}
				else {
					mc.p[ncells*mat + i+sizex*j] = 0.0;
				}
			}
		}
	}
  printf("Full matrix, material centric, alg 2: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 3 - Average density of each material over neighborhood of each cell
  t1 = omp_get_wtime();
  #if defined(OMP)
  #pragma omp parallel for collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int mat = 0; mat < Nmats; mat++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
		for (int j = 1; j < sizey-1; j++) {
  #if defined(ACC)
  #pragma acc loop independent
  #endif
//#pragma omp simd
			for (int i = 1; i < sizex-1; i++) {
				if (mc.Vf[ncells*mat + i+sizex*j] > 0.0) {
					// o: outer
					double xo = mc.x[i+sizex*j];
					double yo = mc.y[i+sizex*j];

					double rho_sum = 0.0;
					int Nn = 0;

					for (int nj = -1; nj <= 1; nj++) {
						if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
							continue;

						for (int ni = -1; ni <= 1; ni++) {
							if ((i + ni < 0) || (i + ni >= sizex)) // TODO: better way?
								continue;

							if (mc.Vf[ncells*mat + (i+ni)+sizex*(j+nj)] > 0.0) {
								double dsqr = 0.0;

								// i: inner
								double xi = mc.x[(i+ni)+sizex*(j+nj)];
								double yi = mc.y[(i+ni)+sizex*(j+nj)];

								dsqr += (xo - xi) * (xo - xi);
								dsqr += (yo - yi) * (yo - yi);

								rho_sum += mc.rho[ncells*mat + i+sizex*j] / dsqr;
								Nn += 1;
							}
						}
					}

					mc.rho_mat_ave[ncells*mat + i+sizex*j] = rho_sum / Nn;
				}
				else {
					mc.rho_mat_ave[ncells*mat + i+sizex*j] = 0.0;
				}
			}
		}
	}
  printf("Full matrix, material centric, alg 2: %g sec\n", omp_get_wtime()-t1);
  }
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
