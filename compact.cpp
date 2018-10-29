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

struct compact_data
{
	int sizex;
	int sizey;
	int Nmats;
	double * __restrict__ rho_compact;
	double * __restrict__ rho_compact_list;
	double * __restrict__ rho_mat_ave_compact;
	double * __restrict__ rho_mat_ave_compact_list;
	double * __restrict__ p_compact;
	double * __restrict__ p_compact_list;
	double * __restrict__ Vf_compact_list;
	double * __restrict__ t_compact;
	double * __restrict__ t_compact_list;
	double * __restrict__ V;
	double * __restrict__ x;
	double * __restrict__ y;
	double * __restrict__ n;
	double * __restrict__ rho_ave_compact;
	int * __restrict__ imaterial;
	int * __restrict__ matids;
	int * __restrict__ nextfrac;
	int * __restrict__ mmc_index;
	int * __restrict__ mmc_i;
	int * __restrict__ mmc_j;
	int mm_len;
	int mmc_cells;
};

void compact_cell_centric(full_data cc, compact_data ccc)
{
	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;
	int mmc_cells = ccc.mmc_cells;
  int mm_len = ccc.mm_len;

  #if defined(ACC)
  #pragma acc data copy(ccc.imaterial[0:sizex*sizey],ccc.matids[0:mm_len], ccc.nextfrac[0:mm_len], ccc.x[0:sizex*sizey], ccc.y[0:sizex*sizey],ccc.n[Nmats], ccc.rho_compact[0:sizex*sizey], ccc.rho_compact_list[0:mm_len], ccc.rho_ave_compact[0:sizex*sizey], ccc.p_compact[0:sizex*sizey], ccc.p_compact_list[0:mm_len], ccc.t_compact[0:sizex*sizey], ccc.t_compact_list[0:mm_len], ccc.V[0:sizex*sizey], ccc.Vf_compact_list[0:mm_len], ccc.mmc_index[0:mmc_cells+1], ccc.mmc_i[0:mmc_cells], ccc.mmc_j[0:mmc_cells], ccc.rho_mat_ave_compact[0:sizex*sizey], ccc.rho_mat_ave_compact_list[0:mm_len])
  #endif
  { 
	// Cell-centric algorithms
	// Computational loop 1 - average density in cell
  double t1 = omp_get_wtime();
  #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int j = 0; j < sizey; j++) {
  #if defined(OMP)
  #pragma omp simd
  #elif defined(ACC)
  #pragma acc loop independent
  #endif
		for (int i = 0; i < sizex; i++) {

#ifdef FUSED
			double ave = 0.0;
			int ix = ccc.imaterial[i+sizex*j];
			if (ix <= 0) {
				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
#ifdef LINKED
#pragma novector
				for (ix = -ix; ix >= 0; ix = ccc.nextfrac[ix]) {
					ave += ccc.rho_compact_list[ix] * ccc.Vf_compact_list[ix];
				}
#else
				for (int idx = ccc.mmc_index[-ix]; idx < ccc.mmc_index[-ix+1]; idx++) {
					ave += ccc.rho_compact_list[idx] * ccc.Vf_compact_list[idx];	
				}
#endif
				ccc.rho_ave_compact[i+sizex*j] = ave/ccc.V[i+sizex*j];
			}
			else {
#endif
				// We use a distinct output array for averages.
				// In case of a pure cell, the average density equals to the total.
				ccc.rho_ave_compact[i+sizex*j] = ccc.rho_compact[i+sizex*j] / ccc.V[i+sizex*j];
#ifdef FUSED
			}
#endif
		}
	}
#ifndef FUSED
  #if defined(OMP)
  #pragma omp parallel for simd
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
  for (int c = 0; c < ccc.mmc_cells; c++) {
    double ave = 0.0;
    for (int m = ccc.mmc_index[c]; m < ccc.mmc_index[c+1]; m++) {
      ave +=  ccc.rho_compact_list[m] * ccc.Vf_compact_list[m];
    }
    ccc.rho_ave_compact[ccc.mmc_i[c]+sizex*ccc.mmc_j[c]] = ave/ccc.V[ccc.mmc_i[c]+sizex*ccc.mmc_j[c]];
  }
#endif
  printf("Compact matrix, cell centric, alg 1: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 2 - Pressure for each cell and each material
  t1 = omp_get_wtime();
  
  #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int j = 0; j < sizey; j++) {
  #if defined(OMP)
  #pragma omp simd
  #elif defined(ACC)
  #pragma acc loop independent
  #endif
		for (int i = 0; i < sizex; i++) {
			int ix = ccc.imaterial[i+sizex*j];


			if (ix <= 0) {
#ifdef FUSED
				// NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
				// The solution below is what I believe to good.

				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = ccc.nextfrac[ix]) {
					double nm = ccc.n[ccc.matids[ix]];
					ccc.p_compact_list[ix] = (nm * ccc.rho_compact_list[ix] * ccc.t_compact_list[ix]) / ccc.Vf_compact_list[ix];
				}
#else
				for (int idx = ccc.mmc_index[-ix]; idx < ccc.mmc_index[-ix+1]; idx++) {
					double nm = ccc.n[ccc.matids[idx]];
					ccc.p_compact_list[idx] = (nm * ccc.rho_compact_list[idx] * ccc.t_compact_list[idx]) / ccc.Vf_compact_list[idx];
				}
#endif
#endif
			}
			else {
				// NOTE: HACK: we index materials from zero, but zero can be a list index
				int mat = ix - 1;
				// NOTE: There is no division by Vf here, because the fractional volume is 1.0 in the pure cell case.
				ccc.p_compact[i+sizex*j] = ccc.n[mat] * ccc.rho_compact[i+sizex*j] * ccc.t_compact[i+sizex*j];;
			}
		}
	}
printf("mm_len: %d, mmc_cells %d, mmc_index[mmc_cells] %d\n", mm_len, mmc_cells, ccc.mmc_index[mmc_cells]); 
#ifndef FUSED
  #if defined(OMP)
  #pragma omp parallel for simd
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
  for (int idx = 0; idx < ccc.mmc_index[mmc_cells]; idx++) {
    double nm = ccc.n[ccc.matids[idx]];
    ccc.p_compact_list[idx] = (nm * ccc.rho_compact_list[idx] * ccc.t_compact_list[idx]) / ccc.Vf_compact_list[idx];
  }
#endif

  printf("Compact matrix, cell centric, alg 2: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 3 - Average density of each material over neighborhood of each cell
  t1 = omp_get_wtime();
  #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
	for (int j = 1; j < sizey-1; j++) {
  #if defined(OMP)
  #pragma omp simd
  #elif defined(ACC)
  #pragma acc loop independent
  #endif
		for (int i = 1; i < sizex-1; i++) {
			// o: outer
			double xo = ccc.x[i+sizex*j];
			double yo = ccc.y[i+sizex*j];

			// There are at most 9 neighbours in 2D case.
			double dsqr[9];

			// for all neighbours
			for (int nj = -1; nj <= 1; nj++) {
				for (int ni = -1; ni <= 1; ni++) {

					dsqr[(nj+1)*3 + (ni+1)] = 0.0;

					// i: inner
					double xi = ccc.x[(i+ni)+sizex*(j+nj)];
					double yi = ccc.y[(i+ni)+sizex*(j+nj)];

					dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
					dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
				}
			}

			int ix = ccc.imaterial[i+sizex*j];

			if (ix <= 0) {
				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
				#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = ccc.nextfrac[ix]) {
				#else
				for (int ix = ccc.mmc_index[-ccc.imaterial[i+sizex*j]]; ix < ccc.mmc_index[-ccc.imaterial[i+sizex*j]+1]; ix++) {
				#endif

					int mat = ccc.matids[ix];
					double rho_sum = 0.0;
					int Nn = 0;

					// for all neighbours
					for (int nj = -1; nj <= 1; nj++) {
						for (int ni = -1; ni <= 1; ni++) {
							int ci = i+ni, cj = j+nj;
							int jx = ccc.imaterial[ci+sizex*cj];

							if (jx <= 0) {
								// condition is 'jx >= 0', this is the equivalent of
								// 'until jx < 0' from the paper
								#ifdef LINKED
								for (jx = -jx; jx >= 0; jx = ccc.nextfrac[jx]) {
								#else
								for (int jx = ccc.mmc_index[-ccc.imaterial[ci+sizex*cj]]; jx < ccc.mmc_index[-ccc.imaterial[ci+sizex*cj]+1]; jx++) {
								#endif
									if (ccc.matids[jx] == mat) {
										rho_sum += ccc.rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
										Nn += 1;

										// The loop has an extra condition: "and not found".
										// This makes sense, if the material is found, there won't be any more of the same.
										break;
									}
								}
							}
							else {
								// NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
								// In contrast, Algorithm 10 loads matids[jx] which I think is wrong.

								// NOTE: HACK: we index materials from zero, but zero can be a list index
								int mat_neighbour = jx - 1;
								if (mat == mat_neighbour) {
									rho_sum += ccc.rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
									Nn += 1;
								}
							} // end if (jx <= 0)
						} // end for (int ni)
					} // end for (int nj)

					ccc.rho_mat_ave_compact_list[ix] = rho_sum / Nn;
				} // end for (ix = -ix)
			} // end if (ix <= 0)
			else {
				// NOTE: In this case, the cell is a pure cell, its material index is in ix.
				// In contrast, Algorithm 10 loads matids[ix] which I think is wrong.

				// NOTE: HACK: we index materials from zero, but zero can be a list index
				int mat = ix - 1;

				double rho_sum = 0.0;
				int Nn = 0;

				// for all neighbours
				for (int nj = -1; nj <= 1; nj++) {
					if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
						continue;

					for (int ni = -1; ni <= 1; ni++) {
						if ((i + ni < 0) || (i + ni >= sizex)) // TODO: better way?
							continue;

						int ci = i+ni, cj = j+nj;
						int jx = ccc.imaterial[ci+sizex*cj];

						if (jx <= 0) {
							// condition is 'jx >= 0', this is the equivalent of
							// 'until jx < 0' from the paper
							#ifdef LINKED
							for (jx = -jx; jx >= 0; jx = ccc.nextfrac[jx]) {
							#else
							for (int jx = ccc.mmc_index[-ccc.imaterial[ci+sizex*cj]]; jx < ccc.mmc_index[-ccc.imaterial[ci+sizex*cj]+1]; jx++) {
							#endif
								if (ccc.matids[jx] == mat) {
									rho_sum += ccc.rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
									Nn += 1;

									// The loop has an extra condition: "and not found".
									// This makes sense, if the material is found, there won't be any more of the same.
									break;
								}
							}
						}
						else {
							// NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
							// In contrast, Algorithm 10 loads matids[jx] which I think is wrong.

							// NOTE: HACK: we index materials from zero, but zero can be a list index
							int mat_neighbour = jx - 1;
							if (mat == mat_neighbour) {
								rho_sum += ccc.rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
								Nn += 1;
							}
						} // end if (jx <= 0)
					} // end for (int ni)
				} // end for (int nj)

				ccc.rho_mat_ave_compact[i+sizex*j] = rho_sum / Nn;
			} // end else
		}
	}
  printf("Compact matrix, cell centric, alg 3: %g sec\n", omp_get_wtime()-t1);
  }
}

bool compact_check_results(full_data cc, compact_data ccc) 
{

	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;
	int mmc_cells = ccc.mmc_cells;

	printf("Checking results of compact representation... ");

	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			if (fabs(cc.rho_ave[i+sizex*j] - ccc.rho_ave_compact[i+sizex*j]) > 0.0001) {
				printf("1. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d)\n",
					cc.rho_ave[i+sizex*j], ccc.rho_ave_compact[i+sizex*j], i, j);
				return false;
			}
			int ix = ccc.imaterial[i+sizex*j];
			if (ix <= 0) {
#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = ccc.nextfrac[ix]) {
#else
        for (int ix = ccc.mmc_index[-ccc.imaterial[i+sizex*j]]; ix < ccc.mmc_index[-ccc.imaterial[i+sizex*j]+1]; ix++) {
#endif
					int mat = ccc.matids[ix];
					if (fabs(cc.p[(i+sizex*j)*Nmats+mat] - ccc.p_compact_list[ix]) > 0.0001) {
						printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
							cc.p[(i+sizex*j)*Nmats+mat], ccc.p_compact_list[ix], i, j, mat);
						return false;
					}

					if (fabs(cc.rho[(i+sizex*j)*Nmats+mat] - ccc.rho_compact_list[ix]) > 0.0001) {
						printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
							cc.rho[(i+sizex*j)*Nmats+mat], ccc.rho_compact_list[ix], i, j, mat);
						return false;
					}
				}
			}
			else {
				// NOTE: HACK: we index materials from zero, but zero can be a list index
				int mat = ix - 1;
				if (fabs(cc.p[(i+sizex*j)*Nmats+mat] - ccc.p_compact[i+sizex*j]) > 0.0001) {
					printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						cc.p[(i+sizex*j)*Nmats+mat], ccc.p_compact[i+sizex*j], i, j, mat);
					return false;
				}

				if (fabs(cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] - ccc.rho_mat_ave_compact[i+sizex*j]) > 0.0001) {
					printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						cc.rho_mat_ave[(i+sizex*j)*Nmats+mat], ccc.rho_mat_ave_compact[i+sizex*j], i, j, mat);
					return false;
				}
			}
		}
	}

	printf("All tests passed!\n");
	return true;
}
