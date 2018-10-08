#include <math.h>
#include <stdio.h>
#include <omp.h>

void compact_cell_centric(int sizex, int sizey, int Nmats,
	int *imaterial, int *matids, int *nextfrac,
	double *x, double *y, double *n,
	double *rho_compact, double *rho_compact_list, 
	double *rho_mat_ave_compact, double *rho_mat_ave_compact_list, 
  double *rho_ave_compact,
	double *p_compact, double *p_compact_list,
	double *t_compact, double *t_compact_list,
	double *V, double *Vf_compact_list, int mm_len, int mmc_cells, int *mmc_index, int *mmc_i, int *mmc_j)
{
  #if defined(ACC)
  #pragma acc data copy(imaterial[0:sizex*sizey],matids[0:mm_len], nextfrac[0:mm_len], x[0:sizex*sizey], y[0:sizex*sizey],n[Nmats], rho_compact[0:sizex*sizey], rho_compact_list[0:mm_len], rho_ave_compact[0:sizex*sizey], p_compact[0:sizex*sizey], p_compact_list[0:mm_len], t_compact[0:sizex*sizey], t_compact_list[0:mm_len], V[0:sizex*sizey], Vf_compact_list[0:mm_len], mmc_index[0:mmc_cells+1], mmc_i[0:mmc_cells], mmc_j[0:mmc_cells],rho_mat_ave_compact[0:sizex*sizey], rho_mat_ave_compact_list[0:mm_len])
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
		for (int i = 0; i < sizex; i++) {

#ifdef FUSED
			double ave = 0.0;
			int ix = imaterial[i+sizex*j];
			if (ix <= 0) {
				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
					ave += rho_compact_list[ix] * Vf_compact_list[ix];
				}
#else
				for (int idx = mmc_index[-ix]; idx < mmc_index[-ix+1]; idx++) {
					ave += rho_compact_list[idx] * Vf_compact_list[idx];	
				}
#endif
				rho_ave_compact[i+sizex*j] = ave/V[i+sizex*j];
			}
			else {
#endif
				// We use a distinct output array for averages.
				// In case of a pure cell, the average density equals to the total.
				rho_ave_compact[i+sizex*j] = rho_compact[i+sizex*j] / V[i+sizex*j];
#ifdef FUSED
			}
#endif
		}
	}
#ifndef FUSED
  #if defined(OMP)
  #pragma omp parallel for
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
  for (int c = 0; c < mmc_cells; c++) {
    double ave = 0.0;
    for (int m = mmc_index[c]; m < mmc_index[c+1]; m++) {
      ave +=  rho_compact_list[m] * Vf_compact_list[m];
    }
    rho_ave_compact[mmc_i[c]+sizex*mmc_j[c]] = ave/V[mmc_i[c]+sizex*mmc_j[c]];
  }
#endif
  printf("Compact matrix, cell centric, alg 1: %g sec\n", omp_get_wtime()-t1);

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
			int ix = imaterial[i+sizex*j];


			if (ix <= 0) {
#ifdef FUSED
				// NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
				// The solution below is what I believe to good.

				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
					double nm = n[matids[ix]];
					p_compact_list[ix] = (nm * rho_compact_list[ix] * t_compact_list[ix]) / Vf_compact_list[ix];
				}
#else
				for (int idx = mmc_index[-ix]; idx < mmc_index[-ix+1]; idx++) {
					double nm = n[matids[idx]];
					p_compact_list[idx] = (nm * rho_compact_list[idx] * t_compact_list[idx]) / Vf_compact_list[idx];
				}
#endif
#endif
			}
			else {
				// NOTE: HACK: we index materials from zero, but zero can be a list index
				int mat = ix - 1;
				// NOTE: There is no division by Vf here, because the fractional volume is 1.0 in the pure cell case.
				p_compact[i+sizex*j] = n[mat] * rho_compact[i+sizex*j] * t_compact[i+sizex*j];;
			}
		}
	}
#ifndef FUSED
  #if defined(OMP)
  #pragma omp parallel for
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #endif
  for (int idx = 0; idx < mmc_index[mmc_cells]; idx++) {
    double nm = n[matids[idx]];
    p_compact_list[idx] = (nm * rho_compact_list[idx] * t_compact_list[idx]) / Vf_compact_list[idx];
  }
#endif

  printf("Compact matrix, cell centric, alg 2: %g sec\n", omp_get_wtime()-t1);

	// Computational loop 3 - Average density of each material over neighborhood of each cell
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
			// o: outer
			double xo = x[i+sizex*j];
			double yo = y[i+sizex*j];

			// There are at most 9 neighbours in 2D case.
			double dsqr[9];

			// for all neighbours
			for (int nj = -1; nj <= 1; nj++) {
				if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
					continue;

				for (int ni = -1; ni <= 1; ni++) {
					if ((i + ni < 0) || (i + ni >= sizex)) // TODO: better way?
						continue;

					dsqr[(nj+1)*3 + (ni+1)] = 0.0;

					// i: inner
					double xi = x[(i+ni)+sizex*(j+nj)];
					double yi = y[(i+ni)+sizex*(j+nj)];

					dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
					dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
				}
			}

			int ix = imaterial[i+sizex*j];

			if (ix <= 0) {
				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
				#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
				#else
				for (int ix = mmc_index[-imaterial[i+sizex*j]]; ix < mmc_index[-imaterial[i+sizex*j]+1]; ix++) {
				#endif

					int mat = matids[ix];
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
							int jx = imaterial[ci+sizex*cj];

							if (jx <= 0) {
								// condition is 'jx >= 0', this is the equivalent of
								// 'until jx < 0' from the paper
								#ifdef LINKED
								for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
								#else
								for (int jx = mmc_index[-imaterial[ci+sizex*cj]]; jx < mmc_index[-imaterial[ci+sizex*cj]+1]; jx++) {
								#endif
									if (matids[jx] == mat) {
										rho_sum += rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
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
									rho_sum += rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
									Nn += 1;
								}
							} // end if (jx <= 0)
						} // end for (int ni)
					} // end for (int nj)

					rho_mat_ave_compact_list[ix] = rho_sum / Nn;
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
						int jx = imaterial[ci+sizex*cj];

						if (jx <= 0) {
							// condition is 'jx >= 0', this is the equivalent of
							// 'until jx < 0' from the paper
							#ifdef LINKED
							for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
							#else
							for (int jx = mmc_index[-imaterial[ci+sizex*cj]]; jx < mmc_index[-imaterial[ci+sizex*cj]+1]; jx++) {
							#endif
								if (matids[jx] == mat) {
									rho_sum += rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)];
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
								rho_sum += rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)];
								Nn += 1;
							}
						} // end if (jx <= 0)
					} // end for (int ni)
				} // end for (int nj)

				rho_mat_ave_compact[i+sizex*j] = rho_sum / Nn;
			} // end else
		}
	}
  printf("Compact matrix, cell centric, alg 3: %g sec\n", omp_get_wtime()-t1);
  }
}

bool compact_check_results(int sizex, int sizey, int Nmats,
	int *imaterial, int *matids, int *nextfrac,
	double *rho_ave, double *rho_ave_compact,
	double *p, double *p_compact, double *p_compact_list,
	double *rho, double *rho_compact, double *rho_compact_list,  double *rho_mat_ave, double *rho_mat_ave_compact, double *rho_mat_ave_compact_list, int *mmc_index) 
{
	printf("Checking results of compact representation... ");

	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			if (abs(rho_ave[i+sizex*j] - rho_ave_compact[i+sizex*j]) > 0.0001) {
				printf("1. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d)\n",
					rho_ave[i+sizex*j], rho_ave_compact[i+sizex*j], i, j);
				return false;
			}
			int ix = imaterial[i+sizex*j];
			if (ix <= 0) {
#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
#else
        for (int ix = mmc_index[-imaterial[i+sizex*j]]; ix < mmc_index[-imaterial[i+sizex*j]+1]; ix++) {
#endif
					int mat = matids[ix];
					if (abs(p[(i+sizex*j)*Nmats+mat] - p_compact_list[ix]) > 0.0001) {
						printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
							p[(i+sizex*j)*Nmats+mat], p_compact_list[ix], i, j, mat);
						return false;
					}

					if (abs(rho[(i+sizex*j)*Nmats+mat] - rho_compact_list[ix]) > 0.0001) {
						printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
							rho[(i+sizex*j)*Nmats+mat], rho_compact_list[ix], i, j, mat);
						return false;
					}
				}
			}
			else {
				// NOTE: HACK: we index materials from zero, but zero can be a list index
				int mat = ix - 1;
				if (abs(p[(i+sizex*j)*Nmats+mat] - p_compact[i+sizex*j]) > 0.0001) {
					printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						p[(i+sizex*j)*Nmats+mat], p_compact[i+sizex*j], i, j, mat);
					return false;
				}

				if (abs(rho_mat_ave[(i+sizex*j)*Nmats+mat] - rho_mat_ave_compact[i+sizex*j]) > 0.0001) {
					printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						rho_mat_ave[(i+sizex*j)*Nmats+mat], rho_mat_ave_compact[i+sizex*j], i, j, mat);
					return false;
				}
			}
		}
	}

	printf("All tests passed!\n");
	return true;
}
