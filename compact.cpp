#include <math.h>
#include <stdio.h>

void compact_cell_centric(int sizex, int sizey,
	int *imaterial, int *matids, int *nextfrac,
	double *x, double *y, double *n,
	double *rho_compact, double *rho_compact_list, double *rho_ave_compact,
	double *p_compact, double *p_compact_list,
	double *t_compact, double *t_compact_list,
	double *V, double *Vf_compact_list)
{
	// Cell-centric algorithms
	// Computational loop 1 - average density in cell
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			double ave = 0.0;
			int ix = imaterial[i+sizex*j];

			if (ix <= 0) {
				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
					ave += rho_compact_list[ix] * Vf_compact_list[ix];
				}
				rho_ave_compact[i+sizex*j] = ave/V[i+sizex*j];
			}
			else {
				// We use a distinct output array for averages.
				// In case of a pure cell, the average density equals to the total.
				rho_ave_compact[i+sizex*j] = rho_compact[i+sizex*j] / V[i+sizex*j];
			}
		}
	}

	// Computational loop 2 - Pressure for each cell and each material
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			int ix = imaterial[i+sizex*j];

			if (ix <= 0) {
				// NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
				// The solution below is what I believe to good.

				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
					double nm = n[matids[ix]];
					p_compact_list[ix] = (nm * rho_compact_list[ix] * t_compact_list[ix]) / Vf_compact_list[ix];
				}
			}
			else {
				// NOTE: HACK: we index materials from zero, but zero can be a list index
				int mat = ix - 1;
				// NOTE: There is no division by Vf here, because the fractional volume is 1.0 in the pure cell case.
				p_compact[i+sizex*j] = n[mat] * rho_compact[i+sizex*j] * t_compact[i+sizex*j];;
			}
		}
	}

	// Computational loop 3 - Average density of each material over neighborhood of each cell
#define LOOP_BODY(nj_min, nj_max, ni_min, ni_max) \
	do { \
		/* o: outer */ \
		double xo = x[i+sizex*j]; \
		double yo = y[i+sizex*j]; \
		\
		/* There are at most 9 neighbours in 2D case. */ \
		double dsqr[9]; \
		\
		/* for all neighbours */ \
		for (int nj = nj_min; nj <= nj_max; nj++) { \
			for (int ni = ni_min; ni <= ni_max; ni++) { \
				dsqr[(nj+1)*3 + (ni+1)] = 0.0; \
				\
				/* i: inner */ \
				double xi = x[(i+ni)+sizex*(j+nj)]; \
				double yi = y[(i+ni)+sizex*(j+nj)]; \
				\
				dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi); \
				dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi); \
			} \
		} \
		\
		int ix = imaterial[i+sizex*j]; \
		\
		if (ix <= 0) { \
			/* condition is 'ix >= 0', this is the equivalent of \
			 * 'until ix < 0' from the paper */ \
			for (ix = -ix; ix >= 0; ix = nextfrac[ix]) { \
				int mat = matids[ix]; \
				double rho_sum = 0.0; \
				int Nn = 0; \
				\
				/* for all neighbours */ \
				for (int nj = nj_min; nj <= nj_max; nj++) { \
					for (int ni = ni_min; ni <= ni_max; ni++) { \
						int ci = i+ni, cj = j+nj; \
						int jx = imaterial[ci+sizex*cj]; \
						\
						if (jx <= 0) { \
							/* condition is 'jx >= 0', this is the equivalent of \
							 * 'until jx < 0' from the paper */ \
							for (jx = -jx; jx >= 0; jx = nextfrac[jx]) { \
								if (matids[jx] == mat) { \
									rho_sum += rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)]; \
									Nn += 1; \
									\
									/* The loop has an extra condition: "and not found". \
									 * This makes sense, if the material is found, there won't be any more of the same. */ \
									break; \
								} \
							} \
						} \
						else { \
							/* NOTE: In this case, the neighbour is a pure cell, its material index is in jx. \
							 * In contrast, Algorithm 10 loads matids[jx] which I think is wrong. */ \
							\
							/* NOTE: HACK: we index materials from zero, but zero can be a list index */ \
							int mat_neighbour = jx - 1; \
							if (mat == mat_neighbour) { \
								rho_sum += rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)]; \
								Nn += 1; \
							} \
						} /* end if (jx <= 0) */ \
					} /* end for (int ni) */ \
				} /* end for (int nj) */ \
				\
				rho_compact_list[ix] = rho_sum / Nn; \
			} /* end for (ix = -ix) */ \
		} /* end if (ix <= 0) */ \
		else { \
			/* NOTE: In this case, the cell is a pure cell, its material index is in ix. \
			 * In contrast, Algorithm 10 loads matids[ix] which I think is wrong. */ \
			\
			/* NOTE: HACK: we index materials from zero, but zero can be a list index */ \
			int mat = ix - 1; \
			\
			double rho_sum = 0.0; \
			int Nn = 0; \
			\
			/* for all neighbours */ \
			for (int nj = nj_min; nj <= nj_max; nj++) { \
				for (int ni = ni_min; ni <= ni_max; ni++) { \
					int ci = i+ni, cj = j+nj; \
					int jx = imaterial[ci+sizex*cj]; \
					\
					if (jx <= 0) { \
						/* condition is 'jx >= 0', this is the equivalent of \
						 * 'until jx < 0' from the paper */ \
						for (jx = -jx; jx >= 0; jx = nextfrac[jx]) { \
							if (matids[jx] == mat) { \
								rho_sum += rho_compact_list[jx] / dsqr[(nj+1)*3 + (ni+1)]; \
								Nn += 1; \
								\
								/* The loop has an extra condition: "and not found". \
								 * This makes sense, if the material is found, there won't be any more of the same. */ \
								break; \
							} \
						} \
					} \
					else { \
						/* NOTE: In this case, the neighbour is a pure cell, its material index is in jx. \
						 * In contrast, Algorithm 10 loads matids[jx] which I think is wrong. */ \
						\
						/* NOTE: HACK: we index materials from zero, but zero can be a list index */ \
						int mat_neighbour = jx - 1; \
						if (mat == mat_neighbour) { \
							rho_sum += rho_compact[ci+sizex*cj] / dsqr[(nj+1)*3 + (ni+1)]; \
							Nn += 1; \
						} \
					} /* end if (jx <= 0) */ \
				} /* end for (int ni) */ \
			} /* end for (int nj) */ \
			\
			rho_compact[i+sizex*j] = rho_sum / Nn; \
		} /* end else */ \
	} while (0)

	int j, i;

	// j fixed
	j = 0;
	for (i = 1; i < sizex - 1; i++) {
		LOOP_BODY(0, 1, -1, 1);
	}
	j = sizey - 1;
	for (i = 1; i < sizex - 1; i++) {
		LOOP_BODY(-1, 0, -1, 1);
	}

	// i fixed
	i = 0;
	for (j = 1; j < sizey - 1; j++) {
		LOOP_BODY(-1, 1, 0, 1);
	}	
	i = sizex - 1;
	for (j = 1; j < sizey - 1; j++) {
		LOOP_BODY(-1, 1, -1, 0);
	}

	// corners
	j = 0; i = 0;
	LOOP_BODY(0, 1, 0, 1);

	j = 0; i = sizex - 1;
	LOOP_BODY(0, 1, -1, 0);

	j = sizey - 1; i = 0;
	LOOP_BODY(-1, 0, 0, 1);

	j = sizey - 1; i = sizex - 1;
	LOOP_BODY(-1, 0, -1, 0);

	for (j = 1; j < sizey - 1; j++) {
		for (i = 1; i < sizex - 1; i++) {
			LOOP_BODY(-1, 1, -1, 1);
		}
	}

#undef LOOP_BODY
}

bool compact_check_results(int sizex, int sizey, int Nmats,
	int *imaterial, int *matids, int *nextfrac,
	double *rho_ave, double *rho_ave_compact,
	double *p, double *p_compact, double *p_compact_list,
	double *rho, double *rho_compact, double *rho_compact_list)
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
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
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

				if (abs(rho[(i+sizex*j)*Nmats+mat] - rho_compact[i+sizex*j]) > 0.0001) {
					printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						rho[(i+sizex*j)*Nmats+mat], rho_compact[i+sizex*j], i, j, mat);
					return false;
				}
			}
		}
	}

	printf("All tests passed!\n");
	return true;
}
