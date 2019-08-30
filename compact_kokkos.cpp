#include <math.h>
#include <stdio.h>
#include <omp.h>
//extern "C" double omp_get_wtime(); 
#include <Kokkos_Core.hpp>

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
int init = 0;
void compact_cell_centric(full_data cc, compact_data ccc, double &a1, double &a2, double &a3, int argc, char** argv)
{
	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;
	int mmc_cells = ccc.mmc_cells;
  int mm_len = ccc.mm_len;
  if (init++==0) {Kokkos::initialize (argc, argv); init=1;}
  {
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > rho_compact_b(ccc.rho_compact, sizex*sizey);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > rho_compact_list_b(ccc.rho_compact_list, mm_len);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > rho_mat_ave_compact_b(ccc.rho_mat_ave_compact, sizex*sizey);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > rho_mat_ave_compact_list_b(ccc.rho_mat_ave_compact_list, mm_len);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > p_compact_b(ccc.p_compact, sizex*sizey);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > p_compact_list_b(ccc.p_compact_list, mm_len);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > Vf_compact_list_b(ccc.Vf_compact_list, mm_len);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > t_compact_b(ccc.t_compact, sizex*sizey);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > t_compact_list_b(ccc.t_compact_list, mm_len);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > V_b(ccc.V, sizex*sizey);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > x_b(ccc.x, sizex*sizey);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > y_b(ccc.y, sizex*sizey);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > n_b(ccc.n, Nmats);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > rho_ave_compact_b(ccc.rho_ave_compact, sizex*sizey);
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > imaterial_b(ccc.imaterial, sizex*sizey);
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > matids_b(ccc.matids, mm_len);
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > nextfrac_b(ccc.nextfrac, mm_len);
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > mmc_index_b(ccc.mmc_index, mmc_cells+1);
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > mmc_i_b(ccc.mmc_i, mmc_cells);
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > mmc_j_b(ccc.mmc_j, mmc_cells);
  Kokkos::View<double*> rho_compact("rho_compact", sizex*sizey);
  Kokkos::View<double*> rho_compact_list("rho_compact_list", mm_len);
  Kokkos::View<double*> rho_mat_ave_compact("rho_mat_ave_compact", sizex*sizey);
  Kokkos::View<double*> rho_mat_ave_compact_list("rho_mat_ave_compact_list", mm_len);
  Kokkos::View<double*> p_compact("p_compact", sizex*sizey);
  Kokkos::View<double*> p_compact_list("p_compact_list", mm_len);
  Kokkos::View<double*> Vf_compact_list("Vf_compact_list", mm_len);
  Kokkos::View<double*> t_compact("t_compact", sizex*sizey);
  Kokkos::View<double*> t_compact_list("t_compact_list", mm_len);
  Kokkos::View<double*> V("V", sizex*sizey);
  Kokkos::View<double*> x("x", sizex*sizey);
  Kokkos::View<double*> y("y", sizex*sizey);
  Kokkos::View<double*> n("n", Nmats);
  Kokkos::View<double*> rho_ave_compact("rho_ave_compact", sizex*sizey);
  Kokkos::View<int*> imaterial("imaterial", sizex*sizey);
  Kokkos::View<int*> matids("matids", mm_len);
  Kokkos::View<int*> nextfrac("nextfrac", mm_len);
  Kokkos::View<int*> mmc_index("mmc_index", mmc_cells+1);
  Kokkos::View<int*> mmc_i("mmc_i", mmc_cells);
  Kokkos::View<int*> mmc_j("mmc_j", mmc_cells);
  Kokkos::deep_copy(rho_compact, rho_compact_b);
  Kokkos::deep_copy(rho_compact_list, rho_compact_list_b);
  Kokkos::deep_copy(rho_mat_ave_compact, rho_mat_ave_compact_b);
  Kokkos::deep_copy(rho_mat_ave_compact_list, rho_mat_ave_compact_list_b);
  Kokkos::deep_copy(p_compact, p_compact_b);
  Kokkos::deep_copy(p_compact_list, p_compact_list_b);
  Kokkos::deep_copy(Vf_compact_list, Vf_compact_list_b);
  Kokkos::deep_copy(t_compact, t_compact_b);
  Kokkos::deep_copy(t_compact_list, t_compact_list_b);
  Kokkos::deep_copy(V, V_b);
  Kokkos::deep_copy(x, x_b);
  Kokkos::deep_copy(y, y_b);
  Kokkos::deep_copy(n, n_b);
  Kokkos::deep_copy(rho_ave_compact, rho_ave_compact_b);
  Kokkos::deep_copy(imaterial, imaterial_b);
  Kokkos::deep_copy(matids, matids_b);
  Kokkos::deep_copy(nextfrac, nextfrac_b);
  Kokkos::deep_copy(mmc_index, mmc_index_b);
  Kokkos::deep_copy(mmc_i, mmc_i_b);
  Kokkos::deep_copy(mmc_j, mmc_j_b);
  if (Nmats < 1)
    printf("%d\n", Nmats);

   
	// Cell-centric algorithms
	// Computational loop 1 - average density in cell
  Kokkos::fence();
  double t1 = omp_get_wtime();
      Kokkos::parallel_for (sizex*sizey, KOKKOS_LAMBDA (const int id) {
              int i = id%sizex;
              int j = id/sizex;
      #ifdef FUSED
			double ave = 0.0;
			int ix = imaterial(i+sizex*j);
			if (ix <= 0) {
				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
      #ifdef LINKED
      #pragma novector
				for (ix = -ix; ix >= 0; ix = nextfrac(ix)) {
					ave += rho_compact_list(ix) * Vf_compact_list(ix);
				}
      #else
				for (int idx = mmc_index(-ix); idx < mmc_index(-ix+1); idx++) {
					ave += rho_compact_list(idx) * Vf_compact_list(idx);	
				}
      #endif
				rho_ave_compact(i+sizex*j) = ave/V(i+sizex*j);
			}
			else {
      #endif
				// We use a distinct output array for averages.
				// In case of a pure cell, the average density equals to the total.
				rho_ave_compact(i+sizex*j) = rho_compact(i+sizex*j) / V(i+sizex*j);
      #ifdef FUSED
			}
      #endif
    });

   Kokkos::parallel_for (mmc_cells, KOKKOS_LAMBDA (const int c) {
    double ave = 0.0;
    for (int m = mmc_index(c); m < mmc_index(c+1); m++) {
      ave +=  rho_compact_list(m) * Vf_compact_list(m);
    }
    rho_ave_compact(mmc_i(c)+sizex*mmc_j(c)) = ave/V(mmc_i(c)+sizex*mmc_j(c));
  });

  Kokkos::fence();
  a1 += omp_get_wtime()-t1;
#ifdef DEBUG
  printf("Compact matrix, cell centric, alg 1: %g sec\n", a1);
#endif
	// Computational loop 2 - Pressure for each cell and each material
  Kokkos::fence();
  t1 = omp_get_wtime();

      Kokkos::parallel_for (sizex*sizey, KOKKOS_LAMBDA (const int id) {
              int i = id%sizex;
              int j = id/sizex;

			int ix = imaterial(i+sizex*j);


#ifdef FUSED
			if (ix <= 0) {
				// NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
				// The solution below is what I believe to good.

				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = nextfrac(ix)) {
					double nm = n(matids(ix));
					p_compact_list(ix) = (nm * rho_compact_list(ix) * t_compact_list(ix)) / Vf_compact_list(ix);
				}
#else
				for (int idx = mmc_index(-ix); idx < mmc_index(-ix+1); idx++) {
					double nm = n(matids(idx));
					p_compact_list(idx) = (nm * rho_compact_list(idx) * t_compact_list(idx)) / Vf_compact_list(idx);
				}
#endif
			}
			else {
#else
        if (ix > 0) {
#endif //FUSED

				// NOTE: HACK: we index materials from zero, but zero can be a list index
				int mat = ix - 1;
				// NOTE: There is no division by Vf here, because the fractional volume is 1.0 in the pure cell case.
				p_compact(i+sizex*j) = n(mat) * rho_compact(i+sizex*j) * t_compact(i+sizex*j);
			}
    });

#ifndef FUSED
   Kokkos::parallel_for (ccc.mmc_index[mmc_cells], KOKKOS_LAMBDA (const int idx) {
    double nm = n(matids(idx));
    p_compact_list(idx) = (nm * rho_compact_list(idx) * t_compact_list(idx)) / Vf_compact_list(idx);
  });
#endif

  Kokkos::fence();
  a2 += omp_get_wtime()-t1;
#ifdef DEBUG
  printf("Compact matrix, cell centric, alg 2: %g sec\n", a2);
#endif

	// Computational loop 3 - Average density of each material over neighborhood of each cell
  Kokkos::fence();
  t1 = omp_get_wtime();
           Kokkos::parallel_for ((sizex-2)*(sizey-2), KOKKOS_LAMBDA (const int id) {
              int i = id%(sizex-2)+1;
              int j = id/(sizex-2)+1;

			// o: outer
			double xo = x(i+sizex*j);
			double yo = y(i+sizex*j);

			// There are at most 9 neighbours in 2D case.
			double dsqr[9];

			// for all neighbours
			for (int nj = -1; nj <= 1; nj++) {
				for (int ni = -1; ni <= 1; ni++) {

					dsqr[(nj+1)*3 + (ni+1)] = 0.0;

					// i: inner
					double xi = x((i+ni)+sizex*(j+nj));
					double yi = y((i+ni)+sizex*(j+nj));

					dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
					dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
				}
			}

			int ix = imaterial(i+sizex*j);

			if (ix <= 0) {
				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
				#ifdef LINKED
				for (ix = -ix; ix >= 0; ix = nextfrac(ix)) {
				#else
				for (int ix = mmc_index(-imaterial(i+sizex*j)); ix < mmc_index(-imaterial(i+sizex*j)+1); ix++) {
				#endif

					int mat = matids(ix);
					double rho_sum = 0.0;
					int Nn = 0;

					// for all neighbours
					for (int nj = -1; nj <= 1; nj++) {
						for (int ni = -1; ni <= 1; ni++) {
							int ci = i+ni, cj = j+nj;
							int jx = imaterial(ci+sizex*cj);

							if (jx <= 0) {
								// condition is 'jx >= 0', this is the equivalent of
								// 'until jx < 0' from the paper
								#ifdef LINKED
								for (jx = -jx; jx >= 0; jx = nextfrac(jx)) {
								#else
								for (int jx = mmc_index(-imaterial(ci+sizex*cj)); jx < mmc_index(-imaterial(ci+sizex*cj)+1); jx++) {
								#endif
									if (matids(jx) == mat) {
										rho_sum += rho_compact_list(jx) / dsqr[(nj+1)*3 + (ni+1)];
										Nn += 1;

										// The loop has an extra condition: "and not found".
										// This makes sense, if the material is found, there won't be any more of the same.
										break;
									}
								}
							}
							else {
								// NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
								// In contrast, Algorithm 10 loads matids(jx) which I think is wrong.

								// NOTE: HACK: we index materials from zero, but zero can be a list index
								int mat_neighbour = jx - 1;
								if (mat == mat_neighbour) {
									rho_sum += rho_compact(ci+sizex*cj) / dsqr[(nj+1)*3 + (ni+1)];
									Nn += 1;
								}
							} // end if (jx <= 0)
						} // end for (int ni)
					} // end for (int nj)

					rho_mat_ave_compact_list(ix) = rho_sum / Nn;
				} // end for (ix = -ix)
			} // end if (ix <= 0)
			else {
				// NOTE: In this case, the cell is a pure cell, its material index is in ix.
				// In contrast, Algorithm 10 loads matids(ix) which I think is wrong.

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
						int jx = imaterial(ci+sizex*cj);

						if (jx <= 0) {
							// condition is 'jx >= 0', this is the equivalent of
							// 'until jx < 0' from the paper
							#ifdef LINKED
							for (jx = -jx; jx >= 0; jx = nextfrac(jx)) {
							#else
							for (int jx = mmc_index(-imaterial(ci+sizex*cj)); jx < mmc_index(-imaterial(ci+sizex*cj)+1); jx++) {
							#endif
								if (matids(jx) == mat) {
									rho_sum += rho_compact_list(jx) / dsqr[(nj+1)*3 + (ni+1)];
									Nn += 1;

									// The loop has an extra condition: "and not found".
									// This makes sense, if the material is found, there won't be any more of the same.
									break;
								}
							}
						}
						else {
							// NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
							// In contrast, Algorithm 10 loads matids(jx) which I think is wrong.

							// NOTE: HACK: we index materials from zero, but zero can be a list index
							int mat_neighbour = jx - 1;
							if (mat == mat_neighbour) {
								rho_sum += rho_compact(ci+sizex*cj) / dsqr[(nj+1)*3 + (ni+1)];
								Nn += 1;
							}
						} // end if (jx <= 0)
					} // end for (int ni)
				} // end for (int nj)

				rho_mat_ave_compact(i+sizex*j) = rho_sum / Nn;
			} // end else
            });
  Kokkos::fence();
  a3 = omp_get_wtime()-t1;
#ifdef DEBUG
  printf("Compact matrix, cell centric, alg 3: %g sec\n", a3);
#endif
  Kokkos::deep_copy(rho_compact_b, rho_compact);
  Kokkos::deep_copy(rho_compact_list_b, rho_compact_list);
  Kokkos::deep_copy(rho_mat_ave_compact_b, rho_mat_ave_compact);
  Kokkos::deep_copy(rho_mat_ave_compact_list_b, rho_mat_ave_compact_list);
  Kokkos::deep_copy(p_compact_b, p_compact);
  Kokkos::deep_copy(p_compact_list_b, p_compact_list);
  Kokkos::deep_copy(Vf_compact_list_b, Vf_compact_list);
  Kokkos::deep_copy(t_compact_b, t_compact);
  Kokkos::deep_copy(t_compact_list_b, t_compact_list);
  Kokkos::deep_copy(V_b, V);
  Kokkos::deep_copy(x_b, x);
  Kokkos::deep_copy(y_b, y);
  Kokkos::deep_copy(n_b, n);
  Kokkos::deep_copy(rho_ave_compact_b, rho_ave_compact);
  Kokkos::deep_copy(imaterial_b, imaterial);
  Kokkos::deep_copy(matids_b, matids);
  Kokkos::deep_copy(nextfrac_b, nextfrac);
  Kokkos::deep_copy(mmc_index_b, mmc_index);
  Kokkos::deep_copy(mmc_i_b, mmc_i);
  Kokkos::deep_copy(mmc_j_b, mmc_j);
        }
 if (init==11) Kokkos::finalize ();
}

bool compact_check_results(full_data cc, compact_data ccc) 
{

	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;
	int mmc_cells = ccc.mmc_cells;

#ifdef DEBUG
	printf("Checking results of compact representation... ");
#endif

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

#ifdef DEBUG
	printf("All tests passed!\n");
#endif
	return true;
}
