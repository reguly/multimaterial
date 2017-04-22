#include <math.h>
#include <stdio.h>

void full_matrix_cell_centric(int sizex, int sizey, int Nmats,
	double *rho, double *p, double *Vf, double *t,
	double *V, double *x, double *y,
	double *n, double *rho_ave)
{
	// Cell-centric algorithms
	// Computational loop 1 - average density in cell
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++){
			double ave = 0.0;
			for (int mat = 0; mat < Nmats; mat++) {
				// Optimisation:
				if (Vf[(i+sizex*j)*Nmats+mat] > 0.0)
					ave += rho[(i+sizex*j)*Nmats+mat]*Vf[(i+sizex*j)*Nmats+mat];
			}
			rho_ave[i+sizex*j] = ave/V[i+sizex*j];
		}
	}

	// Computational loop 2 - Pressure for each cell and each material
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
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
	}

	// Computational loop 3 - Average density of each material over neighborhood of each cell
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			// o: outer
			double xo = x[i+sizex*j];
			double yo = y[i+sizex*j];

			// There are at most 9 neighbours in 2D case.
			double dsqr[9];

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

			for (int mat = 0; mat < Nmats; mat++) {
				if (Vf[(i+sizex*j)*Nmats+mat] > 0.0) {
					double rho_sum = 0.0;
					int Nn = 0;

					for (int nj = -1; nj <= 1; nj++) {
						if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
							continue;

						for (int ni = -1; ni <= 1; ni++) {
							if ((i + ni < 0) || (i + ni >= sizex)) // TODO: better way?
								continue;

							if (Vf[((i+ni)+sizex*(j+nj))*Nmats+mat] > 0.0) {
								rho_sum += rho[((i+ni)+sizex*(j+nj))*Nmats+mat] / dsqr[(nj+1)*3 + (ni+1)];
								Nn += 1;
							}
						}
					}
					rho[(i+sizex*j)*Nmats+mat] = rho_sum / Nn;
				}
				else {
					rho[(i+sizex*j)*Nmats+mat] = 0.0;
				}
			}
		}
	}
}

void full_matrix_material_centric(int sizex, int sizey, int Nmats,
	double *rho, double *p, double *Vf, double *t,
	double *V, double *x, double *y,
	double *n, double *rho_ave)
{
	int ncells = sizex * sizey;

	// Material-centric algorithms
	// Computational loop 1 - average density in cell
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			rho_ave[i+sizex*j] = 0.0;
		}
	}

	for (int mat = 0; mat < Nmats; mat++) {
		for (int j = 0; j < sizey; j++) {
			for (int i = 0; i < sizex; i++) {
				// Optimisation:
				if (Vf[ncells*mat + i+sizex*j] > 0.0)
					rho_ave[i+sizex*j] += rho[ncells*mat + i+sizex*j] * Vf[ncells*mat + i+sizex*j];
			}
		}
	}

	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			rho_ave[i+sizex*j] /= V[i+sizex*j];
		}
	}

	// Computational loop 2 - Pressure for each cell and each material
	for (int mat = 0; mat < Nmats; mat++) {
		double nm = n[mat];

		for (int j = 0; j < sizey; j++) {
			for (int i = 0; i < sizex; i++) {
				if (Vf[ncells*mat + i+sizex*j] > 0.0) {
					p[ncells*mat + i+sizex*j] = (nm * rho[ncells*mat + i+sizex*j] * t[ncells*mat + i+sizex*j]) / Vf[ncells*mat + i+sizex*j];
				}
				else {
					p[ncells*mat + i+sizex*j] = 0.0;
				}
			}
		}
	}

	// Computational loop 3 - Average density of each material over neighborhood of each cell
	for (int mat = 0; mat < Nmats; mat++) {
		for (int j = 0; j < sizey; j++) {
			for (int i = 0; i < sizex; i++) {
				if (Vf[ncells*mat + i+sizex*j] > 0.0) {
					// o: outer
					double xo = x[i+sizex*j];
					double yo = y[i+sizex*j];

					double rho_sum = 0.0;
					int Nn = 0;

					for (int nj = -1; nj <= 1; nj++) {
						if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
							continue;

						for (int ni = -1; ni <= 1; ni++) {
							if ((i + ni < 0) || (i + ni >= sizex)) // TODO: better way?
								continue;

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

					rho[ncells*mat + i+sizex*j] = rho_sum / Nn;
				}
				else {
					rho[ncells*mat + i+sizex*j] = 0.0;
				}
			}
		}
	}
}

bool full_matrix_check_results(int sizex, int sizey, int Nmats,
	double *rho_ave, double *rho_ave_mat, double *p, double *p_mat,
	double *rho, double *rho_mat)
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

				if (abs(rho[(i+sizex*j)*Nmats+mat] - rho_mat[ncells*mat + i+sizex*j]) > 0.0001) {
					printf("3. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						rho[(i+sizex*j)*Nmats+mat], rho_mat[ncells*mat + i+sizex*j], i, j, mat);
					return false;
				}
			}
		}
	}

	printf("All tests passed!\n");
	return true;
}
