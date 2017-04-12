/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* Copyright (c) 2013, Istvan Reguly and others. 
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Istvan Reguly ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief initial version of mutli-material code with full dense matrix representaiton
  * @author Istvan Reguly
  */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>

void cell_centric_calculations(int sizex, int sizey, int Nmats,
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
			// TODO: is this still local to CPU (i.e. no need for memory transfer)?
			double dsqr[9];

			for (int nj = -1; nj <= 1; nj++) {
				if ((j + nj < 0) || (j + nj >= sizey)) // TODO: better way?
					continue;

				for (int ni = -1; nj <= 1; nj++) {
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

						for (int ni = -1; nj <= 1; nj++) {
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

void material_centric_calculations(int sizex, int sizey, int Nmats,
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

						for (int ni = -1; nj <= 1; nj++) {
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

int main(int argc, char* argv[]) {
	int sizex = 1000;
	int sizey = 1000;
	int ncells = sizex*sizey;

	int Nmats = 50;

	//Allocate the four state variables for all Nmats materials and all cells 
	//density
	double *rho =  (double*)malloc(Nmats*ncells*sizeof(double));
	memset(rho, 0, Nmats*ncells*sizeof(double));
	//pressure
	double *p = (double*)malloc(Nmats*ncells*sizeof(double));
	memset(p, 0, Nmats*ncells*sizeof(double));
	//Fractional volume
	double *Vf = (double*)malloc(Nmats*ncells*sizeof(double));
	memset(Vf, 0, Nmats*ncells*sizeof(double));
	//temperature
	double *t = (double*)malloc(Nmats*ncells*sizeof(double));
	memset(t, 0, Nmats*ncells*sizeof(double));

	// Buffers for material-centric representation
	//density
	double *rho_mat =  (double*)malloc(Nmats*ncells*sizeof(double));
	//pressure
	double *p_mat = (double*)malloc(Nmats*ncells*sizeof(double));
	//Fractional volume
	double *Vf_mat = (double*)malloc(Nmats*ncells*sizeof(double));
	//temperature
	double *t_mat = (double*)malloc(Nmats*ncells*sizeof(double));

	//Allocate per-cell only datasets
	double *V = (double*)malloc(ncells*sizeof(double));
	double *x = (double*)malloc(ncells*sizeof(double));
	double *y = (double*)malloc(ncells*sizeof(double));

	//Allocate per-material only datasets
	double *n = (double*)malloc(Nmats*sizeof(double)); // number of moles

	//Allocate output datasets
	double *rho_ave = (double*)malloc(ncells*sizeof(double));
	double *rho_ave_mat = (double*)malloc(ncells*sizeof(double));

	// Cell-centric mixed material storage
	double *rho_mixed = (double*)malloc(ncells*sizeof(double));
	double *p_mixed = (double*)malloc(ncells*sizeof(double));
	double *t_mixed = (double*)malloc(ncells*sizeof(double));

	int *nmats = (int*)malloc(ncells*sizeof(int));
	int *imaterial = (int*)malloc(ncells*sizeof(int));

	// List
	int list_size = 49000 * 2 + 600 * 3 + 400 * 4;

	int *nextfrac = (int*)malloc(list_size*sizeof(int));
	int *frac2cell = (int*)malloc(list_size*sizeof(int));
	int *matids = (int*)malloc(list_size*sizeof(int));

	double *Vf_mixed_list = (double*)malloc(list_size*sizeof(double));
	double *rho_mixed_list = (double*)malloc(list_size*sizeof(double));
	double *t_mixed_list = (double*)malloc(list_size*sizeof(double));
	double *p_mixed_list = (double*)malloc(list_size*sizeof(double));

	int imaterial_pure_cell;
	int imaterial_multi_cell;

	//Initialise arrays
	double dx = 1.0/sizex;
	double dy = 1.0/sizey;
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			V[i+j*sizex] = dx*dy;
			x[i+j*sizex] = dx*i;
			y[i+j*sizex] = dy*j;
		}
	}

	for (int mat = 0; mat < Nmats; mat++) {
		n[mat] = 1.0; // dummy value
	}

	//Pure cells and simple overlaps
	int width = sizex/Nmats;
	//Top
	for (int mat = 0; mat < Nmats/2; mat++) {
		for (int j = mat*width; j < sizey/2; j++) {
			for (int i = mat*width-(mat>0); i < (mat+1)*width; i++) { //+1 for overlap
				rho[(i+sizex*j)*Nmats+mat] = 1.0;
				t[(i+sizex*j)*Nmats+mat] = 1.0;
				p[(i+sizex*j)*Nmats+mat] = 1.0;
			}
			for (int i = sizex-mat*width-1; i >= sizex-(mat+1)*width-1; i--) { //+1 for overlap
				rho[(i+sizex*j)*Nmats+mat] = 1.0;
				t[(i+sizex*j)*Nmats+mat] = 1.0;
				p[(i+sizex*j)*Nmats+mat] = 1.0;
			}
		}

		for (int j = mat*width-(mat>0); j < (mat+1)*width; j++) { //+1 for overlap
			for (int i = mat*width-(mat>0); i < sizex-mat*width; i++) {
				rho[(i+sizex*j)*Nmats+mat] = 1.0;
				t[(i+sizex*j)*Nmats+mat] = 1.0;
				p[(i+sizex*j)*Nmats+mat] = 1.0;
			}
		}
	}
	
	//Bottom
	for (int mat = 0; mat < Nmats/2; mat++) {
		for (int j = sizey/2-1; j < sizey-mat*width; j++) {
			for (int i = mat*width-(mat>0); i < (mat+1)*width; i++) { //+1 for overlap
				rho[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
				t[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
				p[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
			}
			for (int i = sizex-mat*width-1; i >= sizex-(mat+1)*width-1; i--) { //+1 for overlap
				rho[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
				t[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
				p[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
			}
		}
		for (int j = sizey-mat*width-1; j >= sizey-(mat+1)*width-(mat<(Nmats/2-1)); j--) { //+1 for overlap
			for (int i = mat*width; i < sizex-mat*width; i++) {
				rho[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
				t[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
				p[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
			}
		}
	}
	//Fill in corners
	for (int mat = 1; mat < Nmats/2; mat++) {
		for (int j = sizey/2-3; j < sizey/2-1;j++)
			for (int i = 2; i < 5; i++) {
				//x neighbour material
				rho[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;
				rho[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;t[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;
				//y neighbour material
				rho[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;
				rho[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat] = 1.0;t[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat] = 1.0;
				//x-y neighbour material
				rho[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;
				rho[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;t[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat] = 1.0;
			}
		for (int j = sizey/2; j < sizey/2+2;j++)
			for (int i = 2; i < 5; i++) {
				//x neighbour material
				rho[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats+Nmats/2+mat-1] = 1.0;
				rho[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat] = 1.0;t[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats+Nmats/2+mat] = 1.0;
				//y neighbour material
				rho[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;
				rho[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;t[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;

			}
	}
	int only_8 = 0;
	for (int mat = Nmats/2+1; mat < Nmats; mat++) {
		for (int j = sizey/2-3; j < sizey/2-1;j++)
			for (int i = 2; i < 5; i++) {
				//x neighbour material
				rho[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;
				rho[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat] = 1.0;t[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat] = 1.0;
				//y neighbour material
				rho[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;
				rho[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;t[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;
			}
		for (int j = sizey/2; j < sizey/2+2;j++)
			for (int i = 2; i < 4; i++) {
				if (i < 3 && only_8<6) {
					//y neighbour material
					rho[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;
					rho[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat] = 1.0;t[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat] = 1.0;
				}
				if (i==2 && only_8==0) {
					//x-y neighbour material
					rho[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;
					rho[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat-1] = 1.0;t[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats-Nmats/2+mat] = 1.0;
				}
				//x neighbour material
				if (mat >= Nmats-8 && j==sizey/2+1 && i==3) if (only_8++>=4) {
					break;
				}
				rho[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;t[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;p[(mat*width+i-2+sizex*j)*Nmats+mat-1] = 1.0;
				rho[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;t[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;p[(mat*width-i+sizex*j)*Nmats+mat] = 1.0;
			}
	}
	for (int mat=Nmats/2+1; mat < Nmats/2+5; mat++) {
		int i = 2; int j = sizey/2+1;
		rho[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat] = 0.0;t[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 0.0;p[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 0.0;
	}

	FILE *f;
	int print_to_file = 0;

	if (print_to_file==1)
		FILE *f = fopen("map.txt","w");

	//Compute fractions and count cells
	int cell_counts_by_mat[4] = {0,0,0,0};
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			int count = 0;
			for (int mat = 0; mat < Nmats; mat++) {
				count += rho[(i+sizex*j)*Nmats+mat]!=0.0;
			}
			if (count == 0) {
				printf("Error: no materials in cell %d %d\n",i,j);
				if (print_to_file)
					fclose(f);

				goto end;
			}
			cell_counts_by_mat[count-1]++;

			if (print_to_file) {
				if (i!=0) fprintf(f,", %d",count);
				else fprintf(f,"%d",count);
			}

			for (int mat = 0; mat < Nmats; mat++) {
				if (rho[(i+sizex*j)*Nmats+mat]!=0.0) Vf[(i+sizex*j)*Nmats+mat]=1.0/count;
			}
		}
		if (print_to_file)
			fprintf(f,"\n");
	}
	printf("Pure cells %d, 2-materials %d, 3 materials %d, 4 materials %d\n",
		cell_counts_by_mat[0],cell_counts_by_mat[1],cell_counts_by_mat[2],cell_counts_by_mat[3]);

	if (print_to_file)
		fclose(f);

	// Convert representation to material-centric (using extra buffers)
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			for (int mat = 0; mat < Nmats; mat++) {
				rho_mat[ncells*mat + i+sizex*j] = rho[(i+sizex*j)*Nmats+mat];
				p_mat[ncells*mat + i+sizex*j] = p[(i+sizex*j)*Nmats+mat];
				Vf_mat[ncells*mat + i+sizex*j] = Vf[(i+sizex*j)*Nmats+mat];
				t_mat[ncells*mat + i+sizex*j] = t[(i+sizex*j)*Nmats+mat];
			}
		}
	}

	cell_centric_calculations(sizex, sizey, Nmats, rho, p, Vf, t, V, x, y, n, rho_ave);
	material_centric_calculations(sizex, sizey, Nmats, rho_mat, p_mat, Vf_mat, t_mat, V, x, y, n, rho_ave_mat);

	// Check results
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			if (abs(rho_ave[i+sizex*j] - rho_ave_mat[i+sizex*j]) > 0.0001) {
				printf("1. cell-centric and material-centric values are not equal! (%f, %f, %d, %d)\n",
					rho_ave[i+sizex*j], rho_ave_mat[i+sizex*j], i, j);
				goto end;
			}

			for (int mat = 0; mat < Nmats; mat++) {
				if (abs(p[(i+sizex*j)*Nmats+mat] - p_mat[ncells*mat + i+sizex*j]) > 0.0001) {
					printf("2. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						p[(i+sizex*j)*Nmats+mat], p_mat[ncells*mat + i+sizex*j], i, j, mat);
					goto end;
				}

				if (abs(rho[(i+sizex*j)*Nmats+mat] - rho_mat[ncells*mat + i+sizex*j]) > 0.0001) {
					printf("3. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
						rho[(i+sizex*j)*Nmats+mat], rho_mat[ncells*mat + i+sizex*j], i, j, mat);
					goto end;
				}
			}
		}
	}

	printf("All tests passed!\n");

	// Copy data from cell-centric full matrix storage to cell-centric mixed material storage
	imaterial_pure_cell = 1; // TODO: why is this needed?
	imaterial_multi_cell = 0;

	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			int mat_indices[4] = { -1, -1, -1, -1 };
			int matindex = 0;
			int count = 0;

			for (int mat = 0; mat < Nmats; mat++) {
				if (rho[(i+sizex*j)*Nmats+mat]!=0.0) {
					mat_indices[matindex++] = mat;
					count += 1;
				}
			}

			if (count == 0) {
				printf("Error: no materials in cell %d %d\n",i,j);
				goto end;
			}

			if (count == 1) {
				int mat = mat_indices[0];
				rho_mixed[i+sizex*j] = rho[(i+sizex*j)*Nmats+mat];
				p_mixed[i+sizex*j] = p[(i+sizex*j)*Nmats+mat];
				t_mixed[i+sizex*j] = t[(i+sizex*j)*Nmats+mat];
				nmats[i+sizex*j] = -1;
				imaterial[i+sizex*j] = imaterial_pure_cell++;
			}
			else { // count > 1
				nmats[i+sizex*j] = count;
				// note the minus sign, it needs to be negative
				imaterial[i+sizex*j] = -imaterial_multi_cell;

				for (int list_idx = imaterial_multi_cell; list_idx < imaterial_multi_cell + count; ++list_idx) {
					// if last iteration
					if (list_idx == imaterial_multi_cell + count - 1)
						nextfrac[list_idx] = -1;
					else // not last
						nextfrac[list_idx] = list_idx + 1;

					frac2cell[list_idx] = i+sizex*j;

					int mat = mat_indices[list_idx - imaterial_multi_cell];
					matids[list_idx] = mat;

					Vf_mixed_list[list_idx] = Vf[(i+sizex*j)*Nmats+mat];
					rho_mixed_list[list_idx] = rho[(i+sizex*j)*Nmats+mat];
					p_mixed_list[list_idx] = p[(i+sizex*j)*Nmats+mat];
					t_mixed_list[list_idx] = t[(i+sizex*j)*Nmats+mat];
				}

				imaterial_multi_cell += count;
			}
		}
	}

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
					ave += rho_mixed_list[ix] * Vf_mixed_list[ix];
				}
				rho_mixed[i+sizex*j] = ave/V[i+sizex*j];
			}
		}
	}

	// Check results
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++) {
			if (abs(rho_ave[i+sizex*j] - rho_mixed[i+sizex*j]) > 0.0001) {
				printf("1. cell-centric and mixed material cell-centric values are not equal! (%f, %f, %d, %d)\n",
					rho_ave[i+sizex*j], rho_mixed[i+sizex*j], i, j);
				goto end;
			}
		}
	}

end:
	free(rho_mat); free(p_mat); free(Vf_mat); free(t_mat);
	free(rho); free(p); free(Vf); free(t);
	free(V); free(x); free(y);
	free(n);
	free(rho_ave); free(rho_ave_mat);

	free(rho_mixed); free(p_mixed); free(t_mixed);
	free(nmats); free(imaterial);
	free(nextfrac); free(frac2cell); free(matids);
	free(Vf_mixed_list); free(rho_mixed_list);
	free(t_mixed_list); free(p_mixed_list);
	return 0;
}
