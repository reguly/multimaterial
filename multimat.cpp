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
#include <string.h>

extern void full_matrix_cell_centric(int sizex, int sizey, int Nmats,
	double *rho, double *p, double *Vf, double *t,
	double *V, double *x, double *y,
	double *n, double *rho_ave);

extern void full_matrix_material_centric(int sizex, int sizey, int Nmats,
	double *rho, double *p, double *Vf, double *t,
	double *V, double *x, double *y,
	double *n, double *rho_ave);

extern bool full_matrix_check_results(int sizex, int sizey, int Nmats,
	double *rho_ave, double *rho_ave_mat, double *p, double *p_mat,
	double *rho, double *rho_mat);

extern void compact_cell_centric(int sizex, int sizey, int Nmats,
	int *imaterial, int *matids, int *nextfrac,
	double *x, double *y, double *n,
	double *rho_compact, double *rho_compact_list, double *rho_ave_compact,
	double *p_compact, double *p_compact_list,
	double *t_compact, double *t_compact_list,
	double *V, double *Vf_compact_list, int mm_len,
	int mmc_cells, int *mmc_index, int *mmc_i, int *mmc_j);

extern bool compact_check_results(int sizex, int sizey, int Nmats,
	int *imaterial, int *matids, int *nextfrac,
	double *rho_ave, double *rho_ave_compact,
	double *p, double *p_compact, double *p_compact_list,
	double *rho, double *rho_compact, double *rho_compact_list, int *mmc_index);


  void initialise_field_static(double *rho, double *t, double *p, int Nmats, int sizex, int sizey) {
	//Pure cells and simple overlaps
	int width = sizex/Nmats;
	//Top
	for (int mat = 0; mat < Nmats/2; mat++) {
#pragma omp parallel for
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

#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
		for (int j = sizey-mat*width-1; j >= sizey-(mat+1)*width-(mat<(Nmats/2-1)); j--) { //+1 for overlap
			for (int i = mat*width; i < sizex-mat*width; i++) {
				rho[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
				t[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
				p[(i+sizex*j)*Nmats+mat+Nmats/2] = 1.0;
			}
		}
	}
	//Fill in corners
#pragma omp parallel for
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
#pragma omp parallel for
	for (int mat=Nmats/2+1; mat < Nmats/2+5; mat++) {
		int i = 2; int j = sizey/2+1;
		rho[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat] = 0.0;t[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 0.0;p[(mat*width+i-2+sizex*j)*Nmats-Nmats/2+mat-1] = 0.0;
	}
}
int main(int argc, char* argv[]) {
	int sizex = 1000;
  if (argc > 1)
    sizex = atoi(argv[1]);
	int sizey = 1000;
  if (argc > 2)
    sizey = atoi(argv[2]);
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
	double *rho_ave_compact = (double*)malloc(ncells*sizeof(double));

	// Cell-centric compact storage
	double *rho_compact = (double*)malloc(ncells*sizeof(double));
	double *p_compact = (double*)malloc(ncells*sizeof(double));
	double *t_compact = (double*)malloc(ncells*sizeof(double));

	int *nmats = (int*)malloc(ncells*sizeof(int));
	int *imaterial = (int*)malloc(ncells*sizeof(int));

	// List
    double mul = ceil((double)sizex/1000.0) * ceil((double)sizey/1000.0);
	int list_size = mul * 49000 * 2 + 600 * 3 + 400 * 4;

	//plain linked list
	int *nextfrac = (int*)malloc(list_size*sizeof(int));
	int *frac2cell = (int*)malloc(list_size*sizeof(int));
	int *matids = (int*)malloc(list_size*sizeof(int));

	//CSR list
	int mmc_cells;
	int *mmc_index = (int *)malloc(list_size*sizeof(int)); //CSR mapping for mix cell idx -> compact list position
	int *mmc_i = (int *)malloc(list_size*sizeof(int)); // mixed cell -> physical cell i coord
	int *mmc_j = (int *)malloc(list_size*sizeof(int)); //  mixed cell -> physical cell j coord
	


	double *Vf_compact_list = (double*)malloc(list_size*sizeof(double));
	double *rho_compact_list = (double*)malloc(list_size*sizeof(double));
	double *t_compact_list = (double*)malloc(list_size*sizeof(double));
	double *p_compact_list = (double*)malloc(list_size*sizeof(double));

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

	initialise_field_static(rho, t, p, Nmats, sizex, sizey);

	FILE *f;
	int print_to_file = 0;

	if (print_to_file==1)
		FILE *f = fopen("map.txt","w");

	//Compute fractions and count cells
	int cell_counts_by_mat[4] = {0,0,0,0};
	mmc_cells = 0;
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
			if (count > 1) mmc_cells++;

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
	printf("Pure cells %d, 2-materials %d, 3 materials %d, 4 materials %d: MMC cells %d\n",
		cell_counts_by_mat[0],cell_counts_by_mat[1],cell_counts_by_mat[2],cell_counts_by_mat[3], mmc_cells);

	if (print_to_file)
		fclose(f);

	// Convert representation to material-centric (using extra buffers)
#pragma omp parallel for
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

	// Copy data from cell-centric full matrix storage to cell-centric compact storage
	imaterial_multi_cell = 0;
	mmc_cells = 0;
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
				rho_compact[i+sizex*j] = rho[(i+sizex*j)*Nmats+mat];
				p_compact[i+sizex*j] = p[(i+sizex*j)*Nmats+mat];
				t_compact[i+sizex*j] = t[(i+sizex*j)*Nmats+mat];
				nmats[i+sizex*j] = -1;
				// NOTE: HACK: we index materials from zero, but zero can be a list index
				imaterial[i+sizex*j] = mat + 1;
			}
			else { // count > 1
				nmats[i+sizex*j] = count;
				// note the minus sign, it needs to be negative
#ifdef LINKED
				imaterial[i+sizex*j] = -imaterial_multi_cell;
#else
				imaterial[i+sizex*j] = -mmc_cells;
#endif
				mmc_index[mmc_cells] = imaterial_multi_cell;
				mmc_i[mmc_cells] = i;
				mmc_j[mmc_cells] = j;
				mmc_cells++;

				for (int list_idx = imaterial_multi_cell; list_idx < imaterial_multi_cell + count; ++list_idx) {
					// if last iteration
					if (list_idx == imaterial_multi_cell + count - 1)
						nextfrac[list_idx] = -1;
					else // not last
						nextfrac[list_idx] = list_idx + 1;

					frac2cell[list_idx] = i+sizex*j;

					int mat = mat_indices[list_idx - imaterial_multi_cell];
					matids[list_idx] = mat;

					Vf_compact_list[list_idx] = Vf[(i+sizex*j)*Nmats+mat];
					rho_compact_list[list_idx] = rho[(i+sizex*j)*Nmats+mat];
					p_compact_list[list_idx] = p[(i+sizex*j)*Nmats+mat];
					t_compact_list[list_idx] = t[(i+sizex*j)*Nmats+mat];
				}

				imaterial_multi_cell += count;
			}
		}
	}
	mmc_index[mmc_cells] = imaterial_multi_cell;

	full_matrix_cell_centric(sizex, sizey, Nmats, rho, p, Vf, t, V, x, y, n, rho_ave);
	full_matrix_material_centric(sizex, sizey, Nmats, rho_mat, p_mat, Vf_mat, t_mat, V, x, y, n, rho_ave_mat);
	// Check results
	if (!full_matrix_check_results(sizex, sizey, Nmats, rho_ave, rho_ave_mat, p, p_mat, rho, rho_mat)) {
		goto end;
	}


	compact_cell_centric(sizex, sizey, Nmats, imaterial, matids, nextfrac, x, y, n,
		rho_compact, rho_compact_list, rho_ave_compact, p_compact, p_compact_list,
		t_compact, t_compact_list, V, Vf_compact_list, imaterial_multi_cell,
		mmc_cells, mmc_index, mmc_i, mmc_j);
	// Check results
	if (!compact_check_results(sizex, sizey, Nmats, imaterial, matids, nextfrac,
			rho_ave, rho_ave_compact, p, p_compact, p_compact_list, rho, rho_compact, rho_compact_list, mmc_index))
	{
		goto end;
	}

end:
	free(rho_mat); free(p_mat); free(Vf_mat); free(t_mat);
	free(rho); free(p); free(Vf); free(t);
	free(V); free(x); free(y);
	free(n);
	free(rho_ave); free(rho_ave_mat); free(rho_ave_compact);

	free(rho_compact); free(p_compact); free(t_compact);
	free(nmats); free(imaterial);
	free(nextfrac); free(frac2cell); free(matids);
	free(Vf_compact_list); free(rho_compact_list);
	free(t_compact_list); free(p_compact_list);
	return 0;
}
