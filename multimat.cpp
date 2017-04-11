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

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>

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

	//Allocate per-cell only datasets
	double *V = (double*)malloc(ncells*sizeof(double));
	double *x = (double*)malloc(ncells*sizeof(double));
	double *y = (double*)malloc(ncells*sizeof(double));

	//Allocate output datasets
	double *rho_ave = (double*)malloc(ncells*sizeof(double));

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
				exit(1);
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

	//Algorithm 1 - average density in cell
	for (int j = 0; j < sizey; j++) {
		for (int i = 0; i < sizex; i++){
			double ave = 0.0;
			for (int mat = 0; mat < Nmats; mat++) {
				ave += rho[(i+sizex*j)*Nmats+mat]*Vf[(i+sizex*j)*Nmats+mat];
			}
			rho_ave[i+sizex*j] = ave/V[i+sizex*j];
		}
	}

	//Algorithm 2

	//Algorithm 3


	free(rho); free(p); free(Vf); free(t);
	free(V); free(x); free(y);
	free(rho_ave);
	return 0;
}
