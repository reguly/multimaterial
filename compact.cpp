#include <math.h>
#include <stdio.h>
//#include <omp.h>
extern "C" double omp_get_wtime(); 

#ifdef SYCL
#include <CL/sycl.hpp>
#include <iostream>

template <typename T>
void cp_to_device(cl::sycl::queue &queue, cl::sycl::buffer<T, 1> &b, T* data) {
  queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = b.template get_access<cl::sycl::access::mode::write>(cgh);
      cgh.copy(data, acc);
      });
}

template <typename T>
void cp_to_host(cl::sycl::queue &queue, cl::sycl::buffer<T, 1> &b, T* data) {
  queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(acc, data);
      });
}
#endif

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

void compact_cell_centric(full_data cc, compact_data ccc, double &a1, double &a2, double &a3)
{
	int sizex = cc.sizex;
	int sizey = cc.sizey;
	int Nmats = cc.Nmats;
	int mmc_cells = ccc.mmc_cells;
  int mm_len = ccc.mm_len;
#ifndef SYCL
  double * __restrict__ rho_compact = ccc.rho_compact;
  double * __restrict__ rho_compact_list = ccc.rho_compact_list;
  double * __restrict__ rho_mat_ave_compact = ccc.rho_mat_ave_compact;
  double * __restrict__ rho_mat_ave_compact_list = ccc.rho_mat_ave_compact_list;
  double * __restrict__ p_compact = ccc.p_compact;
  double * __restrict__ p_compact_list = ccc.p_compact_list;
  double * __restrict__ Vf_compact_list = ccc.Vf_compact_list;
  double * __restrict__ t_compact = ccc.t_compact;
  double * __restrict__ t_compact_list = ccc.t_compact_list;
  double * __restrict__ V = ccc.V;
  double * __restrict__ x = ccc.x;
  double * __restrict__ y = ccc.y;
  double * __restrict__ n = ccc.n;
  double * __restrict__ rho_ave_compact = ccc.rho_ave_compact;
  int * __restrict__ imaterial = ccc.imaterial;
  int * __restrict__ matids = ccc.matids;
  int * __restrict__ nextfrac = ccc.nextfrac;
  int * __restrict__ mmc_index = ccc.mmc_index;
  int * __restrict__ mmc_i = ccc.mmc_i;
  int * __restrict__ mmc_j = ccc.mmc_j;
#else
  //cl::sycl::cpu_selector s;
  cl::sycl::gpu_selector s;
  cl::sycl::queue queue(s);
//  std::cout << "Running on " << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";
  cl::sycl::buffer<double, 1> rho_compact_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<double, 1> rho_compact_list_b((cl::sycl::range<1>(mm_len)));
  cl::sycl::buffer<double, 1> rho_mat_ave_compact_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<double, 1> rho_mat_ave_compact_list_b((cl::sycl::range<1>(mm_len)));
  cl::sycl::buffer<double, 1> p_compact_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<double, 1> p_compact_list_b((cl::sycl::range<1>(mm_len)));
  cl::sycl::buffer<double, 1> Vf_compact_list_b((cl::sycl::range<1>(mm_len)));
  cl::sycl::buffer<double, 1> t_compact_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<double, 1> t_compact_list_b((cl::sycl::range<1>(mm_len)));
  cl::sycl::buffer<double, 1> V_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<double, 1> x_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<double, 1> y_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<double, 1> n_b((cl::sycl::range<1>(Nmats)));
  cl::sycl::buffer<double, 1> rho_ave_compact_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<int, 1> imaterial_b((cl::sycl::range<1>(sizex*sizey)));
  cl::sycl::buffer<int, 1> matids_b((cl::sycl::range<1>(mm_len)));
  cl::sycl::buffer<int, 1> nextfrac_b((cl::sycl::range<1>(mm_len)));
  cl::sycl::buffer<int, 1> mmc_index_b((cl::sycl::range<1>(mmc_cells+1)));
  cl::sycl::buffer<int, 1> mmc_i_b((cl::sycl::range<1>(mmc_cells)));
  cl::sycl::buffer<int, 1> mmc_j_b((cl::sycl::range<1>(mmc_cells)));
  cp_to_device(queue, rho_compact_b, ccc.rho_compact);
  cp_to_device(queue, rho_compact_list_b, ccc.rho_compact_list);
  cp_to_device(queue, rho_mat_ave_compact_b, ccc.rho_mat_ave_compact);
  cp_to_device(queue, rho_mat_ave_compact_list_b, ccc.rho_mat_ave_compact_list);
  cp_to_device(queue, p_compact_b, ccc.p_compact);
  cp_to_device(queue, p_compact_list_b, ccc.p_compact_list);
  cp_to_device(queue, Vf_compact_list_b, ccc.Vf_compact_list);
  cp_to_device(queue, t_compact_b, ccc.t_compact);
  cp_to_device(queue, t_compact_list_b, ccc.t_compact_list);
  cp_to_device(queue, V_b, ccc.V);
  cp_to_device(queue, x_b, ccc.x);
  cp_to_device(queue, y_b, ccc.y);
  cp_to_device(queue, n_b, ccc.n);
  cp_to_device(queue, rho_ave_compact_b, ccc.rho_ave_compact);
  cp_to_device(queue, imaterial_b, ccc.imaterial);
  cp_to_device(queue, matids_b, ccc.matids);
  cp_to_device(queue, nextfrac_b, ccc.nextfrac);
  cp_to_device(queue, mmc_index_b, ccc.mmc_index);
  cp_to_device(queue, mmc_i_b, ccc.mmc_i);
  cp_to_device(queue, mmc_j_b, ccc.mmc_j);
  queue.wait();
#endif
  if (Nmats < 1)
    printf("%d\n", Nmats);
  #if defined(ACC)
  #pragma acc data copy(imaterial[0:sizex*sizey],matids[0:mm_len], nextfrac[0:mm_len], x[0:sizex*sizey], y[0:sizex*sizey],n[0:Nmats], rho_compact[0:sizex*sizey], rho_compact_list[0:mm_len], rho_ave_compact[0:sizex*sizey], p_compact[0:sizex*sizey], p_compact_list[0:mm_len], t_compact[0:sizex*sizey], t_compact_list[0:mm_len], V[0:sizex*sizey], Vf_compact_list[0:mm_len], mmc_index[0:mmc_cells+1], mmc_i[0:mmc_cells], mmc_j[0:mmc_cells], rho_mat_ave_compact[0:sizex*sizey], rho_mat_ave_compact_list[0:mm_len])
  #endif
 #if defined(OMP4)
  #pragma omp target data map(tofrom: imaterial[0:sizex*sizey],matids[0:mm_len], nextfrac[0:mm_len], x[0:sizex*sizey], y[0:sizex*sizey],n[0:Nmats], rho_compact[0:sizex*sizey], rho_compact_list[0:mm_len], rho_ave_compact[0:sizex*sizey], p_compact[0:sizex*sizey], p_compact_list[0:mm_len], t_compact[0:sizex*sizey], t_compact_list[0:mm_len], V[0:sizex*sizey], Vf_compact_list[0:mm_len], mmc_index[0:mmc_cells+1], mmc_i[0:mmc_cells], mmc_j[0:mmc_cells], rho_mat_ave_compact[0:sizex*sizey], rho_mat_ave_compact_list[0:mm_len])
  #endif

  { 
	// Cell-centric algorithms
	// Computational loop 1 - average density in cell
  double t1 = omp_get_wtime();
#ifndef SYCL
  #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #elif defined(OMP4)
  #pragma omp target teams distribute parallel for collapse(2)
  #endif
	for (int j = 0; j < sizey; j++) {
  #if defined(OMP)
  #pragma omp simd
  #elif defined(ACC)
  #pragma acc loop independent
  #endif
		for (int i = 0; i < sizex; i++) {
#else //SYCL
      for (int t2 = 0; t2 < 2; t2++) {
        queue.wait();
        t1 = omp_get_wtime();
      queue.submit([&](cl::sycl::handler &cgh) {
          auto imaterial = imaterial_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_compact = rho_compact_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto V = V_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_ave_compact = rho_ave_compact_b.get_access<cl::sycl::access::mode::write>(cgh);
          #ifdef FUSED
          auto Vf_compact_list = Vf_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_compact_list = rho_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          #ifdef LINKED
          auto nextfrac = nextfrac_b.get_access<cl::sycl::access::mode::read>(cgh);
          #else
          auto mmc_index = mmc_index_b.get_access<cl::sycl::access::mode::read>(cgh);
          #endif
          #endif
          cgh.parallel_for<class alg1_1>(
              cl::sycl::range<1>(sizex*sizey), [=](cl::sycl::id<1> id) {
              int i = id.get(0)%sizex;
              int j = id.get(0)/sizex;
#endif //SYCL
      #ifdef FUSED
			double ave = 0.0;
			int ix = imaterial[i+sizex*j];
			if (ix <= 0) {
				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
      #ifdef LINKED
  #if defined(ACC)
  #pragma acc loop seq 
  #endif
      #pragma novector
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
					ave += rho_compact_list[ix] * Vf_compact_list[ix];
				}
      #else
  #if defined(ACC)
  #pragma acc loop seq 
  #endif
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
#ifndef SYCL
		}
	}
#else //SYCL
    });
  });
  }
  queue.wait();
  double t2 = omp_get_wtime();
  a1 = t2-t1;
  t1 = t2;
#endif //SYCL

#ifndef FUSED
#ifndef SYCL
  #if defined(OMP)
  #pragma omp parallel for simd
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #elif defined(OMP4)
  #pragma omp target teams distribute parallel for
  #endif
  for (int c = 0; c < mmc_cells; c++) {
#else //SYCL
      for (int t2 = 0; t2 < 2; t2++) {
        queue.wait();
        t1 = omp_get_wtime();
    queue.submit([&](cl::sycl::handler &cgh) {
          auto V = V_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto Vf_compact_list = Vf_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_compact_list = rho_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto mmc_index = mmc_index_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto mmc_i = mmc_i_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto mmc_j = mmc_j_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_ave_compact = rho_ave_compact_b.get_access<cl::sycl::access::mode::write>(cgh);
          cgh.parallel_for<class alg1_2>(
              cl::sycl::range<1>(mmc_cells), [=](cl::sycl::id<1> id) {
              int c = id.get(0);
#endif
    double ave = 0.0;
  #if defined(ACC)
  #pragma acc loop seq 
  #endif
    for (int m = mmc_index[c]; m < mmc_index[c+1]; m++) {
      ave +=  rho_compact_list[m] * Vf_compact_list[m];
    }
    rho_ave_compact[mmc_i[c]+sizex*mmc_j[c]] = ave/V[mmc_i[c]+sizex*mmc_j[c]];
#ifndef SYCL
  }
#else //SYCL
  });
  });
  }
#endif //SYCL
#endif
#ifdef SYCL
queue.wait();
#endif

  a1 += omp_get_wtime()-t1;
#ifdef DEBUG
  printf("Compact matrix, cell centric, alg 1: %g sec\n", a1);
#endif
	// Computational loop 2 - Pressure for each cell and each material
  t1 = omp_get_wtime();

#ifndef SYCL  
  #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #elif defined(OMP4)
  #pragma omp target teams distribute parallel for collapse(2)
  #endif
	for (int j = 0; j < sizey; j++) {
  #if defined(OMP)
  #pragma omp simd
  #elif defined(ACC)
  #pragma acc loop independent
  #endif
		for (int i = 0; i < sizex; i++) {
#else //SYCL
      for (int t2 = 0; t2 < 2; t2++) {
        queue.wait();
        t1 = omp_get_wtime();
      queue.submit([&](cl::sycl::handler &cgh) {
          auto imaterial = imaterial_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_compact = rho_compact_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto p_compact = p_compact_b.get_access<cl::sycl::access::mode::write>(cgh);
          auto n = n_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto t_compact = t_compact_b.get_access<cl::sycl::access::mode::read>(cgh);
          #ifdef FUSED
          auto matids = matids_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto Vf_compact_list = Vf_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_compact_list = rho_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto t_compact_list = t_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto p_compact_list = p_compact_list_b.get_access<cl::sycl::access::mode::write>(cgh);
          #ifdef LINKED
          auto nextfrac = nextfrac_b.get_access<cl::sycl::access::mode::read>(cgh);
          #else
          auto mmc_index = mmc_index_b.get_access<cl::sycl::access::mode::read>(cgh);
          #endif
          #endif
          cgh.parallel_for<class alg2_1>(
              cl::sycl::range<1>(sizex*sizey), [=](cl::sycl::id<1> id) {
              int i = id.get(0)%sizex;
              int j = id.get(0)/sizex;
#endif //SYCL

			int ix = imaterial[i+sizex*j];


#ifdef FUSED
			if (ix <= 0) {
				// NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
				// The solution below is what I believe to good.

				// condition is 'ix >= 0', this is the equivalent of
				// 'until ix < 0' from the paper
#ifdef LINKED
  #if defined(ACC)
  #pragma acc loop seq 
  #endif
				for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
					double nm = n[matids[ix]];
					p_compact_list[ix] = (nm * rho_compact_list[ix] * t_compact_list[ix]) / Vf_compact_list[ix];
				}
#else
  #if defined(ACC)
  #pragma acc loop seq 
  #endif
				for (int idx = mmc_index[-ix]; idx < mmc_index[-ix+1]; idx++) {
					double nm = n[matids[idx]];
					p_compact_list[idx] = (nm * rho_compact_list[idx] * t_compact_list[idx]) / Vf_compact_list[idx];
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
				p_compact[i+sizex*j] = n[mat] * rho_compact[i+sizex*j] * t_compact[i+sizex*j];
			}
#ifndef SYCL
		}
	}
#else //SYCL
    });
  });
    }
  queue.wait();
  t2 = omp_get_wtime();
  a2 = t2-t1;
  t1 = t2;
#endif //SYCL

#ifndef FUSED
#ifndef SYCL
  #if defined(OMP)
  #pragma omp parallel for simd
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent
  #elif defined(OMP4)
  #pragma omp target teams distribute parallel for
  #endif
  for (int idx = 0; idx < mmc_index[mmc_cells]; idx++) {
#else //SYCL
      for (int t2 = 0; t2 < 2; t2++) {
        queue.wait();
        t1 = omp_get_wtime();
    queue.submit([&](cl::sycl::handler &cgh) {
          auto matids = matids_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto Vf_compact_list = Vf_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_compact_list = rho_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto n = n_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto t_compact_list = t_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto p_compact_list = p_compact_list_b.get_access<cl::sycl::access::mode::write>(cgh);
          cgh.parallel_for<class alg2_2>(
              cl::sycl::range<1>(ccc.mmc_index[mmc_cells]), [=](cl::sycl::id<1> id) {
              int idx = id.get(0);
#endif

    double nm = n[matids[idx]];
    p_compact_list[idx] = (nm * rho_compact_list[idx] * t_compact_list[idx]) / Vf_compact_list[idx];
#ifndef SYCL
  }
#else //SYCL
  });
  });
  }
#endif //SYCL
#endif

#ifdef SYCL
queue.wait();
#endif
  a2 += omp_get_wtime()-t1;
#ifdef DEBUG
  printf("Compact matrix, cell centric, alg 2: %g sec\n", a2);
#endif

	// Computational loop 3 - Average density of each material over neighborhood of each cell
  t1 = omp_get_wtime();
#ifndef SYCL
  #if defined(OMP)
  #pragma omp parallel for //collapse(2)
  #elif defined(ACC)
  #pragma acc parallel
  #pragma acc loop independent collapse(2)
  #elif defined(OMP4)
  #pragma omp target teams distribute parallel for collapse(2)
  #endif
	for (int j = 1; j < sizey-1; j++) {
  #if defined(OMP)
  #pragma omp simd
//  #elif defined(ACC)
//  #pragma acc loop independent
  #endif
		for (int i = 1; i < sizex-1; i++) {
#else //SYCL
      for (int t2 = 0; t2 < 2; t2++) {
        queue.wait();
        t1 = omp_get_wtime();
      queue.submit([&](cl::sycl::handler &cgh) {
          auto imaterial = imaterial_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto x = x_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto y = y_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_compact = rho_compact_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_compact_list = rho_compact_list_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto matids = matids_b.get_access<cl::sycl::access::mode::read>(cgh);
          auto rho_mat_ave_compact = rho_mat_ave_compact_b.get_access<cl::sycl::access::mode::write>(cgh);
          auto rho_mat_ave_compact_list = rho_mat_ave_compact_list_b.get_access<cl::sycl::access::mode::write>(cgh);

          #ifdef LINKED
          auto nextfrac = nextfrac_b.get_access<cl::sycl::access::mode::read>(cgh);
          #else
          auto mmc_index = mmc_index_b.get_access<cl::sycl::access::mode::read>(cgh);
          #endif
          cgh.parallel_for<class alg3>(
              cl::sycl::range<1>((sizex-2)*(sizey-2)), [=](cl::sycl::id<1> id) {
              int i = id.get(0)%(sizex-2)+1;
              int j = id.get(0)/(sizex-2)+1;
#endif //SYCL

			// o: outer
			double xo = x[i+sizex*j];
			double yo = y[i+sizex*j];

			// There are at most 9 neighbours in 2D case.
			double dsqr[9];

			// for all neighbours
  #if defined(ACC)
  #pragma acc loop seq 
  #endif
			for (int nj = -1; nj <= 1; nj++) {
  #if defined(ACC)
  #pragma acc loop seq 
  #endif
				for (int ni = -1; ni <= 1; ni++) {

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
						for (int ni = -1; ni <= 1; ni++) {
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
#ifndef SYCL
            }
        }
#else //SYCL
            });
        });
      }
#endif //SYCL
#ifdef SYCL
queue.wait();
#endif
  a3 = omp_get_wtime()-t1;
#ifdef DEBUG
  printf("Compact matrix, cell centric, alg 3: %g sec\n", a3);
#endif
#ifdef SYCL
  cp_to_host(queue, rho_ave_compact_b, ccc.rho_ave_compact);
  cp_to_host(queue, rho_mat_ave_compact_b, ccc.rho_mat_ave_compact);
  cp_to_host(queue, rho_mat_ave_compact_list_b, ccc.rho_mat_ave_compact_list);
  cp_to_host(queue, p_compact_b, ccc.p_compact);
  cp_to_host(queue, p_compact_list_b, ccc.p_compact_list);
#endif
  }
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
