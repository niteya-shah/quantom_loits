#include <csignal>

#include <algorithm>
#include <cassert>
#include <omp.h>
#include "omp_sampler.hpp"
#include "../legacy/cpp_sampler.hpp"
#include "../common/test_inputs.hpp"

int main(int argc, char*argv[]){
  int omp_threads = 1;
  if (argc == 2) {
    omp_threads = atoi(argv[1]);
    omp_set_num_threads(omp_threads);
  }
  cpp_sampler* ref_smplr = new cpp_sampler();
  omp_sampler* smplr = new omp_sampler();

  //register regions of interest to be logged:
  assign_filename_for_recording("omp_"+std::to_string(omp_threads)+"_threads.csv");
  mark_for_recording("C++ Master Linear Interpolation");
  mark_for_recording("C++ Fusion Linear Interpolation");
  mark_for_recording("C++ Direct Linear Interpolation");
  mark_for_recording("C++ Dirion Linear Interpolation");
  mark_for_recording("C++ Noindx Linear Interpolation");
  mark_for_recording("OpenMP Master Linear Interpolation");
  mark_for_recording("OpenMP Noindx Linear Interpolation");
  
  tic("ref_calc_grid_indices");
  matrix<size_t> slow_grid_idx = ref_smplr->calc_grid_indices(19,19);
  toc("calc_grid_indices");

  tic("fast_calc_grid_indices");
  matrix<size_t> grid_idx = smplr->calc_grid_indices(19,19);
  toc("fast_calc_grid_indices");
  assert(are_equal(slow_grid_idx, grid_idx) && "calc_grid_indices");
  //auto diff = max_diff(slow_grid_idx, grid_idx);
  //std::cout << "max difference between (calc_grid_indices) implementations = " << diff << std::endl;

  // get max_n
  /*
  //TODO: delete when good!
  tic("max_n");
  std::vector<size_t> n(weights.size());
  matrix<FP> weights_cpy = weights;
  std::transform(weights_cpy.begin(), weights_cpy.end(), n.begin(),  [](FP element){return static_cast<size_t>(std::floor(element*number_of_events));});
  auto max_n_ptr = std::max_element(n.begin(),n.end());
  size_t max_n = *max_n_ptr;
  toc("max_n");
  */
  tic("max_n");
  size_t max_n = 0;
  std::vector<size_t> n(weights.size());
  const std::vector<FP> val = weights.as_vector();
  for(size_t i = 0; i < weights.size(); i++){
    size_t tmp = (size_t)std::floor(val[i]*number_of_events);
    n[i] = tmp;
    if (tmp>max_n){
      max_n = tmp;
    }
  } 
  toc("max_n");

  tic("ref_calc_weight_matrix");
  matrix<FP> ref_weights_matrix = ref_smplr->calc_weight_matrix(n,max_n);
  toc("ref_calc_weight_matrix");
  tic("calc_weight_matrix");
  matrix<FP> weights_matrix = smplr->calc_weight_matrix(n,max_n);
  toc("calc_weight_matrix");
  assert(are_equal(ref_weights_matrix, weights_matrix) && "calc_weight_matrix");

  //--------------------------------------
  //The deepest functions
  //--------------------------------------
  tic("ref_calc_cdf");
  auto ref_cdf = ref_smplr->calc_cdf(x_bins,x_sec_x,acceptance,false);
  toc("ref_calc_cdf");
  tic("calc_cdf");
  auto cdf = smplr->calc_cdf(x_bins,x_sec_x,acceptance,false);
  toc("calc_cdf");
  assert(are_equal(ref_cdf, cdf) && "calc_cdf");
/*
  diff = max_diff(ref_cdf,cdf);
  std::cout << "max difference between (calc_cdf) implementations = " << diff << std::endl;
  */

  tic("u_obs_fill");
  matrix<FP> u_obs(grid_idx.rows(),max_n);
  u_obs.fill(0.5);
  toc("u_obs_fill");

  tic("ref_cdf_flattern");
  matrix<FP> ref_cdf_obs_flat(grid_idx.rows(),cdf.pages());
  for (size_t r = 0; r < grid_idx.rows(); r++){
    size_t x = grid_idx(r,0); size_t y = grid_idx(r,1);
    for (size_t p = 0; p < cdf.pages(); p++){
      ref_cdf_obs_flat(r,p) = cdf(x,y,p);
    }
  }
  toc("ref_cdf_flattern");
  tic("cdf_flattern");
  matrix<FP> cdf_obs_flat(grid_idx.rows(),cdf.pages());
  {
  FP* wo = cdf_obs_flat.ptr(); size_t wo_cols = cdf_obs_flat.cols(), wo_page = cdf_obs_flat.pages();
  const long unsigned int * gi = grid_idx.ptr(); size_t gi_rows = grid_idx.rows(), gi_cols = grid_idx.cols(), gi_page = grid_idx.pages();
  const FP * co = cdf.ptr(); size_t co_cols = cdf.cols(), co_page = cdf.pages();
  #pragma omp parallel for
  for (size_t r = 0; r < gi_rows; r++){
    //size_t x = grid_index(r,settings.x);
    size_t x = gi[r*gi_cols*gi_page + 0];
    //size_t y = grid_index(r,settings.y);
    size_t y = gi[r*gi_cols*gi_page + 1];
    for (size_t p = 0; p < co_page; p++){
      //cdf_obs_flat_idx(r,p) = cdf_obs(x,y,p);
      wo[r*wo_cols*wo_page + p] = co[x*co_cols*co_page + p*co_cols + y];
    }
  }
  }
  toc("cdf_flattern");
  assert(are_equal(ref_cdf_obs_flat,cdf_obs_flat));

  tic("bin_flattern");
  matrix<FP> bin_obs_flat(grid_idx.rows(),x_bins.cols());
  #pragma omp parallel for
  for (size_t r = 0; r < grid_idx.rows(); r++){
    size_t z = grid_idx(r,1);
    for (size_t c = 0; c < x_bins.cols(); c++){
      bin_obs_flat(r,c) = x_bins(z,c);
    }
  }
  toc("bin_flattern");

//#if defined(INSTRUMENTATION)// || defined(TAU)
  tic("C++ Master Linear Interpolation");
  auto ref_obs_gen = ref_smplr->master_linear_interpolation(u_obs, cdf_obs_flat, bin_obs_flat, weights_matrix);
  toc("C++ Master Linear Interpolation");
  tic("C++ Fusion Linear Interpolation");
  auto ref_fusion_obs_gen = ref_smplr->fusion_linear_interpolation(u_obs, cdf_obs_flat, bin_obs_flat, weights_matrix);
  toc("C++ Fusion Linear Interpolation");
  assert(are_equal(ref_obs_gen, ref_fusion_obs_gen) && "fusion_linear_interpolation");
  tic("C++ Direct Linear Interpolation");
  auto ref_direct_obs_gen = ref_smplr->direct_linear_interpolation(u_obs, cdf_obs_flat, bin_obs_flat, weights_matrix);
  toc("C++ Direct Linear Interpolation");
  assert(are_equal(ref_obs_gen, ref_direct_obs_gen) && "direct_linear_interpolation");
  tic("C++ Dirion Linear Interpolation");
  auto ref_dirion_obs_gen = ref_smplr->dirion_linear_interpolation(u_obs, cdf_obs_flat, bin_obs_flat, weights_matrix);
  toc("C++ Dirion Linear Interpolation");
  assert(are_equal(ref_obs_gen, ref_dirion_obs_gen) && "dirion_linear_interpolation");
  tic("C++ Noindx Linear Interpolation");
  auto ref_noindx_obs_gen = ref_smplr->noindx_linear_interpolation(u_obs, cdf_obs_flat, bin_obs_flat, weights_matrix);
  toc("C++ Noindx Linear Interpolation");
  assert(are_equal(ref_obs_gen, ref_noindx_obs_gen) && "noindx_linear_interpolation");
  tic("OpenMP Master Linear Interpolation");
  auto omp_obs_mstr_gen = smplr->master_linear_interpolation(u_obs, cdf_obs_flat, bin_obs_flat, weights_matrix);
  toc("OpenMP Master Linear Interpolation");
  assert(are_equal(ref_obs_gen, omp_obs_mstr_gen) && "omp master_linear_interpolation");
  tic("OpenMP Noindx Linear Interpolation");
  auto omp_obs_noindx_gen = smplr->noindx_linear_interpolation(u_obs, cdf_obs_flat, bin_obs_flat, weights_matrix);
  toc("OpenMP Noindx Linear Interpolation");
  assert(are_equal(ref_obs_gen, omp_obs_noindx_gen) && "noindx linear_interpolation");

//#endif

  //--------------------------------------
  //And back up the stack
  //--------------------------------------
  tic("ref_generate_single_observable_0");
  auto ref_gso = ref_smplr->generate_single_observable(x_bins,x_sec_x,acceptance,weights_matrix,grid_idx,max_n,0);
  toc("ref_generate_single_observable_0");
  tic("generate_single_observable_0");
  auto gso = smplr->generate_single_observable(x_bins,x_sec_x,acceptance,weights_matrix,grid_idx,max_n,0);
  toc("generate_single_observable_0");
  assert(are_equal(ref_gso, gso) && "generate_single_observable_0");
  /*
  diff = max_diff(ref_gso,gso);
  std::cout << "max difference between (generate_single_observable_0) implementations = " << diff << std::endl;
  */

  tic("ref_generate_single_observable_1");
  ref_gso = ref_smplr->generate_single_observable(q2_bins,x_sec_q2,acceptance,weights_matrix,grid_idx,max_n,1);
  toc("ref_generate_single_observable_1");
  tic("generate_single_observable_1");
  gso = smplr->generate_single_observable(q2_bins,x_sec_q2,acceptance,weights_matrix,grid_idx,max_n,1);
  toc("generate_single_observable_1");
  assert(are_equal(ref_gso, gso) && "generate_single_observable_1");
  /*
  diff = max_diff(ref_gso,gso);
  std::cout << "max difference between (generate_single_observable_1) implementations = " << diff << std::endl;
  */

  //gen_events
  tic("ref_gen_events");
  auto ref_gen_events_out = ref_smplr->gen_events(x_bins,
                                                  x_sec_x,
                                                  q2_bins,
                                                  x_sec_q2,
                                                  acceptance,
                                                  weights_matrix,
                                                  grid_idx,
                                                  max_n);
  toc("ref_gen_events");
  tic("gen_events");
  auto gen_events_out = smplr->gen_events(x_bins,
                                          x_sec_x,
                                          q2_bins,
                                          x_sec_q2,
                                          acceptance,
                                          weights_matrix,
                                          grid_idx,
                                          max_n);
  toc("gen_events");
  assert(are_equal(ref_gen_events_out, gen_events_out) && "gen_events");
  /*
  diff = max_diff(ref_gen_events_out,gen_events_out);
  std::cout << "max difference between (gen_events) implementations = " << diff << std::endl;
  */

  tic("ref_forward_single_sample");
  auto ref_res = ref_smplr->forward_single_sample(x_bins,x_sec_x,q2_bins,x_sec_q2,acceptance,weights_matrix,grid_idx,max_n);
  toc("ref_forward_single_sample");
  tic("forward_single_sample");
  auto res = smplr->forward_single_sample(x_bins,x_sec_x,q2_bins,x_sec_q2,acceptance,weights_matrix,grid_idx,max_n);
  toc("forward_single_sample");
  assert(are_equal(ref_res, res) && "forward_single_sample");
  print_green("all good---that region is fine\n");
  return EXIT_SUCCESS;
}

