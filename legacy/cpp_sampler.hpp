#pragma once
#include "../common/matrix.hpp"
#include "../common/utils.hpp"
#define FP double

class cpp_sampler {
  public:
    cpp_sampler(std::string internal_timing_filename="");
    matrix<size_t> calc_grid_indices(size_t size_dim_0, size_t size_dim_1);
    matrix<FP> calc_rho (const matrix<FP>& bins, const matrix<FP>& xsec);

    matrix<FP> linear_interpolation(const matrix<FP>& u,const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights);
    matrix<FP> master_linear_interpolation(const matrix<FP>& u, const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights);
    matrix<FP> fusion_linear_interpolation(const matrix<FP>& u, const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights);
    matrix<FP> direct_linear_interpolation(const matrix<FP>& u, const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights);
    matrix<FP> dirion_linear_interpolation(const matrix<FP>& u, const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights);
    matrix<FP> noindx_linear_interpolation(const matrix<FP>& u, const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights);
    matrix<FP> calc_cdf(const matrix<FP>& obs_bins,
                        const matrix<FP>& obs_xsec,
                        const matrix<unsigned short>&   acceptance,
                        bool transpose);
    matrix<FP> calc_weight_matrix(const std::vector<size_t>& n, size_t max_n);
    matrix<FP> generate_single_observable(const matrix<FP>&  obs_bins,
                                          const matrix<FP>&  obs_xsec,
                                          const matrix<unsigned short>&    acceptance,
                                          const matrix<FP>&  weight_vector,
                                          const matrix<size_t>& grid_index,
                                          size_t              n_max,
                                          int                 obs_type_is_q2);
    matrix<FP> gen_events(const matrix<FP>&  x_bins,
                          const matrix<FP>&  x_sec_x,
                          const matrix<FP>&  q2_bins,
                          const matrix<FP>&  xsec_q2,
                          const matrix<unsigned short>& acceptance,
                          const matrix<FP>&  weight_vector,
                          const matrix<size_t>& grid_idx,
                          size_t max_n);
    matrix<FP> forward_single_sample(const matrix<FP>& x_bins,
                                     const matrix<FP>& xsec_x,
                                     const matrix<FP>& Q2_bins,
                                     const matrix<FP>& xsec_Q2,
                                     const matrix<unsigned short>& acceptance,
                                     const matrix<FP>&  weight_vector,
                                     const matrix<size_t>& grid_idx,
                                     size_t max_n);
    void start_recording();
    void delay_recording();
    void test_functions();

  private:
    matrix<FP> trapez(const matrix<FP>& my, FP dx, const matrix<FP>& mx);
    matrix<FP> cumulative_trapez(const matrix<FP>& my, const matrix<FP>& mx);
    matrix<FP> trapezoid(const matrix<FP>& y, float dx=1.0f, const matrix<FP>& x=matrix<FP>(), size_t dim=BY_ROW);
    void test_trapezoid_implementation();
    void test_matrix_implementation();
};
