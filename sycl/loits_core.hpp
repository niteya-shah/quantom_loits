#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#define QUANTOM_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define QUANTOM_RESTRICT __restrict__
#else
#define QUANTOM_RESTRICT
#endif

namespace sycl_loits {

constexpr double kEpsilon = 1e-5;

struct Shape {
  int64_t batch;
  int64_t nx;
  int64_t ny;
  int64_t kx;
  int64_t kq;
  int64_t cells;
};

struct Allocation {
  int64_t nmax;
  int64_t active;
};

Allocation allocate_counts(const double* QUANTOM_RESTRICT weights,
                           int64_t* QUANTOM_RESTRICT counts,
                           int64_t n_events,
                           const Shape& shape);

void fill_uniform(float* QUANTOM_RESTRICT values,
                  int64_t count,
                  uint64_t seed,
                  uint64_t stream);

void density_x(const double* QUANTOM_RESTRICT bins,
               const double* QUANTOM_RESTRICT xsec,
               double* QUANTOM_RESTRICT norm,
               double* QUANTOM_RESTRICT rho,
               const Shape& shape);

void density_q(const double* QUANTOM_RESTRICT bins,
               const double* QUANTOM_RESTRICT xsec,
               double* QUANTOM_RESTRICT norm,
               double* QUANTOM_RESTRICT rho,
               const Shape& shape);

void cdf_x(const double* QUANTOM_RESTRICT bins,
           const double* QUANTOM_RESTRICT rho,
           const bool* QUANTOM_RESTRICT acceptance,
           double* QUANTOM_RESTRICT cdf,
           const Shape& shape);

void cdf_q(const double* QUANTOM_RESTRICT bins,
           const double* QUANTOM_RESTRICT rho,
           const bool* QUANTOM_RESTRICT acceptance,
           double* QUANTOM_RESTRICT cdf,
           const Shape& shape);

void interpolate_x(const double* QUANTOM_RESTRICT bins,
                   const double* QUANTOM_RESTRICT cdf,
                   const float* QUANTOM_RESTRICT u,
                   const int64_t* QUANTOM_RESTRICT counts,
                   int64_t nmax,
                   double* QUANTOM_RESTRICT dense,
                   int16_t* QUANTOM_RESTRICT indices,
                   const Shape& shape);

void interpolate_q(const double* QUANTOM_RESTRICT bins,
                   const double* QUANTOM_RESTRICT cdf,
                   const float* QUANTOM_RESTRICT u,
                   const int64_t* QUANTOM_RESTRICT counts,
                   int64_t nmax,
                   double* QUANTOM_RESTRICT dense,
                   int16_t* QUANTOM_RESTRICT indices,
                   const Shape& shape);

int64_t compact(const double* QUANTOM_RESTRICT dense_x,
                const double* QUANTOM_RESTRICT dense_q,
                const int64_t* QUANTOM_RESTRICT counts,
                int64_t nmax,
                double* QUANTOM_RESTRICT events,
                int64_t* QUANTOM_RESTRICT packed,
                int64_t* QUANTOM_RESTRICT row_offsets,
                const Shape& shape);

void interpolation_vjp_x(const double* QUANTOM_RESTRICT grad_events,
                         const int64_t* QUANTOM_RESTRICT packed,
                         const int64_t* QUANTOM_RESTRICT row_offsets,
                         const double* QUANTOM_RESTRICT bins,
                         const double* QUANTOM_RESTRICT cdf,
                         const float* QUANTOM_RESTRICT u,
                         const int16_t* QUANTOM_RESTRICT indices,
                         int64_t nmax,
                         double* QUANTOM_RESTRICT grad_cdf,
                         const Shape& shape);

void interpolation_vjp_q(const double* QUANTOM_RESTRICT grad_events,
                         const int64_t* QUANTOM_RESTRICT packed,
                         const int64_t* QUANTOM_RESTRICT row_offsets,
                         const double* QUANTOM_RESTRICT bins,
                         const double* QUANTOM_RESTRICT cdf,
                         const float* QUANTOM_RESTRICT u,
                         const int16_t* QUANTOM_RESTRICT indices,
                         int64_t nmax,
                         double* QUANTOM_RESTRICT grad_cdf,
                         const Shape& shape);

void cdf_vjp_x(const double* QUANTOM_RESTRICT bins,
               const bool* QUANTOM_RESTRICT acceptance,
               const double* QUANTOM_RESTRICT grad_cdf,
               double* QUANTOM_RESTRICT grad_rho,
               const Shape& shape);

void cdf_vjp_q(const double* QUANTOM_RESTRICT bins,
               const bool* QUANTOM_RESTRICT acceptance,
               const double* QUANTOM_RESTRICT grad_cdf,
               double* QUANTOM_RESTRICT grad_rho,
               const Shape& shape);

void density_vjp_x(const double* QUANTOM_RESTRICT bins,
                   const double* QUANTOM_RESTRICT xsec,
                   const double* QUANTOM_RESTRICT norm,
                   const double* QUANTOM_RESTRICT grad_rho,
                   double* QUANTOM_RESTRICT grad_xsec,
                   const Shape& shape);

void density_vjp_q(const double* QUANTOM_RESTRICT bins,
                   const double* QUANTOM_RESTRICT xsec,
                   const double* QUANTOM_RESTRICT norm,
                   const double* QUANTOM_RESTRICT grad_rho,
                   double* QUANTOM_RESTRICT grad_xsec,
                   const Shape& shape);

}  // namespace sycl_loits
