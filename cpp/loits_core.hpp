#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#define restrict __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define restrict __restrict__
#else
#define restrict
#endif

namespace loits {

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

Allocation allocate_counts(const double* restrict weights,
                           int64_t* restrict counts,
                           int64_t n_events,
                           const Shape& shape) noexcept;

void fill_uniform(float* restrict values,
                  int64_t count,
                  uint64_t seed,
                  uint64_t stream);

void density_x(const double* restrict bins,
               const double* restrict xsec,
               double* restrict norm,
               double* restrict rho,
               const Shape& shape) noexcept;

void density_q(const double* restrict bins,
               const double* restrict xsec,
               double* restrict norm,
               double* restrict rho,
               const Shape& shape) noexcept;

void cdf_x(const double* restrict bins,
           const double* restrict rho,
           const bool* restrict acceptance,
           double* restrict cdf,
           const Shape& shape) noexcept;

void cdf_q(const double* restrict bins,
           const double* restrict rho,
           const bool* restrict acceptance,
           double* restrict cdf,
           const Shape& shape) noexcept;

void interpolate_x(const double* restrict bins,
                   const double* restrict cdf,
                   const float* restrict u,
                   const int64_t* restrict counts,
                   int64_t nmax,
                   double* restrict dense,
                   int16_t* restrict indices,
                   const Shape& shape) noexcept;

void interpolate_q(const double* restrict bins,
                   const double* restrict cdf,
                   const float* restrict u,
                   const int64_t* restrict counts,
                   int64_t nmax,
                   double* restrict dense,
                   int16_t* restrict indices,
                   const Shape& shape) noexcept;

int64_t compact(const double* restrict dense_x,
                const double* restrict dense_q,
                const int64_t* restrict counts,
                int64_t nmax,
                double* restrict events,
                int64_t* restrict packed,
                const Shape& shape) noexcept;

void interpolation_vjp_x(const double* restrict grad_events,
                         const int64_t* restrict packed,
                         int64_t nvalid,
                         const double* restrict bins,
                         const double* restrict cdf,
                         const float* restrict u,
                         const int16_t* restrict indices,
                         int64_t nmax,
                         double* restrict grad_cdf,
                         const Shape& shape) noexcept;

void interpolation_vjp_q(const double* restrict grad_events,
                         const int64_t* restrict packed,
                         int64_t nvalid,
                         const double* restrict bins,
                         const double* restrict cdf,
                         const float* restrict u,
                         const int16_t* restrict indices,
                         int64_t nmax,
                         double* restrict grad_cdf,
                         const Shape& shape) noexcept;

void cdf_vjp_x(const double* restrict bins,
               const bool* restrict acceptance,
               const double* restrict grad_cdf,
               double* restrict grad_rho,
               const Shape& shape) noexcept;

void cdf_vjp_q(const double* restrict bins,
               const bool* restrict acceptance,
               const double* restrict grad_cdf,
               double* restrict grad_rho,
               const Shape& shape) noexcept;

void density_vjp_x(const double* restrict bins,
                   const double* restrict xsec,
                   const double* restrict norm,
                   const double* restrict grad_rho,
                   double* restrict grad_xsec,
                   const Shape& shape) noexcept;

void density_vjp_q(const double* restrict bins,
                   const double* restrict xsec,
                   const double* restrict norm,
                   const double* restrict grad_rho,
                   double* restrict grad_xsec,
                   const Shape& shape) noexcept;

}  // namespace loits
