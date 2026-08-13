#include "loits_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>

namespace openmp_loits {
namespace {

inline int16_t interval(const double* restrict cdf, int64_t k, double u) noexcept {
  int64_t selected = -1;
  for (int64_t t = 0; t < k; ++t) selected += static_cast<int64_t>(u >= cdf[t]);
  if (selected < 0) selected = 0;
  if (selected > k - 2) selected = k - 2;
  return static_cast<int16_t>(selected);
}

inline void decode_cell(int64_t global_cell,
                        const Shape& s,
                        int64_t& b,
                        int64_t& ix,
                        int64_t& iy) noexcept {
  b = global_cell / s.cells;
  const int64_t cell = global_cell - b * s.cells;
  ix = cell / s.ny;
  iy = cell - ix * s.ny;
}



}  // namespace

Allocation allocate_counts(const double* restrict weights,
                           int64_t* restrict counts,
                           int64_t n_events,
                           const Shape& s) noexcept {
  const int64_t n = s.batch * s.cells;
  int64_t nmax = 0;
  int64_t active = 0;
  const double scale = static_cast<double>(n_events);
#pragma omp parallel for schedule(static) reduction(max : nmax) reduction(+ : active)
  for (int64_t i = 0; i < n; ++i) {
    const int64_t count = static_cast<int64_t>(std::abs(weights[i] * scale));
    counts[i] = count;
    active += count;
    nmax = std::max(nmax, count);
  }
  return {nmax, active};
}

void fill_uniform(float* restrict values,
                  int64_t count,
                  uint64_t seed,
                  uint64_t stream) {
  constexpr int64_t block_size = 4096;
  const int64_t blocks = (count + block_size - 1) / block_size;
#pragma omp parallel for schedule(static)
  for (int64_t block = 0; block < blocks; ++block) {
    std::seed_seq seed_sequence{
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32),
        static_cast<uint32_t>(stream),
        static_cast<uint32_t>(stream >> 32),
        static_cast<uint32_t>(block),
        static_cast<uint32_t>(static_cast<uint64_t>(block) >> 32)};
    std::mt19937 generator(seed_sequence);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    const int64_t begin = block * block_size;
    const int64_t end = std::min(begin + block_size, count);
    for (int64_t i = begin; i < end; ++i) values[i] = distribution(generator);
  }
}

void density_x(const double* restrict bins,
               const double* restrict xsec,
               double* restrict norm,
               double* restrict rho,
               const Shape& s) noexcept {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t xsec_y_stride = s.nx * s.kx;
#pragma omp parallel for schedule(static)
  for (int64_t curve = 0; curve < curves; ++curve) {
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* restrict xb = bins + b * bins_batch_stride + ix * s.kx;
    const double* restrict xs = xsec + (b * s.ny + iy) * xsec_y_stride + ix * s.kx;
    double* restrict r = rho + (b * s.ny + iy) * xsec_y_stride + ix * s.kx;
    double n = 0.0;
    for (int64_t t = 1; t < s.kx; ++t) n += 0.5 * (xb[t] - xb[t - 1]) * (xs[t - 1] + xs[t]);
    norm[(b * s.ny + iy) * s.nx + ix] = n;
    const double inv_n = 1.0 / n;
    for (int64_t t = 0; t < s.kx; ++t) r[t] = xs[t] * inv_n;
  }
}

void density_q(const double* restrict bins,
               const double* restrict xsec,
               double* restrict norm,
               double* restrict rho,
               const Shape& s) noexcept {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t xsec_x_stride = s.ny * s.kq;
#pragma omp parallel for schedule(static)
  for (int64_t curve = 0; curve < curves; ++curve) {
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* restrict qb = bins + b * bins_batch_stride + iy * s.kq;
    const double* restrict xs = xsec + (b * s.nx + ix) * xsec_x_stride + iy * s.kq;
    double* restrict r = rho + (b * s.nx + ix) * xsec_x_stride + iy * s.kq;
    double n = 0.0;
    for (int64_t t = 1; t < s.kq; ++t) n += 0.5 * (qb[t] - qb[t - 1]) * (xs[t - 1] + xs[t]);
    norm[(b * s.nx + ix) * s.ny + iy] = n;
    const double inv_n = 1.0 / n;
    for (int64_t t = 0; t < s.kq; ++t) r[t] = xs[t] * inv_n;
  }
}

void cdf_x(const double* restrict bins,
           const double* restrict rho,
           const bool* restrict acceptance,
           double* restrict cdf,
           const Shape& s) noexcept {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t curve_y_stride = s.nx * s.kx;
#pragma omp parallel for schedule(static)
  for (int64_t curve = 0; curve < curves; ++curve) {
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* restrict xb = bins + b * bins_batch_stride + ix * s.kx;
    const double* restrict r = rho + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    double* restrict out = cdf + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    const double acc = acceptance[(b * s.nx + ix) * s.ny + iy] ? 1.0 : 0.0;
    double cumulative = 0.0;
    out[0] = 0.0;
    for (int64_t t = 1; t < s.kx; ++t) {
      cumulative += 0.5 * (xb[t] - xb[t - 1]) * (r[t - 1] + r[t]);
      out[t] = cumulative * acc;
    }
  }
}

void cdf_q(const double* restrict bins,
           const double* restrict rho,
           const bool* restrict acceptance,
           double* restrict cdf,
           const Shape& s) noexcept {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t curve_x_stride = s.ny * s.kq;
#pragma omp parallel for schedule(static)
  for (int64_t curve = 0; curve < curves; ++curve) {
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* restrict qb = bins + b * bins_batch_stride + iy * s.kq;
    const double* restrict r = rho + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    double* restrict out = cdf + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    const double acc = acceptance[(b * s.nx + ix) * s.ny + iy] ? 1.0 : 0.0;
    double cumulative = 0.0;
    out[0] = 0.0;
    for (int64_t t = 1; t < s.kq; ++t) {
      cumulative += 0.5 * (qb[t] - qb[t - 1]) * (r[t - 1] + r[t]);
      out[t] = cumulative * acc;
    }
  }
}

void interpolate_x(const double* restrict bins,
                   const double* restrict cdf,
                   const float* restrict u,
                   const int64_t* restrict counts,
                   int64_t nmax,
                   double* restrict dense,
                   int16_t* restrict indices,
                   const Shape& s) noexcept {
  const int64_t global_cells = s.batch * s.cells;
  const int64_t slots_per_batch = s.cells * nmax;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t cdf_batch_stride = s.ny * s.nx * s.kx;
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t global_cell = 0; global_cell < global_cells; ++global_cell) {
    int64_t b, ix, iy;
    decode_cell(global_cell, s, b, ix, iy);
    const int64_t cell = ix * s.ny + iy;
    const int64_t count = counts[global_cell];
    const double* restrict bin = bins + b * bins_batch_stride + ix * s.kx;
    const double* restrict curve = cdf + b * cdf_batch_stride + (iy * s.nx + ix) * s.kx;
    const int64_t base = b * slots_per_batch + cell * nmax;
    for (int64_t slot = 0; slot < count; ++slot) {
      const int64_t p = base + slot;
      const double uv = static_cast<double>(u[p]);
      const int16_t j = interval(curve, s.kx, uv);
      indices[p] = j;
      const double c0 = curve[j];
      const double inv_d = 1.0 / (curve[j + 1] - c0 + kEpsilon);
      const double m = (bin[j + 1] - bin[j]) * inv_d;
      dense[p] = bin[j] + m * (uv - c0);
    }
  }
}

void interpolate_q(const double* restrict bins,
                   const double* restrict cdf,
                   const float* restrict u,
                   const int64_t* restrict counts,
                   int64_t nmax,
                   double* restrict dense,
                   int16_t* restrict indices,
                   const Shape& s) noexcept {
  const int64_t global_cells = s.batch * s.cells;
  const int64_t slots_per_batch = s.cells * nmax;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t cdf_batch_stride = s.nx * s.ny * s.kq;
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t global_cell = 0; global_cell < global_cells; ++global_cell) {
    int64_t b, ix, iy;
    decode_cell(global_cell, s, b, ix, iy);
    const int64_t cell = ix * s.ny + iy;
    const int64_t count = counts[global_cell];
    const double* restrict curve = cdf + b * cdf_batch_stride + (ix * s.ny + iy) * s.kq;
    const double* restrict bin = bins + b * bins_batch_stride + iy * s.kq;
    const int64_t base = b * slots_per_batch + cell * nmax;
    for (int64_t slot = 0; slot < count; ++slot) {
      const int64_t p = base + slot;
      const double uv = static_cast<double>(u[p]);
      const int16_t j = interval(curve, s.kq, uv);
      indices[p] = j;
      const double c0 = curve[j];
      const double inv_d = 1.0 / (curve[j + 1] - c0 + kEpsilon);
      const double m = (bin[j + 1] - bin[j]) * inv_d;
      dense[p] = bin[j] + m * (uv - c0);
    }
  }
}

int64_t compact(const double* restrict dense_x,
                const double* restrict dense_q,
                const int64_t* restrict counts,
                int64_t nmax,
                double* restrict events,
                int64_t* restrict packed,
                int64_t* restrict row_offsets,
                const Shape& s) noexcept {
  const int64_t global_cells = s.batch * s.cells;
  const int64_t slots_per_batch = s.cells * nmax;
  row_offsets[0] = 0;
#pragma omp parallel for schedule(static)
  for (int64_t global_cell = 0; global_cell < global_cells; ++global_cell) {
    const int64_t b = global_cell / s.cells;
    const int64_t cell = global_cell - b * s.cells;
    const int64_t base = b * slots_per_batch + cell * nmax;
    int64_t valid = 0;
    for (int64_t slot = 0; slot < counts[global_cell]; ++slot) {
      const int64_t p = base + slot;
      const double xv = dense_x[p];
      const double qv = dense_q[p];
      valid += static_cast<int64_t>(std::isfinite(xv) && std::isfinite(qv) && xv * qv != 0.0);
    }
    row_offsets[global_cell + 1] = valid;
  }

  for (int64_t global_cell = 0; global_cell < global_cells; ++global_cell) {
    row_offsets[global_cell + 1] += row_offsets[global_cell];
  }

#pragma omp parallel for schedule(static)
  for (int64_t global_cell = 0; global_cell < global_cells; ++global_cell) {
    const int64_t b = global_cell / s.cells;
    const int64_t cell = global_cell - b * s.cells;
    const int64_t base = b * slots_per_batch + cell * nmax;
    int64_t row = row_offsets[global_cell];
    for (int64_t slot = 0; slot < counts[global_cell]; ++slot) {
      const int64_t p = base + slot;
      const double xv = dense_x[p];
      const double qv = dense_q[p];
      if (std::isfinite(xv) && std::isfinite(qv) && xv * qv != 0.0) {
        events[row * 2] = xv;
        events[row * 2 + 1] = qv;
        packed[row] = p;
        ++row;
      }
    }
  }
  return row_offsets[global_cells];
}

void interpolation_vjp_x(const double* restrict grad_events,
                         const int64_t* restrict packed,
                         const int64_t* restrict row_offsets,
                         const double* restrict bins,
                         const double* restrict cdf,
                         const float* restrict u,
                         const int16_t* restrict indices,
                         int64_t nmax,
                         double* restrict grad_cdf,
                         const Shape& s) noexcept {
  const int64_t global_cells = s.batch * s.cells;
  const int64_t slots_per_batch = s.cells * nmax;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t cdf_batch_stride = s.ny * s.nx * s.kx;
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t global_cell = 0; global_cell < global_cells; ++global_cell) {
    int64_t b, ix, iy;
    decode_cell(global_cell, s, b, ix, iy);
    const double* restrict curve = cdf + b * cdf_batch_stride + (iy * s.nx + ix) * s.kx;
    const double* restrict bin = bins + b * bins_batch_stride + ix * s.kx;
    double* restrict gcurve = grad_cdf + b * cdf_batch_stride + (iy * s.nx + ix) * s.kx;
    std::fill(gcurve, gcurve + s.kx, 0.0);
    for (int64_t row = row_offsets[global_cell]; row < row_offsets[global_cell + 1]; ++row) {
      const int64_t p = packed[row];
      const int64_t j = indices[p];
      const double c0 = curve[j];
      const double inv_d = 1.0 / (curve[j + 1] - c0 + kEpsilon);
      const double t = static_cast<double>(u[p]) - c0;
      const double upstream = grad_events[row * 2] * (bin[j + 1] - bin[j]);
      gcurve[j] += upstream * (-inv_d + t * inv_d * inv_d);
      gcurve[j + 1] -= upstream * t * inv_d * inv_d;
    }
  }
}

void interpolation_vjp_q(const double* restrict grad_events,
                         const int64_t* restrict packed,
                         const int64_t* restrict row_offsets,
                         const double* restrict bins,
                         const double* restrict cdf,
                         const float* restrict u,
                         const int16_t* restrict indices,
                         int64_t nmax,
                         double* restrict grad_cdf,
                         const Shape& s) noexcept {
  const int64_t global_cells = s.batch * s.cells;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t cdf_batch_stride = s.nx * s.ny * s.kq;
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t global_cell = 0; global_cell < global_cells; ++global_cell) {
    int64_t b, ix, iy;
    decode_cell(global_cell, s, b, ix, iy);
    const double* restrict curve = cdf + b * cdf_batch_stride + (ix * s.ny + iy) * s.kq;
    const double* restrict bin = bins + b * bins_batch_stride + iy * s.kq;
    double* restrict gcurve = grad_cdf + b * cdf_batch_stride + (ix * s.ny + iy) * s.kq;
    std::fill(gcurve, gcurve + s.kq, 0.0);
    for (int64_t row = row_offsets[global_cell]; row < row_offsets[global_cell + 1]; ++row) {
      const int64_t p = packed[row];
      const int64_t j = indices[p];
      const double c0 = curve[j];
      const double inv_d = 1.0 / (curve[j + 1] - c0 + kEpsilon);
      const double t = static_cast<double>(u[p]) - c0;
      const double upstream = grad_events[row * 2 + 1] * (bin[j + 1] - bin[j]);
      gcurve[j] += upstream * (-inv_d + t * inv_d * inv_d);
      gcurve[j + 1] -= upstream * t * inv_d * inv_d;
    }
  }
}

void cdf_vjp_x(const double* restrict bins,
               const bool* restrict acceptance,
               const double* restrict grad_cdf,
               double* restrict grad_rho,
               const Shape& s) noexcept {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t curve_y_stride = s.nx * s.kx;
#pragma omp parallel for schedule(static)
  for (int64_t curve = 0; curve < curves; ++curve) {
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* restrict xb = bins + b * bins_batch_stride + ix * s.kx;
    const double* restrict gc = grad_cdf + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    double* restrict gr = grad_rho + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    std::fill(gr, gr + s.kx, 0.0);
    const double acc = acceptance[(b * s.nx + ix) * s.ny + iy] ? 1.0 : 0.0;
    double suffix = 0.0;
    for (int64_t t = s.kx - 1; t > 0; --t) {
      suffix += gc[t];
      const double g = 0.5 * (xb[t] - xb[t - 1]) * acc * suffix;
      gr[t - 1] += g;
      gr[t] += g;
    }
  }
}

void cdf_vjp_q(const double* restrict bins,
               const bool* restrict acceptance,
               const double* restrict grad_cdf,
               double* restrict grad_rho,
               const Shape& s) noexcept {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t curve_x_stride = s.ny * s.kq;
#pragma omp parallel for schedule(static)
  for (int64_t curve = 0; curve < curves; ++curve) {
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* restrict qb = bins + b * bins_batch_stride + iy * s.kq;
    const double* restrict gc = grad_cdf + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    double* restrict gr = grad_rho + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    std::fill(gr, gr + s.kq, 0.0);
    const double acc = acceptance[(b * s.nx + ix) * s.ny + iy] ? 1.0 : 0.0;
    double suffix = 0.0;
    for (int64_t t = s.kq - 1; t > 0; --t) {
      suffix += gc[t];
      const double g = 0.5 * (qb[t] - qb[t - 1]) * acc * suffix;
      gr[t - 1] += g;
      gr[t] += g;
    }
  }
}

void density_vjp_x(const double* restrict bins,
                   const double* restrict xsec,
                   const double* restrict norm,
                   const double* restrict grad_rho,
                   double* restrict grad_xsec,
                   const Shape& s) noexcept {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t curve_y_stride = s.nx * s.kx;
#pragma omp parallel for schedule(static)
  for (int64_t curve = 0; curve < curves; ++curve) {
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* restrict xb = bins + b * bins_batch_stride + ix * s.kx;
    const double* restrict xs = xsec + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    const double* restrict gr = grad_rho + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    double* restrict gx = grad_xsec + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    const double n = norm[(b * s.ny + iy) * s.nx + ix];
    const double inv_n = 1.0 / n;
    const double inv_n2 = inv_n * inv_n;
    double grad_norm = 0.0;
    for (int64_t t = 0; t < s.kx; ++t) {
      gx[t] = gr[t] * inv_n;
      grad_norm -= gr[t] * xs[t] * inv_n2;
    }
    for (int64_t t = 1; t < s.kx; ++t) {
      const double g = 0.5 * (xb[t] - xb[t - 1]) * grad_norm;
      gx[t - 1] += g;
      gx[t] += g;
    }
  }
}

void density_vjp_q(const double* restrict bins,
                   const double* restrict xsec,
                   const double* restrict norm,
                   const double* restrict grad_rho,
                   double* restrict grad_xsec,
                   const Shape& s) noexcept {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t curve_x_stride = s.ny * s.kq;
#pragma omp parallel for schedule(static)
  for (int64_t curve = 0; curve < curves; ++curve) {
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* restrict qb = bins + b * bins_batch_stride + iy * s.kq;
    const double* restrict xs = xsec + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    const double* restrict gr = grad_rho + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    double* restrict gx = grad_xsec + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    const double n = norm[(b * s.nx + ix) * s.ny + iy];
    const double inv_n = 1.0 / n;
    const double inv_n2 = inv_n * inv_n;
    double grad_norm = 0.0;
    for (int64_t t = 0; t < s.kq; ++t) {
      gx[t] = gr[t] * inv_n;
      grad_norm -= gr[t] * xs[t] * inv_n2;
    }
    for (int64_t t = 1; t < s.kq; ++t) {
      const double g = 0.5 * (qb[t] - qb[t - 1]) * grad_norm;
      gx[t - 1] += g;
      gx[t] += g;
    }
  }
}

}  // namespace openmp_loits
