#include "loits_core.hpp"

#include "../rng/philox.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace sycl_loits {

sycl::queue& queue();

namespace {

inline int16_t interval(const double* QUANTOM_RESTRICT cdf,
                        int64_t k,
                        double u) {
  int64_t selected = -1;
  for (int64_t t = 0; t < k; ++t) {
    selected += static_cast<int64_t>(u >= cdf[t]);
  }
  if (selected < 0) selected = 0;
  if (selected > k - 2) selected = k - 2;
  return static_cast<int16_t>(selected);
}

inline void decode_cell(int64_t global_cell,
                        const Shape& s,
                        int64_t& b,
                        int64_t& ix,
                        int64_t& iy) {
  b = global_cell / s.cells;
  const int64_t cell = global_cell - b * s.cells;
  ix = cell / s.ny;
  iy = cell - ix * s.ny;
}

}  // namespace

Allocation allocate_counts(const double* QUANTOM_RESTRICT weights,
                           int64_t* QUANTOM_RESTRICT counts,
                           int64_t n_events,
                           const Shape& s) {
  const int64_t n = s.batch * s.cells;

  auto& q = queue();
  int64_t* result = sycl::malloc_shared<int64_t>(2, q);
  result[0] = 0;
  result[1] = 0;
  const double scale = static_cast<double>(n_events);

  q.single_task([=]() {
    int64_t nmax = 0;
    int64_t active = 0;
    for (int64_t i = 0; i < n; ++i) {
      const double scaled = weights[i] * scale;
      const int64_t count = static_cast<int64_t>(scaled < 0.0 ? -scaled : scaled);
      counts[i] = count;
      active += count;
      if (count > nmax) nmax = count;
    }
    result[0] = nmax;
    result[1] = active;
  });
  q.wait_and_throw();

  const Allocation allocation{result[0], result[1]};
  sycl::free(result, q);
  return allocation;
}

void fill_uniform(float* QUANTOM_RESTRICT values,
                  int64_t count,
                  uint64_t seed,
                  uint64_t stream) {
  const int64_t blocks = (count + 3) / 4;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(blocks)), [=](sycl::id<1> id) {
    const int64_t block = static_cast<int64_t>(id[0]);
    const auto random = quantom::rng::philox4x32_10(
        static_cast<uint64_t>(block), stream, seed);
    const int64_t base = block * 4;
    if (base < count) values[base] = quantom::rng::uniform_float(random.x0);
    if (base + 1 < count) values[base + 1] = quantom::rng::uniform_float(random.x1);
    if (base + 2 < count) values[base + 2] = quantom::rng::uniform_float(random.x2);
    if (base + 3 < count) values[base + 3] = quantom::rng::uniform_float(random.x3);
  });
}

void density_x(const double* QUANTOM_RESTRICT bins,
               const double* QUANTOM_RESTRICT xsec,
               double* QUANTOM_RESTRICT norm,
               double* QUANTOM_RESTRICT rho,
               const Shape& s) {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t xsec_y_stride = s.nx * s.kx;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(curves)), [=](sycl::id<1> id) {
    const int64_t curve = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* xb = bins + b * bins_batch_stride + ix * s.kx;
    const double* xs = xsec + (b * s.ny + iy) * xsec_y_stride + ix * s.kx;
    double* r = rho + (b * s.ny + iy) * xsec_y_stride + ix * s.kx;
    double n = 0.0;
    for (int64_t t = 1; t < s.kx; ++t) {
      n += 0.5 * (xb[t] - xb[t - 1]) * (xs[t - 1] + xs[t]);
    }
    norm[(b * s.ny + iy) * s.nx + ix] = n;
    const double inv_n = 1.0 / n;
    for (int64_t t = 0; t < s.kx; ++t) r[t] = xs[t] * inv_n;
  });
}

void density_q(const double* QUANTOM_RESTRICT bins,
               const double* QUANTOM_RESTRICT xsec,
               double* QUANTOM_RESTRICT norm,
               double* QUANTOM_RESTRICT rho,
               const Shape& s) {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t xsec_x_stride = s.ny * s.kq;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(curves)), [=](sycl::id<1> id) {
    const int64_t curve = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* qb = bins + b * bins_batch_stride + iy * s.kq;
    const double* xs = xsec + (b * s.nx + ix) * xsec_x_stride + iy * s.kq;
    double* r = rho + (b * s.nx + ix) * xsec_x_stride + iy * s.kq;
    double n = 0.0;
    for (int64_t t = 1; t < s.kq; ++t) {
      n += 0.5 * (qb[t] - qb[t - 1]) * (xs[t - 1] + xs[t]);
    }
    norm[(b * s.nx + ix) * s.ny + iy] = n;
    const double inv_n = 1.0 / n;
    for (int64_t t = 0; t < s.kq; ++t) r[t] = xs[t] * inv_n;
  });
}

void cdf_x(const double* QUANTOM_RESTRICT bins,
           const double* QUANTOM_RESTRICT rho,
           const bool* QUANTOM_RESTRICT acceptance,
           double* QUANTOM_RESTRICT cdf,
           const Shape& s) {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t curve_y_stride = s.nx * s.kx;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(curves)), [=](sycl::id<1> id) {
    const int64_t curve = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* xb = bins + b * bins_batch_stride + ix * s.kx;
    const double* r = rho + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    double* out = cdf + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    const double acc = acceptance[(b * s.nx + ix) * s.ny + iy] ? 1.0 : 0.0;
    double cumulative = 0.0;
    out[0] = 0.0;
    for (int64_t t = 1; t < s.kx; ++t) {
      cumulative += 0.5 * (xb[t] - xb[t - 1]) * (r[t - 1] + r[t]);
      out[t] = cumulative * acc;
    }
  });
}

void cdf_q(const double* QUANTOM_RESTRICT bins,
           const double* QUANTOM_RESTRICT rho,
           const bool* QUANTOM_RESTRICT acceptance,
           double* QUANTOM_RESTRICT cdf,
           const Shape& s) {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t curve_x_stride = s.ny * s.kq;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(curves)), [=](sycl::id<1> id) {
    const int64_t curve = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* qb = bins + b * bins_batch_stride + iy * s.kq;
    const double* r = rho + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    double* out = cdf + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    const double acc = acceptance[(b * s.nx + ix) * s.ny + iy] ? 1.0 : 0.0;
    double cumulative = 0.0;
    out[0] = 0.0;
    for (int64_t t = 1; t < s.kq; ++t) {
      cumulative += 0.5 * (qb[t] - qb[t - 1]) * (r[t - 1] + r[t]);
      out[t] = cumulative * acc;
    }
  });
}

void interpolate_x(const double* QUANTOM_RESTRICT bins,
                   const double* QUANTOM_RESTRICT cdf,
                   const float* QUANTOM_RESTRICT u,
                   const int64_t* QUANTOM_RESTRICT counts,
                   int64_t nmax,
                   double* QUANTOM_RESTRICT dense,
                   int16_t* QUANTOM_RESTRICT indices,
                   const Shape& s) {
  const int64_t total_slots = s.batch * s.cells * nmax;
  const int64_t slots_per_batch = s.cells * nmax;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t cdf_batch_stride = s.ny * s.nx * s.kx;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(total_slots)), [=](sycl::id<1> id) {
    const int64_t p = static_cast<int64_t>(id[0]);
    const int64_t b = p / slots_per_batch;
    const int64_t within_batch = p - b * slots_per_batch;
    const int64_t cell = within_batch / nmax;
    const int64_t slot = within_batch - cell * nmax;
    const int64_t global_cell = b * s.cells + cell;
    if (slot >= counts[global_cell]) return;
    const int64_t ix = cell / s.ny;
    const int64_t iy = cell - ix * s.ny;
    const double* bin = bins + b * bins_batch_stride + ix * s.kx;
    const double* curve = cdf + b * cdf_batch_stride + (iy * s.nx + ix) * s.kx;
    const double uv = static_cast<double>(u[p]);
    const int16_t j = interval(curve, s.kx, uv);
    indices[p] = j;
    const double c0 = curve[j];
    const double inv_d = 1.0 / (curve[j + 1] - c0 + kEpsilon);
    const double m = (bin[j + 1] - bin[j]) * inv_d;
    dense[p] = bin[j] + m * (uv - c0);
  });
}

void interpolate_q(const double* QUANTOM_RESTRICT bins,
                   const double* QUANTOM_RESTRICT cdf,
                   const float* QUANTOM_RESTRICT u,
                   const int64_t* QUANTOM_RESTRICT counts,
                   int64_t nmax,
                   double* QUANTOM_RESTRICT dense,
                   int16_t* QUANTOM_RESTRICT indices,
                   const Shape& s) {
  const int64_t total_slots = s.batch * s.cells * nmax;
  const int64_t slots_per_batch = s.cells * nmax;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t cdf_batch_stride = s.nx * s.ny * s.kq;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(total_slots)), [=](sycl::id<1> id) {
    const int64_t p = static_cast<int64_t>(id[0]);
    const int64_t b = p / slots_per_batch;
    const int64_t within_batch = p - b * slots_per_batch;
    const int64_t cell = within_batch / nmax;
    const int64_t slot = within_batch - cell * nmax;
    const int64_t global_cell = b * s.cells + cell;
    if (slot >= counts[global_cell]) return;
    const int64_t ix = cell / s.ny;
    const int64_t iy = cell - ix * s.ny;
    const double* curve = cdf + b * cdf_batch_stride + (ix * s.ny + iy) * s.kq;
    const double* bin = bins + b * bins_batch_stride + iy * s.kq;
    const double uv = static_cast<double>(u[p]);
    const int16_t j = interval(curve, s.kq, uv);
    indices[p] = j;
    const double c0 = curve[j];
    const double inv_d = 1.0 / (curve[j + 1] - c0 + kEpsilon);
    const double m = (bin[j + 1] - bin[j]) * inv_d;
    dense[p] = bin[j] + m * (uv - c0);
  });
}

int64_t compact(const double* QUANTOM_RESTRICT dense_x,
                const double* QUANTOM_RESTRICT dense_q,
                const int64_t* QUANTOM_RESTRICT counts,
                int64_t nmax,
                double* QUANTOM_RESTRICT events,
                int64_t* QUANTOM_RESTRICT packed,
                int64_t* QUANTOM_RESTRICT row_offsets,
                const Shape& s) {
  const int64_t global_cells = s.batch * s.cells;
  auto& q = queue();
  const int64_t slots_per_batch = s.cells * nmax;
  const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
  const size_t wg = max_wg < 256 ? max_wg : 256;
  const sycl::nd_range<1> launch{
      sycl::range<1>(static_cast<size_t>(global_cells) * wg),
      sycl::range<1>(wg)};

  q.parallel_for(launch, [=](sycl::nd_item<1> item) {
    const int64_t global_cell = static_cast<int64_t>(item.get_group(0));
    const int64_t lane = static_cast<int64_t>(item.get_local_id(0));
    const int64_t b = global_cell / s.cells;
    const int64_t cell = global_cell - b * s.cells;
    const int64_t base = b * slots_per_batch + cell * nmax;
    const int64_t count = counts[global_cell];
    int64_t valid_total = 0;

    for (int64_t chunk = 0; chunk < count; chunk += static_cast<int64_t>(wg)) {
      const int64_t slot = chunk + lane;
      int64_t flag = 0;
      if (slot < count) {
        const int64_t p = base + slot;
        const double xv = dense_x[p];
        const double qv = dense_q[p];
        flag = static_cast<int64_t>(sycl::isfinite(xv) && sycl::isfinite(qv) && xv * qv != 0.0);
      }
      valid_total += sycl::reduce_over_group(item.get_group(), flag, sycl::plus<int64_t>());
    }

    if (lane == 0) {
      if (global_cell == 0) row_offsets[0] = 0;
      row_offsets[global_cell + 1] = valid_total;
    }
  });

  int64_t* valid_shared = sycl::malloc_shared<int64_t>(1, q);
  *valid_shared = 0;
  q.single_task([=]() {
    int64_t prefix = 0;
    for (int64_t cell = 0; cell < global_cells; ++cell) {
      const int64_t count = row_offsets[cell + 1];
      row_offsets[cell] = prefix;
      prefix += count;
    }
    row_offsets[global_cells] = prefix;
    *valid_shared = prefix;
  });
  q.wait_and_throw();
  const int64_t valid = *valid_shared;
  sycl::free(valid_shared, q);

  if (valid > 0) {
    q.parallel_for(launch, [=](sycl::nd_item<1> item) {
      const int64_t global_cell = static_cast<int64_t>(item.get_group(0));
      const int64_t lane = static_cast<int64_t>(item.get_local_id(0));
      const int64_t b = global_cell / s.cells;
      const int64_t cell = global_cell - b * s.cells;
      const int64_t base = b * slots_per_batch + cell * nmax;
      const int64_t count = counts[global_cell];
      int64_t chunk_base = 0;

      for (int64_t chunk = 0; chunk < count; chunk += static_cast<int64_t>(wg)) {
        const int64_t slot = chunk + lane;
        int64_t flag = 0;
        double xv = 0.0;
        double qv = 0.0;
        int64_t p = 0;
        if (slot < count) {
          p = base + slot;
          xv = dense_x[p];
          qv = dense_q[p];
          flag = static_cast<int64_t>(sycl::isfinite(xv) && sycl::isfinite(qv) && xv * qv != 0.0);
        }

        const int64_t local_offset = sycl::exclusive_scan_over_group(
            item.get_group(), flag, int64_t{0}, sycl::plus<int64_t>());
        const int64_t chunk_valid = sycl::reduce_over_group(
            item.get_group(), flag, sycl::plus<int64_t>());
        if (flag) {
          const int64_t row = row_offsets[global_cell] + chunk_base + local_offset;
          events[row * 2] = xv;
          events[row * 2 + 1] = qv;
          packed[row] = p;
        }
        chunk_base += chunk_valid;
      }
    });
  }
  return valid;
}

void interpolation_vjp_x(const double* QUANTOM_RESTRICT grad_events,
                         const int64_t* QUANTOM_RESTRICT packed,
                         const int64_t* QUANTOM_RESTRICT row_offsets,
                         const double* QUANTOM_RESTRICT bins,
                         const double* QUANTOM_RESTRICT cdf,
                         const float* QUANTOM_RESTRICT u,
                         const int16_t* QUANTOM_RESTRICT indices,
                         int64_t nmax,
                         double* QUANTOM_RESTRICT grad_cdf,
                         const Shape& s) {
  (void)nmax;
  const int64_t global_cells = s.batch * s.cells;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t cdf_batch_stride = s.ny * s.nx * s.kx;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(global_cells)), [=](sycl::id<1> id) {
    const int64_t global_cell = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(global_cell, s, b, ix, iy);
    const double* curve = cdf + b * cdf_batch_stride + (iy * s.nx + ix) * s.kx;
    const double* bin = bins + b * bins_batch_stride + ix * s.kx;
    double* gcurve = grad_cdf + b * cdf_batch_stride + (iy * s.nx + ix) * s.kx;
    for (int64_t t = 0; t < s.kx; ++t) gcurve[t] = 0.0;
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
  });
}

void interpolation_vjp_q(const double* QUANTOM_RESTRICT grad_events,
                         const int64_t* QUANTOM_RESTRICT packed,
                         const int64_t* QUANTOM_RESTRICT row_offsets,
                         const double* QUANTOM_RESTRICT bins,
                         const double* QUANTOM_RESTRICT cdf,
                         const float* QUANTOM_RESTRICT u,
                         const int16_t* QUANTOM_RESTRICT indices,
                         int64_t nmax,
                         double* QUANTOM_RESTRICT grad_cdf,
                         const Shape& s) {
  (void)nmax;
  const int64_t global_cells = s.batch * s.cells;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t cdf_batch_stride = s.nx * s.ny * s.kq;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(global_cells)), [=](sycl::id<1> id) {
    const int64_t global_cell = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(global_cell, s, b, ix, iy);
    const double* curve = cdf + b * cdf_batch_stride + (ix * s.ny + iy) * s.kq;
    const double* bin = bins + b * bins_batch_stride + iy * s.kq;
    double* gcurve = grad_cdf + b * cdf_batch_stride + (ix * s.ny + iy) * s.kq;
    for (int64_t t = 0; t < s.kq; ++t) gcurve[t] = 0.0;
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
  });
}

void cdf_vjp_x(const double* QUANTOM_RESTRICT bins,
               const bool* QUANTOM_RESTRICT acceptance,
               const double* QUANTOM_RESTRICT grad_cdf,
               double* QUANTOM_RESTRICT grad_rho,
               const Shape& s) {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t curve_y_stride = s.nx * s.kx;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(curves)), [=](sycl::id<1> id) {
    const int64_t curve = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* xb = bins + b * bins_batch_stride + ix * s.kx;
    const double* gc = grad_cdf + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    double* gr = grad_rho + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    for (int64_t t = 0; t < s.kx; ++t) gr[t] = 0.0;
    const double acc = acceptance[(b * s.nx + ix) * s.ny + iy] ? 1.0 : 0.0;
    double suffix = 0.0;
    for (int64_t t = s.kx - 1; t > 0; --t) {
      suffix += gc[t];
      const double g = 0.5 * (xb[t] - xb[t - 1]) * acc * suffix;
      gr[t - 1] += g;
      gr[t] += g;
    }
  });
}

void cdf_vjp_q(const double* QUANTOM_RESTRICT bins,
               const bool* QUANTOM_RESTRICT acceptance,
               const double* QUANTOM_RESTRICT grad_cdf,
               double* QUANTOM_RESTRICT grad_rho,
               const Shape& s) {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t curve_x_stride = s.ny * s.kq;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(curves)), [=](sycl::id<1> id) {
    const int64_t curve = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* qb = bins + b * bins_batch_stride + iy * s.kq;
    const double* gc = grad_cdf + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    double* gr = grad_rho + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    for (int64_t t = 0; t < s.kq; ++t) gr[t] = 0.0;
    const double acc = acceptance[(b * s.nx + ix) * s.ny + iy] ? 1.0 : 0.0;
    double suffix = 0.0;
    for (int64_t t = s.kq - 1; t > 0; --t) {
      suffix += gc[t];
      const double g = 0.5 * (qb[t] - qb[t - 1]) * acc * suffix;
      gr[t - 1] += g;
      gr[t] += g;
    }
  });
}

void density_vjp_x(const double* QUANTOM_RESTRICT bins,
                   const double* QUANTOM_RESTRICT xsec,
                   const double* QUANTOM_RESTRICT norm,
                   const double* QUANTOM_RESTRICT grad_rho,
                   double* QUANTOM_RESTRICT grad_xsec,
                   const Shape& s) {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.nx * s.kx;
  const int64_t curve_y_stride = s.nx * s.kx;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(curves)), [=](sycl::id<1> id) {
    const int64_t curve = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* xb = bins + b * bins_batch_stride + ix * s.kx;
    const double* xs = xsec + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    const double* gr = grad_rho + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
    double* gx = grad_xsec + (b * s.ny + iy) * curve_y_stride + ix * s.kx;
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
  });
}

void density_vjp_q(const double* QUANTOM_RESTRICT bins,
                   const double* QUANTOM_RESTRICT xsec,
                   const double* QUANTOM_RESTRICT norm,
                   const double* QUANTOM_RESTRICT grad_rho,
                   double* QUANTOM_RESTRICT grad_xsec,
                   const Shape& s) {
  const int64_t curves = s.batch * s.nx * s.ny;
  const int64_t bins_batch_stride = s.ny * s.kq;
  const int64_t curve_x_stride = s.ny * s.kq;
  queue().parallel_for(sycl::range<1>(static_cast<size_t>(curves)), [=](sycl::id<1> id) {
    const int64_t curve = static_cast<int64_t>(id[0]);
    int64_t b, ix, iy;
    decode_cell(curve, s, b, ix, iy);
    const double* qb = bins + b * bins_batch_stride + iy * s.kq;
    const double* xs = xsec + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    const double* gr = grad_rho + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
    double* gx = grad_xsec + (b * s.nx + ix) * curve_x_stride + iy * s.kq;
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
  });
}

}  // namespace sycl_loits
