#include <torch/extension.h>
#include <ATen/record_function.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "loits_core.hpp"

namespace {

class RegionGuard {
 public:
  RegionGuard(bool enabled, const char* name) {
    if (!enabled) return;
    guard_ = std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (guard_->isActive()) guard_->before(name);
  }

 private:
  std::unique_ptr<at::RecordFunction> guard_;
};

inline void maybe_sync(bool profile_regions) {
  if (profile_regions) {
    sycl_loits::synchronize();
  } else {
    sycl_loits::clear_expected_hip_error();
  }
}

inline void check_fp64_contiguous(const at::Tensor& tensor,
                                  const at::Device& device,
                                  const char* name) {
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == at::kDouble, name, " must be float64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

sycl_loits::Shape validate_forward(const at::Tensor& x_bins,
                                   const at::Tensor& xsec_x,
                                   const at::Tensor& q_bins,
                                   const at::Tensor& xsec_q,
                                   const at::Tensor& weights,
                                   const at::Tensor& acceptance,
                                   int64_t n_events) {
  const auto device = xsec_x.device();
  TORCH_CHECK(device.is_cpu() || device.is_cuda() || device.type() == at::kXPU,
              "SYCL backend supports torch CPU, CUDA/ROCm, or XPU tensors");
  check_fp64_contiguous(x_bins, device, "x_bins");
  check_fp64_contiguous(xsec_x, device, "xsec_x");
  check_fp64_contiguous(q_bins, device, "q_bins");
  check_fp64_contiguous(xsec_q, device, "xsec_q");
  check_fp64_contiguous(weights, device, "weights");
  TORCH_CHECK(acceptance.device() == device, "acceptance must be on ", device);
  TORCH_CHECK(acceptance.scalar_type() == at::kBool, "acceptance must be bool");
  TORCH_CHECK(acceptance.is_contiguous(), "acceptance must be contiguous");
  TORCH_CHECK(n_events >= 0, "n_events must be non-negative");
  TORCH_CHECK(sycl_loits::supports_fp64(), "selected SYCL device does not support float64: ", sycl_loits::device_name());

  TORCH_CHECK(x_bins.dim() == 3, "x_bins must be [batch, nx, kx]");
  TORCH_CHECK(xsec_x.dim() == 4, "xsec_x must be [batch, ny, nx, kx]");
  TORCH_CHECK(q_bins.dim() == 3, "q_bins must be [batch, ny, kq]");
  TORCH_CHECK(xsec_q.dim() == 4, "xsec_q must be [batch, nx, ny, kq]");
  TORCH_CHECK(weights.dim() == 3, "weights must be [batch, nx, ny]");
  TORCH_CHECK(acceptance.dim() == 3, "acceptance must be [batch, nx, ny]");

  const sycl_loits::Shape s{
      x_bins.size(0), x_bins.size(1), q_bins.size(1), x_bins.size(2), q_bins.size(2), x_bins.size(1) * q_bins.size(1)};
  TORCH_CHECK(s.kx >= 2 && s.kq >= 2, "CDF grids require at least two points");
  TORCH_CHECK(s.kx <= INT16_MAX && s.kq <= INT16_MAX, "CDF grids exceed interval-index storage");
  TORCH_CHECK(q_bins.size(0) == s.batch, "q_bins batch mismatch");
  TORCH_CHECK(xsec_x.sizes() == at::IntArrayRef({s.batch, s.ny, s.nx, s.kx}), "xsec_x shape mismatch");
  TORCH_CHECK(xsec_q.sizes() == at::IntArrayRef({s.batch, s.nx, s.ny, s.kq}), "xsec_q shape mismatch");
  TORCH_CHECK(weights.sizes() == at::IntArrayRef({s.batch, s.nx, s.ny}), "weights shape mismatch");
  TORCH_CHECK(acceptance.sizes() == weights.sizes(), "acceptance shape mismatch");
  return s;
}

std::vector<at::Tensor> forward(at::Tensor x_bins,
                                at::Tensor xsec_x,
                                at::Tensor q_bins,
                                at::Tensor xsec_q,
                                at::Tensor weights,
                                at::Tensor acceptance,
                                int64_t n_events,
                                uint64_t seed,
                                uint64_t sequence,
                                bool profile_regions) {
  RegionGuard total(profile_regions, "loits::forward");

  sycl_loits::Shape s{};
  {
    RegionGuard region(profile_regions, "loits::forward::validation");
    s = validate_forward(x_bins, xsec_x, q_bins, xsec_q, weights, acceptance, n_events);
  }

  at::Tensor counts;
  sycl_loits::Allocation allocation{};
  {
    RegionGuard region(profile_regions, "loits::forward::allocation");
    counts = at::empty({s.batch, s.cells}, weights.options().dtype(at::kLong));
    allocation = sycl_loits::allocate_counts(weights.data_ptr<double>(), counts.data_ptr<int64_t>(), n_events, s);
    maybe_sync(profile_regions);
  }

  at::Tensor norm_x;
  at::Tensor rho_x;
  {
    RegionGuard region(profile_regions, "loits::forward::rho_x");
    norm_x = at::empty({s.batch, s.ny, s.nx}, xsec_x.options());
    rho_x = at::empty_like(xsec_x);
    sycl_loits::density_x(x_bins.data_ptr<double>(), xsec_x.data_ptr<double>(), norm_x.data_ptr<double>(), rho_x.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor norm_q;
  at::Tensor rho_q;
  {
    RegionGuard region(profile_regions, "loits::forward::rho_q2");
    norm_q = at::empty({s.batch, s.nx, s.ny}, xsec_q.options());
    rho_q = at::empty_like(xsec_q);
    sycl_loits::density_q(q_bins.data_ptr<double>(), xsec_q.data_ptr<double>(), norm_q.data_ptr<double>(), rho_q.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor cdf_x;
  {
    RegionGuard region(profile_regions, "loits::forward::cdf_x");
    cdf_x = at::empty_like(xsec_x);
    sycl_loits::cdf_x(x_bins.data_ptr<double>(), rho_x.data_ptr<double>(), acceptance.data_ptr<bool>(), cdf_x.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor cdf_q;
  {
    RegionGuard region(profile_regions, "loits::forward::cdf_q2");
    cdf_q = at::empty_like(xsec_q);
    sycl_loits::cdf_q(q_bins.data_ptr<double>(), rho_q.data_ptr<double>(), acceptance.data_ptr<bool>(), cdf_q.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  const int64_t total_slots = s.batch * s.cells * allocation.nmax;
  const auto random_options = xsec_x.options().dtype(at::kFloat);
  at::Tensor u_x;
  at::Tensor u_q;
  {
    RegionGuard region(profile_regions, "loits::forward::random_x");
    u_x = at::empty({s.batch, s.cells, allocation.nmax}, random_options);
    sycl_loits::fill_uniform(u_x.data_ptr<float>(), total_slots, seed, sequence * 2);
    maybe_sync(profile_regions);
  }
  {
    RegionGuard region(profile_regions, "loits::forward::random_q2");
    u_q = at::empty({s.batch, s.cells, allocation.nmax}, random_options);
    sycl_loits::fill_uniform(u_q.data_ptr<float>(), total_slots, seed, sequence * 2 + 1);
    maybe_sync(profile_regions);
  }

  at::Tensor dense_x;
  at::Tensor interval_x;
  {
    RegionGuard region(profile_regions, "loits::forward::interpolation_x");
    dense_x = at::empty({total_slots}, xsec_x.options());
    interval_x = at::empty({total_slots}, xsec_x.options().dtype(at::kShort));
    sycl_loits::interpolate_x(x_bins.data_ptr<double>(), cdf_x.data_ptr<double>(), u_x.data_ptr<float>(), counts.data_ptr<int64_t>(),
                              allocation.nmax, dense_x.data_ptr<double>(), interval_x.data_ptr<int16_t>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor dense_q;
  at::Tensor interval_q;
  {
    RegionGuard region(profile_regions, "loits::forward::interpolation_q2");
    dense_q = at::empty({total_slots}, xsec_q.options());
    interval_q = at::empty({total_slots}, xsec_q.options().dtype(at::kShort));
    sycl_loits::interpolate_q(q_bins.data_ptr<double>(), cdf_q.data_ptr<double>(), u_q.data_ptr<float>(), counts.data_ptr<int64_t>(),
                              allocation.nmax, dense_q.data_ptr<double>(), interval_q.data_ptr<int16_t>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor events;
  at::Tensor packed;
  at::Tensor row_offsets;
  {
    RegionGuard region(profile_regions, "loits::forward::stream_compaction");
    auto events_storage = at::empty({allocation.active, 2}, xsec_x.options());
    auto packed_storage = at::empty({allocation.active}, counts.options());
    row_offsets = at::empty({s.batch * s.cells + 1}, counts.options());
    const int64_t valid = sycl_loits::compact(dense_x.data_ptr<double>(), dense_q.data_ptr<double>(), counts.data_ptr<int64_t>(),
                                              allocation.nmax, events_storage.data_ptr<double>(), packed_storage.data_ptr<int64_t>(),
                                              row_offsets.data_ptr<int64_t>(), s);
    maybe_sync(profile_regions);
    events = events_storage.narrow(0, 0, valid);
    packed = packed_storage.narrow(0, 0, valid);
  }

  sycl_loits::synchronize();
  {
    RegionGuard region(profile_regions, "loits::forward::state_pack");
    return {events, norm_x, norm_q, cdf_x, cdf_q, u_x, u_q, interval_x, interval_q, packed, row_offsets};
  }
}

std::vector<at::Tensor> backward(at::Tensor grad_events,
                                 at::Tensor x_bins,
                                 at::Tensor xsec_x,
                                 at::Tensor q_bins,
                                 at::Tensor xsec_q,
                                 at::Tensor acceptance,
                                 at::Tensor norm_x,
                                 at::Tensor norm_q,
                                 at::Tensor cdf_x,
                                 at::Tensor cdf_q,
                                 at::Tensor u_x,
                                 at::Tensor u_q,
                                 at::Tensor interval_x,
                                 at::Tensor interval_q,
                                 at::Tensor packed,
                                 at::Tensor row_offsets,
                                 bool profile_regions) {
  RegionGuard total(profile_regions, "loits::backward");

  sycl_loits::Shape s{};
  int64_t nmax = 0;
  {
    RegionGuard region(profile_regions, "loits::backward::validation");
    const auto device = xsec_x.device();
    check_fp64_contiguous(grad_events, device, "grad_events");
    TORCH_CHECK(grad_events.dim() == 2 && grad_events.size(1) == 2 && grad_events.size(0) == packed.size(0),
                "grad_events shape mismatch");
    s = {x_bins.size(0), x_bins.size(1), q_bins.size(1), x_bins.size(2), q_bins.size(2), x_bins.size(1) * q_bins.size(1)};
    nmax = u_x.size(2);
  }

  at::Tensor grad_cdf_x;
  {
    RegionGuard region(profile_regions, "loits::backward::interpolation_x");
    grad_cdf_x = at::empty_like(cdf_x);
    sycl_loits::interpolation_vjp_x(grad_events.data_ptr<double>(), packed.data_ptr<int64_t>(), row_offsets.data_ptr<int64_t>(),
                                    x_bins.data_ptr<double>(), cdf_x.data_ptr<double>(), u_x.data_ptr<float>(),
                                    interval_x.data_ptr<int16_t>(), nmax, grad_cdf_x.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor grad_cdf_q;
  {
    RegionGuard region(profile_regions, "loits::backward::interpolation_q2");
    grad_cdf_q = at::empty_like(cdf_q);
    sycl_loits::interpolation_vjp_q(grad_events.data_ptr<double>(), packed.data_ptr<int64_t>(), row_offsets.data_ptr<int64_t>(),
                                    q_bins.data_ptr<double>(), cdf_q.data_ptr<double>(), u_q.data_ptr<float>(),
                                    interval_q.data_ptr<int16_t>(), nmax, grad_cdf_q.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor grad_rho_x;
  {
    RegionGuard region(profile_regions, "loits::backward::cdf_x");
    grad_rho_x = at::empty_like(xsec_x);
    sycl_loits::cdf_vjp_x(x_bins.data_ptr<double>(), acceptance.data_ptr<bool>(), grad_cdf_x.data_ptr<double>(),
                          grad_rho_x.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor grad_rho_q;
  {
    RegionGuard region(profile_regions, "loits::backward::cdf_q2");
    grad_rho_q = at::empty_like(xsec_q);
    sycl_loits::cdf_vjp_q(q_bins.data_ptr<double>(), acceptance.data_ptr<bool>(), grad_cdf_q.data_ptr<double>(),
                          grad_rho_q.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor grad_xsec_x;
  {
    RegionGuard region(profile_regions, "loits::backward::rho_x");
    grad_xsec_x = at::empty_like(xsec_x);
    sycl_loits::density_vjp_x(x_bins.data_ptr<double>(), xsec_x.data_ptr<double>(), norm_x.data_ptr<double>(),
                              grad_rho_x.data_ptr<double>(), grad_xsec_x.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  at::Tensor grad_xsec_q;
  {
    RegionGuard region(profile_regions, "loits::backward::rho_q2");
    grad_xsec_q = at::empty_like(xsec_q);
    sycl_loits::density_vjp_q(q_bins.data_ptr<double>(), xsec_q.data_ptr<double>(), norm_q.data_ptr<double>(),
                              grad_rho_q.data_ptr<double>(), grad_xsec_q.data_ptr<double>(), s);
    maybe_sync(profile_regions);
  }

  sycl_loits::synchronize();
  {
    RegionGuard region(profile_regions, "loits::backward::state_pack");
    return {grad_xsec_x, grad_xsec_q};
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("backward", &backward);
  m.def("device_name", []() { return std::string(sycl_loits::device_name()); });
  m.def("supports_fp64", &sycl_loits::supports_fp64);
  m.def("bind_torch_stream", &sycl_loits::bind_torch_stream);
  m.def("synchronize", &sycl_loits::synchronize);
}
