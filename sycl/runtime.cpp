#include "runtime.hpp"

#if defined(QUANTOM_DPCPP_HIP)
#include <hip/hip_runtime_api.h>
#include <sycl/backend.hpp>

#include <optional>
#include <stdexcept>
#endif

namespace sycl_loits {

#if defined(QUANTOM_DPCPP_HIP)
namespace {
constexpr auto kHipBackend = sycl::backend::ext_oneapi_hip;
using HipNativeQueue = sycl::backend_input_t<kHipBackend, sycl::queue>;

sycl::device hip_device(int index) {
  for (const auto& platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != kHipBackend) continue;
    auto devices = platform.get_devices(sycl::info::device_type::gpu);
    if (index < static_cast<int>(devices.size())) return devices[index];
    index -= static_cast<int>(devices.size());
  }
  throw std::runtime_error("DPC++ HIP device index is not visible to SYCL");
}

struct QueueState {
  int device = -1;
  uintptr_t stream = 0;
  std::optional<sycl::context> context;
  std::optional<sycl::queue> queue;
};

QueueState& queue_state() {
  static thread_local QueueState state;
  return state;
}
}  // namespace

sycl::queue& queue() {
  return *queue_state().queue;
}
#else
sycl::queue& queue() {
  static sycl::queue q{sycl::default_selector_v,
                       sycl::property_list{sycl::property::queue::in_order{}}};
  return q;
}
#endif

void bind_torch_hip_stream(uintptr_t native_stream, int device_index) {
#if defined(QUANTOM_DPCPP_HIP)
  auto& state = queue_state();
  if (!state.context || state.device != device_index) {
    state.queue.reset();
    state.context.emplace(hip_device(device_index));
    state.device = device_index;
    state.stream = 0;
  }
  if (!state.queue || state.stream != native_stream) {
    state.queue.reset();
    state.queue.emplace(sycl::make_queue<kHipBackend>(
        reinterpret_cast<HipNativeQueue>(native_stream), *state.context));
    state.stream = native_stream;
  }
#else
  (void)native_stream;
  (void)device_index;
#endif
}

void finish_submission(bool wait) {
  if (wait) queue().wait_and_throw();
#if defined(QUANTOM_DPCPP_HIP)
  const auto error = hipGetLastError();
  if (error != hipSuccess && error != hipErrorNotFound) {
    throw std::runtime_error(hipGetErrorString(error));
  }
#endif
}

void synchronize() {
  finish_submission(true);
}

}  // namespace sycl_loits
