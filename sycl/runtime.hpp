#pragma once

#include <cstdint>

#if defined(QUANTOM_SYCL_NATIVE)
#include <sycl/sycl.hpp>
#endif

namespace sycl_loits {

#if defined(QUANTOM_SYCL_NATIVE)
sycl::queue& queue();
#endif

void bind_torch_hip_stream(uintptr_t native_stream, int device_index);
void finish_submission(bool wait);
void synchronize();

}  // namespace sycl_loits
