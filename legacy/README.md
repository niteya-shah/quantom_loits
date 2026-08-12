# Legacy native reference

`cpp_sampler.*` is retained only because the pre-existing OpenMP and SYCL implementations still use it as a correctness/reference helper. The active C++ backend is `cpp/loits.cpp` and does not depend on the legacy matrix abstraction or timing/test code.

The legacy reference can be removed when the OpenMP and SYCL backends are migrated to the same tensor/autograd interface.
