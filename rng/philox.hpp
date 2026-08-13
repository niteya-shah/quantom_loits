#pragma once

#include <cstdint>

namespace quantom::rng {

struct Philox4x32Result {
  uint32_t x0;
  uint32_t x1;
  uint32_t x2;
  uint32_t x3;
};

// Philox4x32-10 constants and round structure follow Random123.
// The generator is counter based: (block, stream) is the 128-bit counter and
// seed is the 64-bit key. Each block produces four deterministic uint32 values.
inline Philox4x32Result philox4x32_10(uint64_t block,
                                     uint64_t stream,
                                     uint64_t seed) noexcept {
  constexpr uint32_t kMul0 = 0xD2511F53u;
  constexpr uint32_t kMul1 = 0xCD9E8D57u;
  constexpr uint32_t kWeyl0 = 0x9E3779B9u;
  constexpr uint32_t kWeyl1 = 0xBB67AE85u;

  uint32_t c0 = static_cast<uint32_t>(block);
  uint32_t c1 = static_cast<uint32_t>(block >> 32);
  uint32_t c2 = static_cast<uint32_t>(stream);
  uint32_t c3 = static_cast<uint32_t>(stream >> 32);
  uint32_t k0 = static_cast<uint32_t>(seed);
  uint32_t k1 = static_cast<uint32_t>(seed >> 32);

  for (int round = 0; round < 10; ++round) {
    const uint64_t product0 = static_cast<uint64_t>(kMul0) * c0;
    const uint64_t product1 = static_cast<uint64_t>(kMul1) * c2;
    const uint32_t lo0 = static_cast<uint32_t>(product0);
    const uint32_t hi0 = static_cast<uint32_t>(product0 >> 32);
    const uint32_t lo1 = static_cast<uint32_t>(product1);
    const uint32_t hi1 = static_cast<uint32_t>(product1 >> 32);

    const uint32_t next0 = hi1 ^ c1 ^ k0;
    const uint32_t next1 = lo1;
    const uint32_t next2 = hi0 ^ c3 ^ k1;
    const uint32_t next3 = lo0;
    c0 = next0;
    c1 = next1;
    c2 = next2;
    c3 = next3;

    if (round != 9) {
      k0 += kWeyl0;
      k1 += kWeyl1;
    }
  }

  return {c0, c1, c2, c3};
}

// Use the high 24 bits so conversion to float is exact and always in [0, 1).
inline float uniform_float(uint32_t bits) noexcept {
  return static_cast<float>(bits >> 8) * 0x1.0p-24f;
}

}  // namespace quantom::rng
