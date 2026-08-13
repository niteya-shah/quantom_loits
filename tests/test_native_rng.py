import torch

from cpp.backend import load_extension as load_cpp_extension
from openmp.backend import load_extension as load_openmp_extension
from pytorch.theory import TorchProxyTheoryLite


def make_outputs(grid=3):
    theory = TorchProxyTheoryLite(
        {
            "n_points_x": grid,
            "n_points_y": grid,
            "n_cdf_points_x": 5,
            "n_cdf_points_y": 5,
            "average": False,
        },
        "cpu",
    )
    params = torch.tensor(
        [[0.67, 0.20, 0.23, 0.67, 0.50]],
        dtype=torch.float64,
        requires_grad=True,
    )
    return theory(params)


def native_state(extension, outputs, *, seed, sequence, n_events=80):
    x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = outputs[:6]
    return extension.forward(
        x_bins,
        xsec_x,
        q2_bins,
        xsec_q2,
        weights,
        acceptance,
        n_events,
        seed,
        sequence,
        False,
    )


def test_philox4x32_10_known_answer():
    outputs = make_outputs()
    # Random123 Philox4x32-10 KAT for counter=(0,0,0,0), key=(0,0).
    words = [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    expected = torch.tensor(
        [(word >> 8) * (2.0**-24) for word in words], dtype=torch.float32
    )

    for extension in (load_cpp_extension(), load_openmp_extension()):
        state = native_state(extension, outputs, seed=0, sequence=0)
        u_x = state[5].flatten()
        assert u_x.numel() >= 4
        assert torch.equal(u_x[:4], expected)


def test_cpp_and_openmp_philox_streams_match_exactly():
    outputs = make_outputs()
    cpp_state = native_state(load_cpp_extension(), outputs, seed=23, sequence=7)
    omp_state = native_state(load_openmp_extension(), outputs, seed=23, sequence=7)

    cpp_u_x, cpp_u_q = cpp_state[5], cpp_state[6]
    omp_u_x, omp_u_q = omp_state[5], omp_state[6]

    assert torch.equal(cpp_u_x, omp_u_x)
    assert torch.equal(cpp_u_q, omp_u_q)
    assert not torch.equal(cpp_u_x, cpp_u_q)
    assert bool(((cpp_u_x >= 0.0) & (cpp_u_x < 1.0)).all())
    assert bool(((cpp_u_q >= 0.0) & (cpp_u_q < 1.0)).all())


def test_philox_seed_and_sequence_are_deterministic_and_distinct():
    outputs = make_outputs()
    extension = load_cpp_extension()

    first = native_state(extension, outputs, seed=31, sequence=4)
    repeat = native_state(extension, outputs, seed=31, sequence=4)
    next_sequence = native_state(extension, outputs, seed=31, sequence=5)
    next_seed = native_state(extension, outputs, seed=32, sequence=4)

    assert torch.equal(first[5], repeat[5])
    assert torch.equal(first[6], repeat[6])
    assert not torch.equal(first[5], next_sequence[5])
    assert not torch.equal(first[5], next_seed[5])
