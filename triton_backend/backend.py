from contextlib import nullcontext

import torch
from torch import nn
from torch.profiler import record_function

import triton
import triton.language as tl


_EPSILON: float = 1e-5
_ALLOC_BLOCK: int = 256
_SLOT_BLOCK: int = 128
_CELL_LANES: int = 64
_ITEMS_PER_LANE: int = 2
_SCAN_BLOCK: int = 256


def availability(device: torch.device | str = "cpu") -> tuple[bool, str]:
    device = torch.device(device)
    if device.type == "cuda":
        return (True, "") if torch.cuda.is_available() else (
            False,
            "PyTorch CUDA/ROCm is not available",
        )
    if device.type == "xpu":
        available = hasattr(torch, "xpu") and torch.xpu.is_available()
        return (True, "") if available else (False, "PyTorch XPU is not available")
    return False, "Triton backend is GPU-only (cuda/ROCm or xpu)"


def _region(enabled: bool, name: str):
    return record_function(name) if enabled else nullcontext()


def _device_guard(device: torch.device | str):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.device(device)
    if device.type == "xpu":
        return torch.xpu.device(device)
    return nullcontext()


@triton.jit
def _allocation_kernel(
    weights,
    counts,
    partial_max,
    n,
    n_events,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < n
    weight = tl.load(weights + offsets, mask=mask, other=0.0)
    count = tl.abs(weight * n_events).to(tl.int64)
    tl.store(counts + offsets, count, mask=mask)
    tl.store(partial_max + pid, tl.max(count, axis=0))


@triton.jit
def _allocation_reduce_kernel(
    partial_max,
    result,
    n,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < n
    maxima = tl.load(partial_max + offsets, mask=mask, other=0)
    tl.store(result, tl.max(maxima, axis=0))


@triton.jit
def _density_kernel(
    bins,
    xsec,
    norm,
    rho,
    NX: tl.constexpr,
    NY: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    Q_AXIS: tl.constexpr,
):
    cell = tl.program_id(0).to(tl.int64)
    CELLS: tl.constexpr = NX * NY
    b = cell // CELLS
    within = cell - b * CELLS
    ix = within // NY
    iy = within - ix * NY
    k = tl.arange(0, BLOCK_K)
    mask = k < K

    if Q_AXIS:
        bins_base = (b * NY + iy) * K
        xsec_base = ((b * NX + ix) * NY + iy) * K
    else:
        bins_base = (b * NX + ix) * K
        xsec_base = ((b * NY + iy) * NX + ix) * K

    bv = tl.load(bins + bins_base + k, mask=mask, other=0.0)
    xs = tl.load(xsec + xsec_base + k, mask=mask, other=0.0)
    prev_b = tl.load(
        bins + bins_base + k - 1,
        mask=mask & (k > 0),
        other=0.0,
    )
    prev_xs = tl.load(
        xsec + xsec_base + k - 1,
        mask=mask & (k > 0),
        other=0.0,
    )
    area = tl.where(
        mask & (k > 0),
        0.5 * (bv - prev_b) * (xs + prev_xs),
        0.0,
    )
    n = tl.sum(area, axis=0)
    tl.store(norm + cell, n)
    tl.store(rho + xsec_base + k, xs / n, mask=mask)


@triton.jit
def _cdf_kernel(
    bins,
    rho,
    acceptance,
    cdf,
    NX: tl.constexpr,
    NY: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    Q_AXIS: tl.constexpr,
):
    cell = tl.program_id(0).to(tl.int64)
    CELLS: tl.constexpr = NX * NY
    b = cell // CELLS
    within = cell - b * CELLS
    ix = within // NY
    iy = within - ix * NY
    k = tl.arange(0, BLOCK_K)
    mask = k < K

    if Q_AXIS:
        bins_base = (b * NY + iy) * K
        curve_base = ((b * NX + ix) * NY + iy) * K
    else:
        bins_base = (b * NX + ix) * K
        curve_base = ((b * NY + iy) * NX + ix) * K

    bv = tl.load(bins + bins_base + k, mask=mask, other=0.0)
    rv = tl.load(rho + curve_base + k, mask=mask, other=0.0)
    prev_b = tl.load(
        bins + bins_base + k - 1,
        mask=mask & (k > 0),
        other=0.0,
    )
    prev_r = tl.load(
        rho + curve_base + k - 1,
        mask=mask & (k > 0),
        other=0.0,
    )
    area = tl.where(
        mask & (k > 0),
        0.5 * (bv - prev_b) * (rv + prev_r),
        0.0,
    )
    acc = tl.load(acceptance + cell).to(tl.float64)
    cv = tl.cumsum(area, axis=0) * acc
    tl.store(cdf + curve_base + k, cv, mask=mask)



@triton.jit
def _philox_uniform_kernel(
    output,
    count,
    seed_lo,
    seed_hi,
    stream_lo,
    stream_hi,
    BLOCK: tl.constexpr,
):
    block = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    base = block * 4
    mask = base < count

    block_u64 = block.to(tl.uint64)
    c0 = block_u64.to(tl.uint32)
    c1 = (block_u64 >> 32).to(tl.uint32)
    c2 = tl.full((BLOCK,), stream_lo, tl.uint32)
    c3 = tl.full((BLOCK,), stream_hi, tl.uint32)
    k0 = tl.full((BLOCK,), seed_lo, tl.uint32)
    k1 = tl.full((BLOCK,), seed_hi, tl.uint32)

    m0 = tl.full((BLOCK,), 0xD2511F53, tl.uint32)
    m1 = tl.full((BLOCK,), 0xCD9E8D57, tl.uint32)
    w0 = tl.full((BLOCK,), 0x9E3779B9, tl.uint32)
    w1 = tl.full((BLOCK,), 0xBB67AE85, tl.uint32)

    for round_index in tl.static_range(0, 10):
        hi0 = tl.umulhi(c0, m0)
        lo0 = c0 * m0
        hi1 = tl.umulhi(c2, m1)
        lo1 = c2 * m1
        n0 = hi1 ^ c1 ^ k0
        n1 = lo1
        n2 = hi0 ^ c3 ^ k1
        n3 = lo0
        c0 = n0
        c1 = n1
        c2 = n2
        c3 = n3
        if round_index != 9:
            k0 = k0 + w0
            k1 = k1 + w1

    scale = tl.full((BLOCK,), 5.9604644775390625e-08, tl.float64)
    r0 = (c0 >> 8).to(tl.float64) * scale
    r1 = (c1 >> 8).to(tl.float64) * scale
    r2 = (c2 >> 8).to(tl.float64) * scale
    r3 = (c3 >> 8).to(tl.float64) * scale

    tl.store(output + base + 0, r0, mask=mask & (base + 0 < count))
    tl.store(output + base + 1, r1, mask=mask & (base + 1 < count))
    tl.store(output + base + 2, r2, mask=mask & (base + 2 < count))
    tl.store(output + base + 3, r3, mask=mask & (base + 3 < count))


@triton.jit
def _interpolate_kernel(
    bins,
    cdf,
    u,
    counts,
    dense,
    indices,
    total_slots,
    nmax,
    NX: tl.constexpr,
    NY: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    Q_AXIS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    slot_mask = p < total_slots
    cell = p // nmax
    slot = p - cell * nmax
    count = tl.load(counts + cell, mask=slot_mask, other=0)
    active = slot_mask & (slot < count)

    CELLS: tl.constexpr = NX * NY
    b = cell // CELLS
    within = cell - b * CELLS
    ix = within // NY
    iy = within - ix * NY

    if Q_AXIS:
        bins_base = (b * NY + iy) * K
        curve_base = ((b * NX + ix) * NY + iy) * K
    else:
        bins_base = (b * NX + ix) * K
        curve_base = ((b * NY + iy) * NX + ix) * K

    ks = tl.arange(0, BLOCK_K)
    k_mask = ks < K
    curve = tl.load(
        cdf + curve_base[:, None] + ks[None, :],
        mask=active[:, None] & k_mask[None, :],
        other=0.0,
    )
    uv = tl.load(u + p, mask=active, other=0.0)
    j = tl.sum(
        (k_mask[None, :] & (uv[:, None] >= curve)).to(tl.int32),
        axis=1,
    ) - 1
    j = tl.maximum(0, tl.minimum(j, K - 2))

    c0 = tl.load(cdf + curve_base + j, mask=active, other=0.0)
    c1 = tl.load(cdf + curve_base + j + 1, mask=active, other=0.0)
    b0 = tl.load(bins + bins_base + j, mask=active, other=0.0)
    b1 = tl.load(bins + bins_base + j + 1, mask=active, other=0.0)
    value = b0 + (b1 - b0) * (uv - c0) / (c1 - c0 + 1e-5)

    tl.store(dense + p, value, mask=active)
    tl.store(dense + p, 0.0, mask=slot_mask & ~active)
    tl.store(indices + p, j.to(tl.int16), mask=active)
    tl.store(indices + p, 0, mask=slot_mask & ~active)


@triton.jit
def _count_valid_kernel(
    dense_x,
    dense_q,
    counts,
    valid_counts,
    nmax,
    LANES: tl.constexpr,
    ITEMS_PER_LANE: tl.constexpr,
):
    cell = tl.program_id(0).to(tl.int64)
    count = tl.load(counts + cell).to(tl.int64)
    base = cell * nmax
    max_double = 1.7976931348623157e308

    TILE_SIZE: tl.constexpr = LANES * ITEMS_PER_LANE
    total = count - count  # int64 scalar zero; loop-carried type stays int64
    for start in tl.range(0, count, TILE_SIZE):
        slot = start + tl.arange(0, TILE_SIZE).to(tl.int64)
        mask = slot < count
        x = tl.load(dense_x + base + slot, mask=mask, other=0.0)
        q = tl.load(dense_q + base + slot, mask=mask, other=0.0)
        finite = (
            (x == x)
            & (q == q)
            & (tl.abs(x) <= max_double)
            & (tl.abs(q) <= max_double)
        )
        valid = mask & finite & (x * q != 0.0)
        total = total + tl.sum(valid.to(tl.int64), axis=0)

    tl.store(valid_counts + cell, total)


@triton.jit
def _scan_counts_kernel(
    valid_counts,
    row_offsets,
    n,
    BLOCK: tl.constexpr,
):
    carry = tl.load(valid_counts).to(tl.int64)
    carry = carry - carry  # int64 scalar zero; n is always global_cells > 0
    for start in tl.range(0, n, BLOCK):
        offsets = start + tl.arange(0, BLOCK).to(tl.int64)
        mask = offsets < n
        values = tl.load(valid_counts + offsets, mask=mask, other=0).to(tl.int64)
        inclusive = tl.cumsum(values, axis=0)
        exclusive = carry + inclusive - values
        tl.store(row_offsets + offsets, exclusive, mask=mask)
        carry = carry + tl.sum(values, axis=0)
    tl.store(row_offsets + n, carry)


@triton.jit
def _scatter_compact_kernel(
    dense_x,
    dense_q,
    counts,
    row_offsets,
    events,
    packed,
    nmax,
    LANES: tl.constexpr,
    ITEMS_PER_LANE: tl.constexpr,
):
    cell = tl.program_id(0).to(tl.int64)
    count = tl.load(counts + cell).to(tl.int64)
    dense_base = cell * nmax
    row_base = tl.load(row_offsets + cell).to(tl.int64)
    max_double = 1.7976931348623157e308

    TILE_SIZE: tl.constexpr = LANES * ITEMS_PER_LANE
    carry = count - count  # int64 scalar zero; loop-carried type stays int64
    for start in tl.range(0, count, TILE_SIZE):
        slot = start + tl.arange(0, TILE_SIZE).to(tl.int64)
        mask = slot < count
        p = dense_base + slot
        x = tl.load(dense_x + p, mask=mask, other=0.0)
        q = tl.load(dense_q + p, mask=mask, other=0.0)
        finite = (
            (x == x)
            & (q == q)
            & (tl.abs(x) <= max_double)
            & (tl.abs(q) <= max_double)
        )
        valid = mask & finite & (x * q != 0.0)
        flags = valid.to(tl.int64)
        local_rank = tl.cumsum(flags, axis=0) - flags
        row = row_base + carry + local_rank
        tl.store(events + row * 2 + 0, x, mask=valid)
        tl.store(events + row * 2 + 1, q, mask=valid)
        tl.store(packed + row, p, mask=valid)
        carry = carry + tl.sum(flags, axis=0)


@triton.jit
def _interpolation_vjp_kernel(
    grad_events,
    packed,
    row_offsets,
    bins,
    cdf,
    u,
    indices,
    grad_cdf,
    NX: tl.constexpr,
    NY: tl.constexpr,
    AXIS: tl.constexpr,
    Q_AXIS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LANES: tl.constexpr,
    ITEMS_PER_LANE: tl.constexpr,
):
    cell = tl.program_id(0).to(tl.int64)
    row_begin = tl.load(row_offsets + cell).to(tl.int64)
    row_end = tl.load(row_offsets + cell + 1).to(tl.int64)

    CELLS: tl.constexpr = NX * NY
    b = cell // CELLS
    within = cell - b * CELLS
    ix = within // NY
    iy = within - ix * NY
    if Q_AXIS:
        bins_base = (b * NY + iy) * K
        curve_base = ((b * NX + ix) * NY + iy) * K
    else:
        bins_base = (b * NX + ix) * K
        curve_base = ((b * NY + iy) * NX + ix) * K

    ks = tl.arange(0, BLOCK_K)
    k_mask = ks < K
    lanes = tl.arange(0, LANES).to(tl.int64)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float64)
    TILE_SIZE: tl.constexpr = LANES * ITEMS_PER_LANE

    for start in tl.range(row_begin, row_end, TILE_SIZE):
        for item in tl.static_range(ITEMS_PER_LANE):
            rows = start + lanes + item * LANES
            row_mask = rows < row_end
            p = tl.load(packed + rows, mask=row_mask, other=0).to(tl.int64)
            j = tl.load(indices + p, mask=row_mask, other=0).to(tl.int32)
            uv = tl.load(u + p, mask=row_mask, other=0.0)

            c0 = tl.load(cdf + curve_base + j, mask=row_mask, other=0.0)
            c1 = tl.load(cdf + curve_base + j + 1, mask=row_mask, other=0.0)
            b0 = tl.load(bins + bins_base + j, mask=row_mask, other=0.0)
            b1 = tl.load(bins + bins_base + j + 1, mask=row_mask, other=0.0)

            inv_d = 1.0 / (c1 - c0 + 1e-5)
            t = uv - c0
            upstream = tl.load(
                grad_events + rows * 2 + AXIS,
                mask=row_mask,
                other=0.0,
            ) * (b1 - b0)
            left = upstream * (-inv_d + t * inv_d * inv_d)
            right = -upstream * t * inv_d * inv_d

            j2 = j[:, None]
            k2 = ks[None, :]
            contrib = tl.where(
                row_mask[:, None] & k_mask[None, :] & (j2 == k2),
                left[:, None],
                0.0,
            )
            contrib = contrib + tl.where(
                row_mask[:, None] & k_mask[None, :] & (j2 + 1 == k2),
                right[:, None],
                0.0,
            )
            acc = acc + tl.sum(contrib, axis=0)

    tl.store(grad_cdf + curve_base + ks, acc, mask=k_mask)


@triton.jit
def _cdf_vjp_kernel(
    bins,
    acceptance,
    grad_cdf,
    grad_rho,
    NX: tl.constexpr,
    NY: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    Q_AXIS: tl.constexpr,
):
    cell = tl.program_id(0).to(tl.int64)
    CELLS: tl.constexpr = NX * NY
    b = cell // CELLS
    within = cell - b * CELLS
    ix = within // NY
    iy = within - ix * NY
    k = tl.arange(0, BLOCK_K)
    mask = k < K

    if Q_AXIS:
        bins_base = (b * NY + iy) * K
        curve_base = ((b * NX + ix) * NY + iy) * K
    else:
        bins_base = (b * NX + ix) * K
        curve_base = ((b * NY + iy) * NX + ix) * K

    bv = tl.load(bins + bins_base + k, mask=mask, other=0.0)
    prev_b = tl.load(
        bins + bins_base + k - 1,
        mask=mask & (k > 0),
        other=0.0,
    )
    next_b = tl.load(
        bins + bins_base + k + 1,
        mask=mask & (k + 1 < K),
        other=0.0,
    )
    gc = tl.load(grad_cdf + curve_base + k, mask=mask, other=0.0)
    suffix = tl.cumsum(gc, axis=0, reverse=True)
    suffix_next = suffix - gc
    left = tl.where(k > 0, 0.5 * (bv - prev_b) * suffix, 0.0)
    right = tl.where(k + 1 < K, 0.5 * (next_b - bv) * suffix_next, 0.0)
    acc = tl.load(acceptance + cell).to(tl.float64)
    tl.store(grad_rho + curve_base + k, acc * (left + right), mask=mask)


@triton.jit
def _density_vjp_kernel(
    bins,
    xsec,
    norm,
    grad_rho,
    grad_xsec,
    NX: tl.constexpr,
    NY: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    Q_AXIS: tl.constexpr,
):
    cell = tl.program_id(0).to(tl.int64)
    CELLS: tl.constexpr = NX * NY
    b = cell // CELLS
    within = cell - b * CELLS
    ix = within // NY
    iy = within - ix * NY
    k = tl.arange(0, BLOCK_K)
    mask = k < K

    if Q_AXIS:
        bins_base = (b * NY + iy) * K
        curve_base = ((b * NX + ix) * NY + iy) * K
    else:
        bins_base = (b * NX + ix) * K
        curve_base = ((b * NY + iy) * NX + ix) * K

    bv = tl.load(bins + bins_base + k, mask=mask, other=0.0)
    prev_b = tl.load(
        bins + bins_base + k - 1,
        mask=mask & (k > 0),
        other=0.0,
    )
    next_b = tl.load(
        bins + bins_base + k + 1,
        mask=mask & (k + 1 < K),
        other=0.0,
    )
    xs = tl.load(xsec + curve_base + k, mask=mask, other=0.0)
    gr = tl.load(grad_rho + curve_base + k, mask=mask, other=0.0)
    n = tl.load(norm + cell)
    inv_n = 1.0 / n
    grad_norm = tl.sum(-gr * xs * inv_n * inv_n, axis=0)
    norm_weight = (
        tl.where(k > 0, 0.5 * (bv - prev_b), 0.0)
        + tl.where(k + 1 < K, 0.5 * (next_b - bv), 0.0)
    )
    gx = gr * inv_n + grad_norm * norm_weight
    tl.store(grad_xsec + curve_base + k, gx, mask=mask)


def _allocate(weights: torch.Tensor, n_events: int) -> tuple[torch.Tensor, int]:
    n = weights.numel()
    counts = torch.empty(n, device=weights.device, dtype=torch.int64)
    partials = triton.cdiv(n, _ALLOC_BLOCK)
    partial_max = torch.empty(partials, device=weights.device, dtype=torch.int64)
    _allocation_kernel[(partials,)](
        weights,
        counts,
        partial_max,
        n,
        n_events,
        BLOCK=_ALLOC_BLOCK,
    )
    result = torch.empty(1, device=weights.device, dtype=torch.int64)
    reduce_block = triton.next_power_of_2(partials)
    _allocation_reduce_kernel[(1,)](
        partial_max,
        result,
        partials,
        BLOCK=reduce_block,
    )
    return counts, int(result.item())


def _philox_uniform(
    count: int,
    seed: int,
    stream: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.empty(count, device=device, dtype=torch.float64)
    if count == 0:
        return output
    seed = int(seed) & ((1 << 64) - 1)
    stream = int(stream) & ((1 << 64) - 1)
    seed_lo = seed & 0xFFFFFFFF
    seed_hi = (seed >> 32) & 0xFFFFFFFF
    stream_lo = stream & 0xFFFFFFFF
    stream_hi = (stream >> 32) & 0xFFFFFFFF
    blocks = triton.cdiv(count, 4)
    grid = (triton.cdiv(blocks, _ALLOC_BLOCK),)
    _philox_uniform_kernel[grid](
        output,
        count,
        seed_lo,
        seed_hi,
        stream_lo,
        stream_hi,
        BLOCK=_ALLOC_BLOCK,
    )
    return output


def _forward_impl(
    x_bins: torch.Tensor,
    xsec_x: torch.Tensor,
    q_bins: torch.Tensor,
    xsec_q: torch.Tensor,
    weights: torch.Tensor,
    acceptance: torch.Tensor,
    n_events: int,
    seed: int,
    sequence: int,
    profile_regions: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    batch = x_bins.shape[0]
    nx = x_bins.shape[1]
    ny = q_bins.shape[1]
    kx = x_bins.shape[2]
    kq = q_bins.shape[2]
    cells = nx * ny
    global_cells = batch * cells

    block_kx = triton.next_power_of_2(kx)
    block_kq = triton.next_power_of_2(kq)

    with _region(profile_regions, "loits::forward"):
        with _region(profile_regions, "loits::forward::allocation"):
            counts, nmax = _allocate(weights, n_events)

        norm_x = torch.empty(global_cells, device=xsec_x.device, dtype=torch.float64)
        rho_x = torch.empty_like(xsec_x)
        with _region(profile_regions, "loits::forward::rho_x"):
            _density_kernel[(global_cells,)](
                x_bins,
                xsec_x,
                norm_x,
                rho_x,
                NX=nx,
                NY=ny,
                K=kx,
                BLOCK_K=block_kx,
                Q_AXIS=False,
            )

        norm_q = torch.empty(global_cells, device=xsec_q.device, dtype=torch.float64)
        rho_q = torch.empty_like(xsec_q)
        with _region(profile_regions, "loits::forward::rho_q2"):
            _density_kernel[(global_cells,)](
                q_bins,
                xsec_q,
                norm_q,
                rho_q,
                NX=nx,
                NY=ny,
                K=kq,
                BLOCK_K=block_kq,
                Q_AXIS=True,
            )

        cdf_x = torch.empty_like(xsec_x)
        with _region(profile_regions, "loits::forward::cdf_x"):
            _cdf_kernel[(global_cells,)](
                x_bins,
                rho_x,
                acceptance,
                cdf_x,
                NX=nx,
                NY=ny,
                K=kx,
                BLOCK_K=block_kx,
                Q_AXIS=False,
            )

        cdf_q = torch.empty_like(xsec_q)
        with _region(profile_regions, "loits::forward::cdf_q2"):
            _cdf_kernel[(global_cells,)](
                q_bins,
                rho_q,
                acceptance,
                cdf_q,
                NX=nx,
                NY=ny,
                K=kq,
                BLOCK_K=block_kq,
                Q_AXIS=True,
            )

        total_slots = global_cells * nmax
        with _region(profile_regions, "loits::forward::random_x"):
            u_x = _philox_uniform(total_slots, seed, sequence * 2, xsec_x.device)
        with _region(profile_regions, "loits::forward::random_q2"):
            u_q = _philox_uniform(total_slots, seed, sequence * 2 + 1, xsec_q.device)

        dense_x = torch.empty(total_slots, device=xsec_x.device, dtype=torch.float64)
        interval_x = torch.empty(total_slots, device=xsec_x.device, dtype=torch.int16)
        dense_q = torch.empty(total_slots, device=xsec_q.device, dtype=torch.float64)
        interval_q = torch.empty(total_slots, device=xsec_q.device, dtype=torch.int16)

        if total_slots:
            grid = (triton.cdiv(total_slots, _SLOT_BLOCK),)
            with _region(profile_regions, "loits::forward::interpolation_x"):
                _interpolate_kernel[grid](
                    x_bins,
                    cdf_x,
                    u_x,
                    counts,
                    dense_x,
                    interval_x,
                    total_slots,
                    nmax,
                    NX=nx,
                    NY=ny,
                    K=kx,
                    BLOCK_K=block_kx,
                    Q_AXIS=False,
                    BLOCK=_SLOT_BLOCK,
                )
            with _region(profile_regions, "loits::forward::interpolation_q2"):
                _interpolate_kernel[grid](
                    q_bins,
                    cdf_q,
                    u_q,
                    counts,
                    dense_q,
                    interval_q,
                    total_slots,
                    nmax,
                    NX=nx,
                    NY=ny,
                    K=kq,
                    BLOCK_K=block_kq,
                    Q_AXIS=True,
                    BLOCK=_SLOT_BLOCK,
                )

        with _region(profile_regions, "loits::forward::stream_compaction"):
            valid_counts = torch.empty(
                global_cells,
                device=xsec_x.device,
                dtype=torch.int64,
            )
            row_offsets = torch.empty(
                global_cells + 1,
                device=xsec_x.device,
                dtype=torch.int64,
            )
            if total_slots:
                _count_valid_kernel[(global_cells,)](
                    dense_x,
                    dense_q,
                    counts,
                    valid_counts,
                    nmax,
                    LANES=_CELL_LANES,
                    ITEMS_PER_LANE=_ITEMS_PER_LANE,
                )
                _scan_counts_kernel[(1,)](
                    valid_counts,
                    row_offsets,
                    global_cells,
                    BLOCK=_SCAN_BLOCK,
                )
                valid = int(row_offsets[-1].item())
            else:
                row_offsets.zero_()
                valid = 0

            events = torch.empty(
                (valid, 2),
                device=xsec_x.device,
                dtype=torch.float64,
            )
            packed = torch.empty(
                valid,
                device=xsec_x.device,
                dtype=torch.int64,
            )
            if valid:
                _scatter_compact_kernel[(global_cells,)](
                    dense_x,
                    dense_q,
                    counts,
                    row_offsets,
                    events,
                    packed,
                    nmax,
                    LANES=_CELL_LANES,
                    ITEMS_PER_LANE=_ITEMS_PER_LANE,
                )

    state = (
        norm_x,
        norm_q,
        cdf_x,
        cdf_q,
        u_x,
        u_q,
        interval_x,
        interval_q,
        packed,
        row_offsets,
    )
    return events, state


def _backward_impl(
    grad_events: torch.Tensor,
    x_bins: torch.Tensor,
    xsec_x: torch.Tensor,
    q_bins: torch.Tensor,
    xsec_q: torch.Tensor,
    acceptance: torch.Tensor,
    state: tuple[torch.Tensor, ...],
    profile_regions: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        norm_x,
        norm_q,
        cdf_x,
        cdf_q,
        u_x,
        u_q,
        interval_x,
        interval_q,
        packed,
        row_offsets,
    ) = state

    batch = x_bins.shape[0]
    nx = x_bins.shape[1]
    ny = q_bins.shape[1]
    kx = x_bins.shape[2]
    kq = q_bins.shape[2]
    global_cells = batch * nx * ny
    block_kx = triton.next_power_of_2(kx)
    block_kq = triton.next_power_of_2(kq)

    with _region(profile_regions, "loits::backward"):
        grad_cdf_x = torch.empty_like(xsec_x)
        with _region(profile_regions, "loits::backward::interpolation_x"):
            _interpolation_vjp_kernel[(global_cells,)](
                grad_events,
                packed,
                row_offsets,
                x_bins,
                cdf_x,
                u_x,
                interval_x,
                grad_cdf_x,
                NX=nx,
                NY=ny,
                AXIS=0,
                Q_AXIS=False,
                K=kx,
                BLOCK_K=block_kx,
                LANES=_CELL_LANES,
                ITEMS_PER_LANE=_ITEMS_PER_LANE,
            )

        grad_cdf_q = torch.empty_like(xsec_q)
        with _region(profile_regions, "loits::backward::interpolation_q2"):
            _interpolation_vjp_kernel[(global_cells,)](
                grad_events,
                packed,
                row_offsets,
                q_bins,
                cdf_q,
                u_q,
                interval_q,
                grad_cdf_q,
                NX=nx,
                NY=ny,
                AXIS=1,
                Q_AXIS=True,
                K=kq,
                BLOCK_K=block_kq,
                LANES=_CELL_LANES,
                ITEMS_PER_LANE=_ITEMS_PER_LANE,
            )

        grad_rho_x = torch.empty_like(xsec_x)
        with _region(profile_regions, "loits::backward::cdf_x"):
            _cdf_vjp_kernel[(global_cells,)](
                x_bins,
                acceptance,
                grad_cdf_x,
                grad_rho_x,
                NX=nx,
                NY=ny,
                K=kx,
                BLOCK_K=block_kx,
                Q_AXIS=False,
            )

        grad_rho_q = torch.empty_like(xsec_q)
        with _region(profile_regions, "loits::backward::cdf_q2"):
            _cdf_vjp_kernel[(global_cells,)](
                q_bins,
                acceptance,
                grad_cdf_q,
                grad_rho_q,
                NX=nx,
                NY=ny,
                K=kq,
                BLOCK_K=block_kq,
                Q_AXIS=True,
            )

        grad_xsec_x = torch.empty_like(xsec_x)
        with _region(profile_regions, "loits::backward::rho_x"):
            _density_vjp_kernel[(global_cells,)](
                x_bins,
                xsec_x,
                norm_x,
                grad_rho_x,
                grad_xsec_x,
                NX=nx,
                NY=ny,
                K=kx,
                BLOCK_K=block_kx,
                Q_AXIS=False,
            )

        grad_xsec_q = torch.empty_like(xsec_q)
        with _region(profile_regions, "loits::backward::rho_q2"):
            _density_vjp_kernel[(global_cells,)](
                q_bins,
                xsec_q,
                norm_q,
                grad_rho_q,
                grad_xsec_q,
                NX=nx,
                NY=ny,
                K=kq,
                BLOCK_K=block_kq,
                Q_AXIS=True,
            )

    return grad_xsec_x, grad_xsec_q


class _TritonLOITSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_bins,
        xsec_x,
        q_bins,
        xsec_q,
        weights,
        acceptance,
        n_events,
        seed,
        sequence,
        profile_regions,
    ):
        profile_regions = bool(profile_regions)
        with _device_guard(xsec_x.device):
            events, state = _forward_impl(
                x_bins,
                xsec_x,
                q_bins,
                xsec_q,
                weights,
                acceptance,
                int(n_events),
                int(seed),
                int(sequence),
                profile_regions,
            )
        ctx.profile_regions = profile_regions
        ctx.save_for_backward(
            x_bins,
            xsec_x,
            q_bins,
            xsec_q,
            acceptance,
            *state,
        )
        return events

    @staticmethod
    def backward(ctx, grad_events):
        (
            x_bins,
            xsec_x,
            q_bins,
            xsec_q,
            acceptance,
            *state,
        ) = ctx.saved_tensors
        with _device_guard(grad_events.device):
            grad_xsec_x, grad_xsec_q = _backward_impl(
                grad_events.contiguous(),
                x_bins,
                xsec_x,
                q_bins,
                xsec_q,
                acceptance,
                state,
                ctx.profile_regions,
            )
        return (
            None,
            grad_xsec_x,
            None,
            grad_xsec_q,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class TritonLOITS(nn.Module):
    def __init__(
        self,
        device: torch.device | str = "cuda",
        compile: bool = False,
        profile_regions: bool = False,
        epsilon: float = _EPSILON,
    ):
        super().__init__()
        self.device: torch.device = torch.device(device)
        ok, reason = availability(self.device)
        if not ok:
            raise RuntimeError(reason)
        if epsilon != _EPSILON:
            raise ValueError("The Triton backend currently uses the LOITS epsilon 1e-5")
        self.profile_regions: bool = bool(profile_regions)
        self.seed: int = torch.initial_seed()
        self.sequence: int = 0

    @staticmethod
    def _validate(theory_outputs: tuple[torch.Tensor, ...]) -> None:
        x_bins, xsec_x, q_bins, xsec_q, weights, acceptance = theory_outputs[:6]
        floating = (x_bins, xsec_x, q_bins, xsec_q, weights)
        if any(t.dtype != torch.float64 for t in floating):
            raise TypeError("Triton LOITS requires float64 theory tensors")
        if acceptance.dtype != torch.bool:
            raise TypeError("Triton LOITS requires boolean acceptance")
        if any(not t.is_contiguous() for t in theory_outputs[:6]):
            raise ValueError("Triton LOITS requires contiguous theory tensors")

    def forward(self, theory_outputs: tuple[torch.Tensor, ...], n_events: int) -> torch.Tensor:
        self._validate(theory_outputs)
        x_bins, xsec_x, q_bins, xsec_q, weights, acceptance = theory_outputs[:6]
        if xsec_x.device.type != self.device.type or (
            self.device.index is not None and xsec_x.device.index != self.device.index
        ):
            raise ValueError(
                f"Triton LOITS configured for {self.device}, got tensors on {xsec_x.device}"
            )
        sequence = self.sequence
        self.sequence += 1
        return _TritonLOITSFunction.apply(
            x_bins,
            xsec_x,
            q_bins,
            xsec_q,
            weights,
            acceptance,
            n_events,
            self.seed,
            sequence,
            self.profile_regions,
        )
