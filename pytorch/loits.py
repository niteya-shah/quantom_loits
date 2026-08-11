import torch
from torch import nn


torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True


class EventAllocation(nn.Module):
    def forward(self, weights, n_events):
        counts = torch.abs(weights.detach() * n_events).to(torch.int64)
        slots = torch.arange(counts.max(), device=weights.device)
        return counts.reshape(counts.shape[0], -1, 1) > slots


class Density(nn.Module):
    def forward(self, bins, xsec):
        norm = torch.trapezoid(xsec, bins[:, None, :, :], dim=-1)
        return xsec / norm[..., None]


class CDF(nn.Module):
    def forward(self, bins, rho, acceptance, transpose_acceptance):
        cdf = torch.nn.functional.pad(
            torch.cumulative_trapezoid(rho, bins[:, None, :, :], dim=-1),
            (1, 0),
        )
        acc = acceptance.transpose(1, 2) if transpose_acceptance else acceptance
        return cdf * acc[..., None]


class FlattenObservable(nn.Module):
    def forward(self, bins, cdf, grid0, grid1, transpose_cdf):
        batch = cdf.shape[0]
        points = cdf.shape[-1]
        cdf_flat = cdf.transpose(1, 2) if transpose_cdf else cdf
        cdf_flat = cdf_flat.reshape(batch, grid0 * grid1, points)
        if transpose_cdf:
            bins_flat = bins[:, :, None, :].expand(-1, -1, grid1, -1)
        else:
            bins_flat = bins[:, None, :, :].expand(-1, grid0, -1, -1)
        return cdf_flat, bins_flat.reshape(batch, grid0 * grid1, points)


class RandomSamples(nn.Module):
    def forward(self, mask, reference):
        return torch.rand(mask.shape, device=reference.device, dtype=torch.float32)


class LinearInterpolation(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, u, cdf, bins, mask):
        m = (bins[..., 1:] - bins[..., :-1]) / (
            cdf[..., 1:] - cdf[..., :-1] + self.epsilon
        )
        b = bins[..., :-1] - m * cdf[..., :-1]
        indices = (u[..., None] >= cdf[..., None, :]).sum(dim=-1) - 1
        indices = indices.clamp(0, m.shape[-1] - 1)
        values = torch.gather(m, -1, indices) * u + torch.gather(b, -1, indices)
        return values * mask


class StreamCompaction(nn.Module):
    def forward(self, x, q2, mask):
        events = torch.stack((x, q2), dim=-1)
        valid = mask & torch.isfinite(events).all(dim=-1)
        valid = valid & (events[..., 0] * events[..., 1] != 0)
        return events[valid]


class TorchLOITSCore(nn.Module):
    region_names = (
        "allocation",
        "rho_x",
        "rho_q2",
        "cdf_x",
        "cdf_q2",
        "flatten_x",
        "flatten_q2",
        "random_x",
        "random_q2",
        "interpolation_x",
        "interpolation_q2",
        "stream_compaction",
    )

    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.allocation = EventAllocation()
        self.rho_x = Density()
        self.rho_q2 = Density()
        self.cdf_x = CDF()
        self.cdf_q2 = CDF()
        self.flatten_x = FlattenObservable()
        self.flatten_q2 = FlattenObservable()
        self.random_x = RandomSamples()
        self.random_q2 = RandomSamples()
        self.interpolation_x = LinearInterpolation(epsilon)
        self.interpolation_q2 = LinearInterpolation(epsilon)
        self.stream_compaction = StreamCompaction()

    def forward(self, theory_outputs, n_events):
        x_bins, xsec_x, q2_bins, xsec_q2, weights, acceptance = theory_outputs[:6]
        grid0, grid1 = weights.shape[1], weights.shape[2]

        mask = self.allocation(weights, n_events)

        rho_x = self.rho_x(x_bins, xsec_x)
        rho_q2 = self.rho_q2(q2_bins, xsec_q2)
        cdf_x = self.cdf_x(x_bins, rho_x, acceptance, True)
        cdf_q2 = self.cdf_q2(q2_bins, rho_q2, acceptance, False)

        cdf_x, bins_x = self.flatten_x(x_bins, cdf_x, grid0, grid1, True)
        cdf_q2, bins_q2 = self.flatten_q2(q2_bins, cdf_q2, grid0, grid1, False)

        u_x = self.random_x(mask, xsec_x)
        u_q2 = self.random_q2(mask, xsec_q2)
        x = self.interpolation_x(u_x, cdf_x, bins_x, mask)
        q2 = self.interpolation_q2(u_q2, cdf_q2, bins_q2, mask)
        return self.stream_compaction(x, q2, mask)

    def compile_regions(self):
        for name in self.region_names:
            setattr(self, name, torch.compile(getattr(self, name)))
        return self


class TorchLOITS(nn.Module):
    def __init__(self, device="cpu", compile=True, profile_regions=False, epsilon=1e-5):
        super().__init__()
        self.device = torch.device(device)
        core = TorchLOITSCore(epsilon)
        if compile and profile_regions:
            core.compile_regions()
            self.core = core
        elif compile:
            self.core = torch.compile(core)
        else:
            self.core = core

    def forward(self, theory_outputs, n_events):
        return self.core(theory_outputs, n_events)
