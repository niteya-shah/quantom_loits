import torch
from torch import nn


class TorchProxyTheoryLite(nn.Module):
    def __init__(self, config, device="cpu", dtype=torch.float64):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_parameters = config.get("nParameters", 5)
        self.n_cdf_points_x = config.get("n_cdf_points_x", 10)
        self.n_cdf_points_y = config.get("n_cdf_points_y", 10)
        self.n_points_x = config.get("n_points_x", 10)
        self.n_points_y = config.get("n_points_y", 10)
        self.x_min = config.get("x_min", 0.001)
        self.x_max = config.get("x_max", 0.999)
        self.y_min = config.get("y_min", 0.001)
        self.y_max = config.get("y_max", 0.999)
        self.average = config.get("average", False)
        self.register_buffer(
            "parmin",
            torch.as_tensor(
                config.get("parmin", [-0.5, 2.75, 0.0, 3.0, 0.0]),
                device=self.device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "parmax",
            torch.as_tensor(
                config.get("parmax", [1.0, 4.0, 1.3, 4.5, 1.5]),
                device=self.device,
                dtype=dtype,
            ),
        )

    def get_xsection(self, p, x, y):
        rescaled_p = p * (self.parmax - self.parmin) + self.parmin
        x_dep = torch.pow(x, rescaled_p[0]) * torch.pow((1.0 - x), rescaled_p[1])
        y_dep = torch.pow(y, rescaled_p[2]) * torch.pow((1.0 - y), rescaled_p[3])
        if self.n_parameters > 4:
            return x_dep * y_dep * (1.0 + rescaled_p[4] * (x * y))
        return x_dep * y_dep * (x * y)

    def gen_xy_grid(self):
        ly = torch.linspace(
            self.y_min, self.y_max, self.n_points_y, device=self.device, dtype=self.dtype
        )
        dly = ly[1:] - ly[:-1]
        lymid = 0.5 * (ly[1:] + ly[:-1])
        lymax = ly[1:]
        lymin = ly[:-1]

        lx = torch.linspace(
            self.x_min, self.x_max, self.n_points_x, device=self.device, dtype=self.dtype
        )
        dlx = lx[1:] - lx[:-1]
        lxmid = 0.5 * (lx[1:] + lx[:-1])
        lxmax = lx[1:]
        lxmin = lx[:-1]

        lxmid, lymid = torch.meshgrid(lxmid, lymid, indexing="ij")
        lxmin, lymin = torch.meshgrid(lxmin, lymin, indexing="ij")
        lxmax, lymax = torch.meshgrid(lxmax, lymax, indexing="ij")
        dlx, dly = torch.meshgrid(dlx, dly, indexing="ij")
        return {
            "Lxmid": lxmid,
            "Lxmin": lxmin,
            "Lxmax": lxmax,
            "Lymid": lymid,
            "Lymin": lymin,
            "Lymax": lymax,
            "dLx": dlx,
            "dLy": dly,
            "acc": torch.ones(dlx.size(), device=self.device, dtype=torch.bool),
        }

    def compute_weights(self, p):
        results = self.gen_xy_grid()
        x_avg = results["Lxmid"]
        y_avg = results["Lymid"]
        diff_xsec = self.get_xsection(p, x_avg, y_avg)
        integrand = (
            diff_xsec
            * (x_avg * results["dLx"])
            * (y_avg * results["dLy"])
            * results["acc"]
        )
        total_xsec = torch.sum(integrand)
        results["total_xsec"] = total_xsec
        results["weights"] = integrand / total_xsec
        return results

    def compute_xsec_on_grid(self, p):
        results = self.compute_weights(p)
        grid = results["Lxmin"].shape

        u = torch.linspace(0, 1, self.n_cdf_points_x, device=self.device, dtype=self.dtype)
        x = (
            results["Lxmin"][:, 0].reshape(-1, 1)
            + u * results["dLx"][:, 0].reshape(-1, 1)
        ).flatten()
        xbins = x.reshape(-1, self.n_cdf_points_x)
        y = results["Lymin"].T[:, 0].flatten()
        y, x = torch.meshgrid(y, x, indexing="ij")
        xsec_x = self.get_xsection(p, x, y).reshape(grid[1], -1, self.n_cdf_points_x)

        u = torch.linspace(0, 1, self.n_cdf_points_y, device=self.device, dtype=self.dtype)
        y = (
            results["Lymin"].T[:, 0].reshape(-1, 1)
            + u * results["dLy"].T[:, 0].reshape(-1, 1)
        ).flatten()
        ybins = y.reshape(-1, self.n_cdf_points_y)
        x = results["Lxmin"][:, 0].flatten()
        x, y = torch.meshgrid(x, y, indexing="ij")
        xsec_y = self.get_xsection(p, x, y).reshape(grid[0], -1, self.n_cdf_points_y)

        return (
            xbins,
            xsec_x,
            ybins,
            xsec_y,
            results["weights"],
            results["acc"],
            results["total_xsec"],
        )

    def forward(self, params):
        outputs = torch.vmap(self.compute_xsec_on_grid, randomness="different")(params)
        if not self.average:
            return outputs
        xbins, xsec_x, ybins, xsec_y, weights, acceptance, total_xsec = outputs
        return (
            xbins.mean(0, keepdim=True),
            xsec_x.mean(0, keepdim=True),
            ybins.mean(0, keepdim=True),
            xsec_y.mean(0, keepdim=True),
            weights.mean(0, keepdim=True),
            acceptance.float().mean(0, keepdim=True).to(torch.int),
            total_xsec,
        )
