from contextlib import nullcontext

import torch
from torch import nn
from torch.profiler import record_function

from loits import LOITS
from .theory import TorchProxyTheoryLite


class Discriminator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, events):
        return self.net(events)


class GANTrainer:
    def __init__(
        self,
        backend="torch",
        device="cpu",
        n_events=10000,
        grid_size=10,
        compile=True,
        profile_regions=False,
        seed=0,
    ):
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        self.backend = backend
        self.device = torch.device(device)
        self.n_events = n_events
        self.profile_regions = profile_regions
        theory_config = {
            "n_points_x": grid_size,
            "n_points_y": grid_size,
            "n_cdf_points_x": 10,
            "n_cdf_points_y": 10,
            "average": False,
        }
        self.theory = TorchProxyTheoryLite(theory_config, self.device)
        self.sampler = LOITS(
            backend=backend,
            device=self.device,
            compile=compile and backend == "torch" and profile_regions,
            profile_regions=profile_regions,
        )
        self.discriminator = Discriminator().to(device=self.device, dtype=torch.float64)
        self.params = nn.Parameter(
            torch.tensor([[0.45, 0.35, 0.35, 0.55, 0.35]], device=self.device, dtype=torch.float64)
        )
        self.loss = nn.BCEWithLogitsLoss()

        if compile and backend == "torch" and profile_regions:
            self.theory = torch.compile(self.theory)
            self.discriminator = torch.compile(self.discriminator)

        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        self.g_optimizer = torch.optim.Adam([self.params], lr=1e-3)

        true_params = torch.tensor(
            [[0.67, 0.20, 0.23, 0.67, 0.50]], device=self.device, dtype=torch.float64
        )
        with torch.no_grad():
            self.real_events = self.sampler(self.theory(true_params), self.n_events).detach()

        self.generate = self._generate
        self.discriminator_loss = self._discriminator_loss
        self.generator_loss = self._generator_loss
        if compile and backend == "torch" and not profile_regions:
            self.generate = torch.compile(self._generate)
            self.discriminator_loss = torch.compile(self._discriminator_loss)
            self.generator_loss = torch.compile(self._generator_loss)

    def _generate(self, params):
        return self.sampler(self.theory(params), self.n_events)

    def _discriminator_loss(self, real_events, fake_events):
        real = self.discriminator(real_events)
        fake = self.discriminator(fake_events)
        return self.loss(real, torch.ones_like(real)) + self.loss(fake, torch.zeros_like(fake))

    def _generator_loss(self, params):
        fake = self._generate(params)
        logits = self.discriminator(fake)
        return self.loss(logits, torch.ones_like(logits))

    def _region(self, name):
        return record_function(name) if self.profile_regions else nullcontext()

    def reset_rng(self):
        torch.manual_seed(self.seed)
        impl = self.sampler.impl
        if hasattr(impl, "seed"):
            impl.seed = self.seed
        if hasattr(impl, "sequence"):
            impl.sequence = 0

    def step(self):
        with self._region("gan::discriminator_step"):
            self.d_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = self.generate(self.params)
            d_loss = self.discriminator_loss(self.real_events, fake.detach())
            d_loss.backward()
            self.d_optimizer.step()

        with self._region("gan::generator_step"):
            self.g_optimizer.zero_grad(set_to_none=True)
            g_loss = self.generator_loss(self.params)
            grad = torch.autograd.grad(g_loss, self.params)[0]
            self.params.grad = grad
            self.g_optimizer.step()
        return d_loss.detach(), g_loss.detach()
