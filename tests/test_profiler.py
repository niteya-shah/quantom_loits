from pytorch.gan import GANTrainer
from pytorch.profiler import RegionHooks, TrainingProfiler


class CountingTrainer:
    def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0

    def reset_rng(self):
        self.reset_calls += 1

    def step(self):
        self.step_calls += 1


def test_measure_reseeds_each_step_without_recreating_trainer():
    trainer = CountingTrainer()
    profiler = TrainingProfiler("cpu")
    samples = profiler.measure(trainer, warmup=3, iterations=10)
    assert len(samples) == 10
    assert trainer.reset_calls == 13
    assert trainer.step_calls == 13


def test_profiler_records_training_and_loits_regions():
    trainer = GANTrainer(
        backend="torch",
        device="cpu",
        n_events=20,
        grid_size=3,
        compile=False,
        profile_regions=True,
    )
    hooks = RegionHooks(trainer.sampler.impl)
    profiler = TrainingProfiler("cpu")
    prof = profiler.run(trainer, warmup=0, iterations=1)
    hooks.close()

    names = {event.name for event in prof.events()}
    assert "gan::training_iteration" in names
    assert "gan::discriminator_step" in names
    assert "gan::generator_step" in names
    assert "loits::forward::rho_x" in names
    assert "loits::forward::stream_compaction" in names
    assert "loits::backward::interpolation_x" in names
