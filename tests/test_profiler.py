from pytorch.gan import GANTrainer
from pytorch.profiler import RegionHooks, TrainingProfiler


class CountingTrainer:
    def __init__(self):
        self.state = 0
        self.snapshot_calls = 0
        self.restore_calls = 0
        self.step_calls = 0
        self.step_states = []

    def snapshot_state(self):
        self.snapshot_calls += 1
        return self.state

    def restore_state(self, state):
        self.restore_calls += 1
        self.state = state

    def step(self):
        self.step_calls += 1
        self.step_states.append(self.state)
        self.state += 1


def test_measure_restores_initial_state_before_every_step():
    trainer = CountingTrainer()
    profiler = TrainingProfiler("cpu")
    samples = profiler.measure(trainer, warmup=3, iterations=10)
    assert len(samples) == 10
    assert trainer.snapshot_calls == 1
    assert trainer.restore_calls == 13
    assert trainer.step_calls == 13
    assert trainer.step_states == [0] * 13


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
