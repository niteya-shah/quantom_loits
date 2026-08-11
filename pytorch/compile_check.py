import torch

from .gan import GANTrainer
from .loits import TorchLOITSCore
from .theory import TorchProxyTheoryLite


def check(name, fn, *args):
    explanation = torch._dynamo.explain(fn)(*args)
    print(f"{name}: graphs={explanation.graph_count} breaks={explanation.graph_break_count}")
    if explanation.break_reasons:
        for reason in explanation.break_reasons:
            print(reason.reason)
        raise SystemExit(1)


def main():
    cfg = {
        "n_points_x": 4,
        "n_points_y": 4,
        "n_cdf_points_x": 10,
        "n_cdf_points_y": 10,
        "average": False,
    }
    theory = TorchProxyTheoryLite(cfg, "cpu")
    params = torch.tensor(
        [[0.67, 0.20, 0.23, 0.67, 0.50]], dtype=torch.float64, requires_grad=True
    )
    outputs = theory(params)
    core = TorchLOITSCore()
    check("theory", theory, params)
    check("loits", core, outputs, 100)

    trainer = GANTrainer("torch", "cpu", 100, 4, compile=False)
    check("gan_generator_loss", trainer._generator_loss, trainer.params)
    check(
        "gan_discriminator_loss",
        trainer._discriminator_loss,
        trainer.real_events,
        trainer.real_events,
    )

    compiled = torch.compile(core, backend="aot_eager", fullgraph=True)
    events = compiled(outputs, 100)
    events.square().mean().backward()
    print("AOTAutograd backward: ok")


if __name__ == "__main__":
    main()
