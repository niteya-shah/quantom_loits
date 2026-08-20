import csv
import time
from pathlib import Path

import torch
from torch.profiler import record_function


class RegionHooks:
    def __init__(self, module, prefix="loits"):
        self.handles = []
        self.forward_contexts = {}
        self.backward_contexts = {}
        core = module.core if hasattr(module, "core") else module
        for name in core.region_names:
            region = getattr(core, name)
            self._attach(region, name, prefix)

    def _attach(self, module, name, prefix):
        key = id(module)

        def forward_pre(*_):
            ctx = record_function(f"{prefix}::forward::{name}")
            ctx.__enter__()
            self.forward_contexts.setdefault(key, []).append(ctx)

        def forward_post(*_):
            self.forward_contexts[key].pop().__exit__(None, None, None)

        def backward_pre(*_):
            ctx = record_function(f"{prefix}::backward::{name}")
            ctx.__enter__()
            self.backward_contexts.setdefault(key, []).append(ctx)

        def backward_post(*_):
            self.backward_contexts[key].pop().__exit__(None, None, None)

        self.handles.extend(
            [
                module.register_forward_pre_hook(forward_pre),
                module.register_forward_hook(forward_post),
                module.register_full_backward_pre_hook(backward_pre),
                module.register_full_backward_hook(backward_post),
            ]
        )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class TrainingProfiler:
    def __init__(self, device):
        self.device = torch.device(device)

    def activities(self):
        activities = [torch.profiler.ProfilerActivity.CPU]
        activity = getattr(torch.profiler.ProfilerActivity, self.device.type.upper(), None)
        if self.device.type != "cpu" and activity is not None:
            activities.append(activity)
        return activities

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "xpu":
            torch.xpu.synchronize(self.device)

    def restore_generator_params(self, trainer, params):
        trainer.restore_generator_params(params)
        self.synchronize()

    def measure(self, trainer, warmup=5, iterations=10):
        params = trainer.snapshot_generator_params()
        for _ in range(warmup):
            self.restore_generator_params(trainer, params)
            trainer.step()
        self.synchronize()

        samples = []
        for _ in range(iterations):
            self.restore_generator_params(trainer, params)
            start = time.perf_counter()
            trainer.step()
            self.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
        return samples

    def run(self, trainer, warmup=5, iterations=10, trace_path=None):
        params = trainer.snapshot_generator_params()
        for _ in range(warmup):
            self.restore_generator_params(trainer, params)
            trainer.step()
        self.synchronize()

        with torch.profiler.profile(
            activities=self.activities(),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            acc_events=True,
        ) as prof:
            for _ in range(iterations):
                self.restore_generator_params(trainer, params)
                with record_function("gan::training_iteration"):
                    trainer.step()
                    self.synchronize()

        if trace_path:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(path))
        return prof

    @staticmethod
    def timing_rows(samples, metadata):
        return [
            dict(metadata, region="gan::training_iteration", occurrence=occurrence, wall_ms=wall_ms)
            for occurrence, wall_ms in enumerate(samples)
        ]

    @staticmethod
    def rows(prof, metadata):
        rows = []
        occurrences = {}

        def relevant(name):
            return name.startswith("gan::") or name.startswith("loits::")

        def relevant_parent(event):
            parent = event.cpu_parent
            while parent is not None and not relevant(parent.name):
                parent = parent.cpu_parent
            return parent

        for event in prof.events():
            if not relevant(event.name):
                continue
            occurrence = occurrences.get(event.name, 0)
            occurrences[event.name] = occurrence + 1
            parent = relevant_parent(event)
            rows.append(
                dict(
                    metadata,
                    region=event.name,
                    occurrence=occurrence,
                    event_id=event.id,
                    parent_event_id=parent.id if parent is not None else "",
                    parent_region=parent.name if parent is not None else "",
                    thread_id=event.thread,
                    sequence_nr=event.sequence_nr,
                    start_us=event.time_range.start,
                    end_us=event.time_range.end,
                    cpu_ms=event.cpu_time_total / 1000.0,
                    device_ms=event.device_time_total / 1000.0,
                    self_cpu_ms=event.self_cpu_time_total / 1000.0,
                    self_device_ms=event.self_device_time_total / 1000.0,
                )
            )
        return rows

    @staticmethod
    def write_csv(path, rows):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
