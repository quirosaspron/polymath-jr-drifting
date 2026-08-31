"""Paired experiment orchestration, checkpoint evaluation, and artifacts.

Training loops stay outside this module. A trainer only needs to consume its
paired inputs and call :class:`EvolutionRecorder` at chosen steps.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .diagnostics import discrepancy_auc, fixed_budget_discrepancy, steps_to_threshold


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducibility and comparison settings, separate from optimizer choices."""

    dataset: str
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    drift_steps: int = 200
    eval_every: int = 5
    primary_metric: str = "signal_swd"
    epsilon: float | None = None
    fixed_budget: int = 200
    output_dir: Path = Path("artifacts")
    device: str = "auto"
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset or not self.methods or not self.seeds:
            raise ValueError("dataset, methods, and seeds must be non-empty")
        if self.drift_steps < 1 or self.eval_every < 1:
            raise ValueError("drift_steps and eval_every must be positive")
        if not 0 <= self.fixed_budget <= self.drift_steps:
            raise ValueError("fixed_budget must be inside the drift range")
        if self.epsilon is not None and self.epsilon < 0:
            raise ValueError("epsilon must be non-negative or None")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class EvaluationSnapshot:
    """One external evaluation event during evolution."""

    step: int
    metrics: dict[str, float]
    sample_path: str | None = None
    checkpoint_path: str | None = None


@dataclass
class RunResult:
    """Scalar curves and artifacts from one method/seed run."""

    method: str
    seed: int
    steps: list[int]
    metrics: dict[str, list[float]]
    elapsed_seconds: float
    snapshots: list[EvaluationSnapshot] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """All paired runs and a method-level convergence summary."""

    config: ExperimentConfig
    runs: list[RunResult]
    summary: list[dict[str, Any]]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, Torch, and available CUDA devices."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic_algorithms
        torch.backends.cudnn.benchmark = not deterministic_algorithms


def clone_state_dict(model: Any) -> dict[str, torch.Tensor]:
    """Clone a CPU initialization that can be loaded by every paired method."""

    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_initial_state(model: Any, state: Mapping[str, torch.Tensor]) -> None:
    """Load an identical initialization without sharing parameter storage."""

    model.load_state_dict({name: value.clone() for name, value in state.items()})


def save_model_checkpoint(
    path: str | Path,
    *,
    model: Any,
    step: int,
    optimizer: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save model/optimizer state and the step needed to resume or inspect it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, destination)
    return destination


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    return value


class EvolutionRecorder:
    """Evaluate fixed queries and optionally persist samples during training.

    ``generate_fn(state, fixed_queries)`` is deliberately supplied by the
    notebook/trainer, so this recorder works for particles, latent generators,
    decoded images, and conditional models without knowing their architecture.
    ``evaluator_fn(generated)`` must return only scalar external metrics.
    """

    def __init__(
        self,
        *,
        fixed_queries: Any,
        generate_fn: Callable[[Any, Any], Any],
        evaluator_fn: Callable[[Any], Mapping[str, float]],
        every: int,
        output_dir: str | Path | None = None,
        save_samples: bool = True,
    ) -> None:
        if every < 1:
            raise ValueError("every must be positive")
        self.fixed_queries = _cpu_snapshot(fixed_queries)
        self.generate_fn = generate_fn
        self.evaluator_fn = evaluator_fn
        self.every = every
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.save_samples = save_samples
        self.steps: list[int] = []
        self.metrics: dict[str, list[float]] = {}
        self.snapshots: list[EvaluationSnapshot] = []
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_record(self, step: int, *, force: bool = False) -> bool:
        return force or step == 0 or step % self.every == 0

    def record(
        self,
        step: int,
        state: Any,
        *,
        force: bool = False,
        extra_metrics: Mapping[str, float] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> EvaluationSnapshot | None:
        """Generate from fixed queries, evaluate, and append one curve point."""

        if not self.should_record(step, force=force):
            return None
        was_training = getattr(state, "training", None)
        if was_training is not None and hasattr(state, "eval"):
            state.eval()
        try:
            with torch.no_grad():
                generated = self.generate_fn(state, self.fixed_queries)
        finally:
            if was_training and hasattr(state, "train"):
                state.train()
        values = {
            name: float(value)
            for name, value in self.evaluator_fn(generated).items()
        }
        values.update({name: float(value) for name, value in (extra_metrics or {}).items()})
        self.steps.append(int(step))
        previous_count = len(self.steps) - 1
        for name in set(self.metrics) | set(values):
            if name not in self.metrics:
                self.metrics[name] = [float("nan")] * previous_count
            self.metrics[name].append(values.get(name, float("nan")))

        sample_path: str | None = None
        if self.output_dir is not None and self.save_samples:
            destination = self.output_dir / f"samples_step_{step:06d}.pt"
            torch.save(_cpu_snapshot(generated), destination)
            sample_path = str(destination)
        snapshot = EvaluationSnapshot(
            step=int(step),
            metrics=values,
            sample_path=sample_path,
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def to_run_result(
        self, *, method: str, seed: int, elapsed_seconds: float
    ) -> RunResult:
        return RunResult(
            method=method,
            seed=seed,
            steps=list(self.steps),
            metrics={name: list(values) for name, values in self.metrics.items()},
            elapsed_seconds=float(elapsed_seconds),
            snapshots=list(self.snapshots),
        )


def timed_run(run_fn: Callable[[], RunResult]) -> RunResult:
    """Run one experiment and fill elapsed time when the trainer leaves it zero."""

    started = time.perf_counter()
    result = run_fn()
    elapsed = time.perf_counter() - started
    if result.elapsed_seconds == 0:
        result.elapsed_seconds = elapsed
    return result


def summarize_runs(
    runs: Sequence[RunResult],
    *,
    metric_name: str,
    epsilon: float | None,
    fixed_budget: int,
) -> list[dict[str, Any]]:
    """Summarize final, AUC, fixed-budget, and optional hitting-time metrics."""

    rows: list[dict[str, Any]] = []
    for method in sorted({run.method for run in runs}):
        method_runs = [run for run in runs if run.method == method]
        finals: list[float] = []
        aucs: list[float] = []
        budget_values: list[float] = []
        hitting_times: list[float] = []
        for run in method_runs:
            if metric_name not in run.metrics:
                raise KeyError(f"{metric_name!r} missing from {method}/{run.seed}")
            curve = run.metrics[metric_name]
            finals.append(float(curve[-1]))
            aucs.append(discrepancy_auc(run.steps, curve))
            budget_values.append(
                fixed_budget_discrepancy(run.steps, curve, budget=fixed_budget)
            )
            if epsilon is not None:
                hitting_times.append(
                    steps_to_threshold(run.steps, curve, epsilon=epsilon)
                )

        def mean_std(values: Sequence[float]) -> tuple[float, float]:
            array = np.asarray(values, dtype=float)
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                return float("inf"), float("nan")
            std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            return float(finite.mean()), std

        final_mean, final_std = mean_std(finals)
        auc_mean, auc_std = mean_std(aucs)
        budget_mean, budget_std = mean_std(budget_values)
        row: dict[str, Any] = {
            "method": method,
            "n_seeds": len(method_runs),
            "final_mean": final_mean,
            "final_std": final_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "fixed_budget_mean": budget_mean,
            "fixed_budget_std": budget_std,
        }
        if epsilon is not None:
            hitting_mean, hitting_std = mean_std(hitting_times)
            row.update(
                {
                    "steps_to_epsilon_mean": hitting_mean,
                    "steps_to_epsilon_std": hitting_std,
                    "threshold_successes": int(np.isfinite(hitting_times).sum()),
                }
            )
        rows.append(row)
    return rows


RunFunction = Callable[[ExperimentConfig, str, int, Any], RunResult]


def run_comparison(
    config: ExperimentConfig,
    *,
    run_one: RunFunction,
    paired_factory: Callable[[int], Any] | None = None,
) -> ComparisonResult:
    """Run each method with the same per-seed paired object.

    The user-authored ``run_one`` owns optimization. This function owns ordering,
    seeding, paired-input reuse, and aggregation.
    """

    runs: list[RunResult] = []
    for seed in config.seeds:
        paired_inputs = None if paired_factory is None else paired_factory(seed)
        for method in config.methods:
            seed_everything(seed)
            result = timed_run(
                lambda method=method, seed=seed: run_one(
                    config, method, seed, paired_inputs
                )
            )
            if result.method != method or result.seed != seed:
                raise ValueError("run_one returned a mismatched method or seed")
            runs.append(result)
    summary = summarize_runs(
        runs,
        metric_name=config.primary_metric,
        epsilon=config.epsilon,
        fixed_budget=config.fixed_budget,
    )
    return ComparisonResult(config=config, runs=runs, summary=summary)


def save_run_artifacts(result: RunResult, *, output_dir: str | Path) -> dict[str, str]:
    """Save one run's scalar curves and snapshot manifest as JSON."""

    destination = Path(output_dir) / result.method / f"seed_{result.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "run.json"
    payload = {
        "method": result.method,
        "seed": result.seed,
        "steps": result.steps,
        "metrics": result.metrics,
        "elapsed_seconds": result.elapsed_seconds,
        "snapshots": [asdict(snapshot) for snapshot in result.snapshots],
        "artifacts": result.artifacts,
    }
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    result.artifacts["run_json"] = str(metrics_path)
    return dict(result.artifacts)


def save_comparison(result: ComparisonResult, *, output_dir: str | Path | None = None) -> Path:
    """Save config, summary, and every run manifest."""

    destination = Path(output_dir or result.config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for run in result.runs:
        save_run_artifacts(run, output_dir=destination)
    path = destination / "comparison.json"
    path.write_text(
        json.dumps(
            _json_safe({"config": result.config.to_dict(), "summary": result.summary}),
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_run_result(path: str | Path) -> RunResult:
    """Load a run manifest without rerunning training."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshots = [EvaluationSnapshot(**item) for item in payload.pop("snapshots", [])]
    return RunResult(snapshots=snapshots, **payload)


__all__ = [
    "ComparisonResult",
    "EvaluationSnapshot",
    "EvolutionRecorder",
    "ExperimentConfig",
    "RunResult",
    "clone_state_dict",
    "load_initial_state",
    "load_run_result",
    "run_comparison",
    "save_comparison",
    "save_model_checkpoint",
    "save_run_artifacts",
    "seed_everything",
    "summarize_runs",
    "timed_run",
]
