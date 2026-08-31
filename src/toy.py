"""Minimal particle evolution used by the first geometry-control notebook."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .data import PairedRunInputs, ToyDriftProblem, make_toy_drift_problem
from .experiments import ComparisonResult, EvolutionRecorder, ExperimentConfig, RunResult, run_comparison
from .evaluators import make_toy_evaluator
from .field import build_xpress_cache, compute_exact_field, compute_xpress_field


def run_particle_evolution(
    *,
    problem: ToyDriftProblem,
    paired_inputs: PairedRunInputs,
    representation: Any,
    evaluator: Any,
    method: str,
    seed: int,
    temperature: float = 1.0,
    step_size: float = 0.1,
    eval_every: int = 5,
    backend: str = "exact",
    output_dir: str | Path | None = None,
) -> RunResult:
    """Evolve particles directly and record fixed external discrepancy curves.

    This is a mechanism sanity check, not the final neural-generator
    experiment. The same target-index schedule and initial particles should be
    used for every representation condition.
    """

    if step_size <= 0 or temperature <= 0:
        raise ValueError("step_size and temperature must be positive")
    if backend not in {"exact", "xpress"}:
        raise ValueError("backend must be 'exact' or 'xpress'")
    n_steps = int(paired_inputs.real_batch_indices.shape[0])
    if n_steps < 1:
        raise ValueError("paired_inputs must contain at least one step")
    if paired_inputs.real_batch_indices.ndim != 2:
        raise ValueError("real_batch_indices must have shape (steps, batch)")
    if paired_inputs.real_batch_indices.max().item() >= problem.target.shape[0]:
        raise IndexError("paired batch index exceeds target size")

    target = problem.target
    particles = problem.initial.detach().clone()
    cache = None
    if backend == "xpress":
        landmark_indices = paired_inputs.landmark_indices
        if landmark_indices.ndim > 1:
            landmark_indices = landmark_indices[0]
        cache = build_xpress_cache(
            target,
            landmark_indices=landmark_indices,
            temperature=temperature,
            representation=representation,
        )

    recorder = EvolutionRecorder(
        fixed_queries=None,
        generate_fn=lambda state, _queries: state,
        evaluator_fn=evaluator.evaluate if hasattr(evaluator, "evaluate") else evaluator,
        every=eval_every,
        output_dir=output_dir,
        save_samples=True,
    )
    started = time.perf_counter()
    recorder.record(0, particles, force=True)
    for step in range(1, n_steps + 1):
        positive_batch = target[paired_inputs.real_batch_indices[step - 1]]
        with torch.no_grad():
            if backend == "exact":
                field = compute_exact_field(
                    particles,
                    positive_batch,
                    temperature=temperature,
                    representation=representation,
                )
            else:
                assert cache is not None
                field = compute_xpress_field(
                    particles,
                    cache=cache,
                    representation=representation,
                )
            particles = particles + step_size * field
        recorder.record(step, particles)

    return recorder.to_run_result(
        method=method,
        seed=seed,
        elapsed_seconds=time.perf_counter() - started,
    )


def run_toy_particle_comparison(
    config: ExperimentConfig,
    *,
    n_samples: int = 512,
    ambient_dim: int = 32,
    signal_dim: int = 2,
    centers: int = 8,
    cluster_std: float = 0.6,
    nuisance_scale: float = 1.0,
    initial_signal_scale: float = 1.5,
    batch_size: int = 256,
    n_landmarks: int = 128,
    temperature: float = 1.0,
    step_size: float = 0.1,
    backend: str = "exact",
) -> ComparisonResult:
    """Run the raw-versus-known-signal particle gate from one notebook call."""

    if set(config.methods) - {"raw", "known_signal"}:
        raise ValueError("toy comparison methods must be 'raw' and/or 'known_signal'")

    def paired_factory(seed: int) -> PairedRunInputs:
        from .data import make_paired_run_inputs

        return make_paired_run_inputs(
            n_steps=config.drift_steps,
            dataset_size=n_samples,
            batch_size=batch_size,
            noise_dim=1,
            n_landmarks=n_landmarks,
            evaluation_size=1,
            seed=seed,
        )

    def run_one(
        run_config: ExperimentConfig,
        method: str,
        seed: int,
        paired_inputs: PairedRunInputs,
    ) -> RunResult:
        problem = make_toy_drift_problem(
            n_samples=n_samples,
            ambient_dim=ambient_dim,
            signal_dim=signal_dim,
            centers=centers,
            cluster_std=cluster_std,
            nuisance_scale=nuisance_scale,
            initial_signal_scale=initial_signal_scale,
            random_state=seed,
        )
        representation = None if method == "raw" else problem.project_signal
        # Discovery notebooks can opt out of writing sample snapshots and
        # manifests while retaining all in-memory curves and summaries.
        save_artifacts = bool(run_config.extras.get("save_artifacts", True))
        run_dir = (
            Path(run_config.output_dir) / method / f"seed_{seed}"
            if save_artifacts
            else None
        )
        return run_particle_evolution(
            problem=problem,
            paired_inputs=paired_inputs,
            representation=representation,
            evaluator=make_toy_evaluator(problem),
            method=method,
            seed=seed,
            temperature=temperature,
            step_size=step_size,
            eval_every=run_config.eval_every,
            backend=backend,
            output_dir=run_dir,
        )

    return run_comparison(config, run_one=run_one, paired_factory=paired_factory)


__all__ = ["run_particle_evolution", "run_toy_particle_comparison"]
