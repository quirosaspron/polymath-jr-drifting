"""Small compatibility tests for the representation experiment infrastructure."""

import torch

from src import drift, driftXpress
from src.data import make_paired_run_inputs, make_toy_drift_problem
from src.evaluators import make_toy_evaluator
from src.experiments import EvolutionRecorder, summarize_runs
from src.field import build_xpress_cache, compute_exact_field, compute_xpress_field
from src.models import LatentGenerator, VAE
from src.toy import run_particle_evolution


def test_exact_identity_field_matches_legacy() -> None:
    generator = torch.Generator().manual_seed(7)
    negative = torch.randn(12, 5, generator=generator)
    positive = torch.randn(15, 5, generator=generator)

    expected = drift.compute_V(negative, positive, T=1.3)
    actual = compute_exact_field(
        negative,
        positive,
        temperature=1.3,
        representation=None,
    )

    torch.testing.assert_close(actual, expected)


def test_xpress_identity_field_matches_legacy() -> None:
    generator = torch.Generator().manual_seed(17)
    negative = torch.randn(10, 4, generator=generator)
    positive = torch.randn(20, 4, generator=generator)
    indices = torch.tensor([1, 3, 5, 8, 13, 17])
    landmarks = positive[indices]
    temperature = 0.9

    legacy_inverse = driftXpress.build_nystrom_cache(landmarks, temperature)
    legacy_ap, legacy_bp = driftXpress.pre_compute_summaries(
        positive, landmarks, legacy_inverse, temperature
    )
    expected = driftXpress.compute_V(
        negative,
        landmarks,
        legacy_inverse,
        legacy_ap,
        legacy_bp,
        T=temperature,
    )

    cache = build_xpress_cache(
        positive,
        landmark_indices=indices,
        temperature=temperature,
    )
    actual = compute_xpress_field(negative, cache=cache)
    torch.testing.assert_close(actual, expected)


def test_paired_inputs_replay_exactly() -> None:
    arguments = dict(
        n_steps=5,
        dataset_size=50,
        batch_size=8,
        noise_dim=3,
        n_landmarks=7,
        evaluation_size=11,
        seed=27,
    )
    first = make_paired_run_inputs(**arguments)
    second = make_paired_run_inputs(**arguments)

    for name, tensor in first.to_dict().items():
        torch.testing.assert_close(tensor, second.to_dict()[name])


def test_toy_projection_and_external_evaluator() -> None:
    problem = make_toy_drift_problem(
        n_samples=128,
        ambient_dim=8,
        signal_dim=2,
        rotate=True,
        random_state=7,
    )
    torch.testing.assert_close(problem.project_signal(problem.target), problem.target_signal)
    torch.testing.assert_close(problem.project_nuisance(problem.initial), problem.initial_nuisance)

    metrics = make_toy_evaluator(problem, metrics=("swd",)).evaluate(problem.initial)
    assert set(metrics) == {"observed_swd", "signal_swd", "nuisance_swd"}
    assert metrics["signal_swd"] > 0
    assert metrics["nuisance_swd"] < 1e-6


def test_evolution_recorder_collects_checkpoint_curves() -> None:
    problem = make_toy_drift_problem(n_samples=64, ambient_dim=6, random_state=17)
    evaluator = make_toy_evaluator(problem, metrics=("swd",))
    recorder = EvolutionRecorder(
        fixed_queries=None,
        generate_fn=lambda particles, _: particles,
        evaluator_fn=evaluator.evaluate,
        every=5,
        save_samples=False,
    )
    recorder.record(0, problem.initial)
    assert recorder.record(1, problem.initial) is None
    recorder.record(5, problem.target)
    run = recorder.to_run_result(method="known_signal", seed=17, elapsed_seconds=1.0)
    summary = summarize_runs(
        [run], metric_name="signal_swd", epsilon=None, fixed_budget=5
    )

    assert run.steps == [0, 5]
    assert summary[0]["fixed_budget_mean"] < 1e-6


def test_ordinary_models_have_notebook_compatible_shapes() -> None:
    vae = VAE()
    images = torch.rand(4, 784)
    mu, logvar, latent = vae.get_latent(images)
    assert mu.shape == logvar.shape == latent.shape == (4, 16)
    assert vae.decode(latent).shape == (4, 784)

    generator = LatentGenerator(num_classes=10)
    generated = generator(torch.randn(4, 16), torch.tensor([0, 1, 2, 3]))
    assert generated.shape == (4, 16)


def test_particle_evolution_records_toy_curve() -> None:
    problem = make_toy_drift_problem(n_samples=64, ambient_dim=6, random_state=27)
    paired = make_paired_run_inputs(
        n_steps=3,
        dataset_size=64,
        batch_size=8,
        noise_dim=1,
        n_landmarks=8,
        evaluation_size=1,
        seed=27,
    )
    result = run_particle_evolution(
        problem=problem,
        paired_inputs=paired,
        representation=None,
        evaluator=make_toy_evaluator(problem, metrics=("swd",)),
        method="raw",
        seed=27,
        eval_every=1,
    )
    assert result.steps == [0, 1, 2, 3]
    assert len(result.metrics["signal_swd"]) == 4


def test_particle_evolution_accepts_known_signal_xpress_field() -> None:
    problem = make_toy_drift_problem(n_samples=48, ambient_dim=6, random_state=37)
    paired = make_paired_run_inputs(
        n_steps=2,
        dataset_size=48,
        batch_size=8,
        noise_dim=1,
        n_landmarks=8,
        evaluation_size=1,
        seed=37,
    )
    result = run_particle_evolution(
        problem=problem,
        paired_inputs=paired,
        representation=problem.project_signal,
        evaluator=make_toy_evaluator(problem, metrics=("swd",)),
        method="known_signal",
        seed=37,
        eval_every=1,
        backend="xpress",
    )
    assert result.steps == [0, 1, 2]
