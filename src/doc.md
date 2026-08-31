# `src` quick guide

The notebooks are the research frontend: hypothesis, configuration, figures,
interpretation, and limitations. `src` holds reusable mechanics.

## Where things live

| Module | Status | What it provides |
| --- | --- | --- |
| `drift.py` | Legacy; do not change | Exact identity-kernel field and existing losses used by old notebooks. |
| `driftXpress.py` | Legacy; do not change | Existing identity-kernel Nyström/landmark approximation. |
| `field.py` | Ready | Exact and DriftXpress-style fields with `representation=` support. |
| `data.py` | Ready | Toy problems, MNIST loaders, and deterministic paired schedules. |
| `eval.py` | Ready | SWD, MMD, Fréchet/FID-style metrics, MNIST evaluator, latent metrics, and plots. |
| `evaluators.py` | Ready | Frozen checkpoint evaluators with fixed references and metric randomness. |
| `toy.py` | Ready | Direct particle evolution and one-call raw-versus-known-signal comparison. |
| `experiments.py` | Ready | Seeds, identical initialization, evolution snapshots, artifacts, paired comparisons, summaries. |
| `diagnostics.py` | Ready | Hitting time, discrepancy AUC, fixed-budget score, field/kernel diagnostics, curve plots. |
| `models/vae.py` | Ready | Ordinary MLP VAE moved out of the separate-training notebook. |
| `models/generator.py` | Ready | Ordinary conditional latent generator. |
| `models/phi.py` | Ready to modify | `PhiNetwork`: a small learned representation used by the drift kernel. |
| `representation.py` | Partly user-owned | `compute_phi_loss` remains the research objective; supervised and health-check helpers are available. |
| `trainers/loops.py` | **You implement** | A phi trainer and a fresh-generator trainer using frozen phi. |
| `training.py` | Ready | Generic objective composition and gradient reports used by current notebooks. |
| `matryoshka.py` | Existing specialized code | Matryoshka models, losses, trainers, and evaluation. |

Old notebooks can continue importing `src.drift`, `src.driftXpress`,
`src.eval`, and `src.training`. The representation experiment uses the new
modules and does not require changing the legacy ones.

## Representation-aware fields

```python
from src.field import (
    build_differentiable_xpress_cache,
    compute_exact_field,
    build_xpress_cache,
    compute_xpress_field,
)
```

The shared kernel is

```text
k_r(z1, z2) = exp(-distance(r(z1), r(z2)) / temperature)
```

The representation changes kernel weights, but the returned vectors remain in
the original latent coordinates.

| Function | Use |
| --- | --- |
| `compute_exact_field(negative, positive, temperature=..., representation=...)` | Drop-in experimental counterpart to `drift.compute_V`. `representation=None` is the raw baseline. |
| `compute_field_at_queries(queries, negative_reference=..., positive_reference=..., ...)` | Evaluate two independent field estimates at the same queries for diagnostics or a future direction-consistency loss. |
| `build_xpress_cache(positive, landmark_indices=..., representation=..., ...)` | Build one cache after VAE and representation are frozen. |
| `build_differentiable_xpress_cache(...)` | Training-only cache that preserves gradients through a changing phi; rebuild it after phi updates. |
| `compute_xpress_field(negative, cache=..., representation=...)` | Evaluate the cached approximate field. |
| `exponential_distance_kernel(x, y, representation=..., ...)` | Inspect the actual learned kernel weights. |

Do not build a cache while phi changes. A changed representation makes the
landmark kernel and positive summaries stale.

## Toy control

```python
from src.data import make_toy_drift_problem
from src.evaluators import make_toy_evaluator

problem = make_toy_drift_problem(
    n_samples=2048,
    ambient_dim=32,
    signal_dim=2,
    rotate=False,
    random_state=7,
)

raw_representation = None
known_signal_representation = problem.project_signal
evaluator = make_toy_evaluator(problem)
```

For the complete direct-particle gate, call `run_toy_particle_comparison`.
The lower-level `run_particle_evolution` function is available when you want to
inspect one condition manually.

The target is an eight-blob signal plus nuisance coordinates. Initial and
target particles share the same nuisance marginal, while their signal
distributions differ. The evaluator reports:

- `signal_swd` / `signal_mmd`: primary convergence measurements;
- `observed_swd` / `observed_mmd`: full-data guardrails;
- `nuisance_swd` / `nuisance_mmd`: whether drift damaged an already-correct marginal.

Start with `rotate=False`; the known representation is easy to inspect. Add a
rotation only after the basic experiment works.

## Deterministic paired comparisons

```python
from src.data import make_paired_run_inputs

paired = make_paired_run_inputs(
    n_steps=200,
    dataset_size=len(problem.target),
    batch_size=500,
    noise_dim=16,
    n_landmarks=128,
    evaluation_size=2048,
    seed=7,
)
```

Reuse the same `paired` object for raw and known-signal runs. It contains:

- `real_batch_indices[step]`;
- `training_noise[step]`;
- one fixed `landmark_indices` tensor for the cached Xpress comparison;
- `evaluation_noise`.

Use `save_paired_run_inputs` and `load_paired_run_inputs` to replay the exact
schedule on another machine. Use `clone_state_dict` / `load_initial_state` from
`experiments.py` so generators also start from identical weights.

## External evaluators

```python
from src.evaluators import DistributionEvaluator, MNISTExternalEvaluator
```

`DistributionEvaluator` fixes reference samples, SWD projection randomness,
and an MMD bandwidth estimated only from the reference. `make_toy_evaluator`
constructs its observed/signal/nuisance views.

`MNISTExternalEvaluator` wraps the frozen classifier-space evaluation already
implemented in `eval.py`. Train that classifier once with
`fit_mnist_evaluator`, freeze it, and reuse it for every method.

External metrics are never passed to the optimizer. If a metric becomes part
of `compute_phi_loss`, use a different held-out evaluator for the final curves.

## Evolution snapshots

`EvolutionRecorder` evaluates the same queries throughout training. Your loop
only decides when parameters are updated.

```python
from src.experiments import EvolutionRecorder

recorder = EvolutionRecorder(
    fixed_queries=paired.evaluation_noise,
    generate_fn=lambda generator, noise: generator(noise.to(device)),
    evaluator_fn=evaluator.evaluate,
    every=5,
    output_dir="artifacts/raw/seed_7",
)

recorder.record(0, generator)

# Your explicit optimizer loop:
# for step in range(1, 201):
#     ... calculate the loss, backward, optimizer.step() ...
#     recorder.record(step, generator, extra_metrics={"train_loss": loss.item()})

run = recorder.to_run_result(method="raw", seed=7, elapsed_seconds=elapsed)
```

The recorder stores scalar curves and optionally generated samples such as
`samples_step_000050.pt`. Use `save_model_checkpoint` separately when a full
model/optimizer checkpoint is useful.

## Comparison and plots

```python
from src.experiments import ExperimentConfig, run_comparison, save_comparison
from src.diagnostics import plot_discrepancy_curves

config = ExperimentConfig(
    dataset="toy32",
    methods=("raw", "known_signal"),
    seeds=(7, 17, 27),
    primary_metric="signal_swd",
    epsilon=None,          # discovery: inspect the scale first
    fixed_budget=200,
)

result = run_comparison(
    config,
    run_one=my_training_function,
    paired_factory=make_inputs_for_seed,
)

plot_discrepancy_curves(result, metric_name="signal_swd")
save_comparison(result)
result.summary
```

`my_training_function(config, method, seed, paired_inputs)` is your training
loop and returns a `RunResult`. `run_comparison` handles method ordering,
re-seeding, reuse of the per-seed paired object, timing, and aggregation.

## Convergence diagnostics

| Function | Meaning |
| --- | --- |
| `steps_to_threshold(steps, discrepancies, epsilon=...)` | First measured step reaching a predeclared target. |
| `discrepancy_auc(steps, discrepancies)` | Total discrepancy accumulated over training; lower means useful states arrived sooner. |
| `fixed_budget_discrepancy(..., budget=...)` | Quality after the same number of updates. |
| `field_consistency(...)` | Cosine agreement, normalized variance, and SNR of independent field estimates. Diagnostic only. |
| `kernel_effective_neighbors(weights)` | Whether a kernel is nearly uniform, extremely local, or numerically zero. |
| `gram_spectrum(matrix)` | Kernel effective rank and conditioning. |
| `capture_parameters` + `generator_update_diagnostics` | Gradient and actual optimizer-update magnitudes. |

## What you implement

The research-owned extension points are:

1. The architecture choices for `PhiNetwork` in `models/phi.py`.
2. `compute_phi_loss` in `representation.py`.
3. `train_phi` and `train_generator_with_frozen_phi` in `trainers/loops.py`.

Ordinary VAE pretraining may remain in its current notebook until that loop is
stable enough to extract; it is not required for the first particle toy.

The supervised signal-recovery helper is only an integration sanity check. The
exploratory notebook also demonstrates a differentiable one-step probe objective;
the final research loss is still undecided. Monitor `validate_representation`
and add regularization only in response to an observed failure.

## Recommended order

1. Verify raw `field.compute_exact_field` matches legacy `drift.compute_V` when representation is `None`.
2. Write a very small particle loop and compare raw versus known signal on one seed.
3. Run the same control with the representation-aware Xpress cache and three paired seeds.
4. Use the supervised phi notebook section to verify that a frozen neural representation reaches the oracle path.
5. Replace the supervised helper with a drift-aware `compute_phi_loss` and validate it first on a small pilot.
6. Freeze phi, build its Xpress cache, initialize a fresh paired generator, and run the final comparison.
