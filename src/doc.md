# `src` function guide

Short reference for the helpers in this folder. Use tensors with the drift
helpers and NumPy arrays (or PyTorch tensors) with `eval.py`.

## Evaluation helpers

Import with:

```python
from src.eval import evaluate_generation, plot_loss_history
```

Unless noted otherwise, feature and latent matrices use shape
`(n_samples, n_features)`.

### Distribution metrics

| Function | Use | Returns |
| --- | --- | --- |
| `sliced_wasserstein_distance(real, generated, n_projections=128, n_quantiles=256, p=1, random_state=0)` | Compare two feature distributions with random 1-D projections. | `float` |
| `mmd_rbf(real, generated, bandwidth=None, max_samples=2048, random_state=0)` | Compare distributions with an RBF-kernel MMD score. | `float` |
| `frechet_distance(real_features, generated_features, eps=1e-6)` | Compute FID-style distance from a common feature space. On raw pixels, interpret it only as a pixel-space Fréchet distance. | `float` |
| `fid(real_features, generated_features, eps=1e-6)` | Alias for `frechet_distance`. | `float` |
| `class_coverage(labels, n_classes=None, min_count=1)` | Fraction of classes represented at least `min_count` times. | `float` in `[0, 1]` |
| `class_entropy(labels_or_probabilities, n_classes=None, normalized=True)` | Diversity of generated class frequencies. Accepts labels or an `(n_samples, n_classes)` probability matrix. | `float` |
| `evaluate_generation(real_features, generated_features, generated_labels=None, ..., include_mmd=True, include_fid=True)` | Run SWD, optional MMD/FID, and optional class metrics together. | `dict[str, float]` |

Example:

```python
metrics = evaluate_generation(
    real_features,
    generated_features,
    generated_labels=generated_labels,
    n_classes=10,
)
```

Use the same fixed feature extractor for `real_features` and
`generated_features` before calling the distribution metrics.

### One-call post-training evaluation

| Function | Use | Returns |
| --- | --- | --- |
| `collect_latent_samples(model, loader, ...)` | Collect deterministic real encoder means, conditional generated latents, images, and labels. | dictionary of NumPy arrays |
| `latent_statistics(real_latents, generated_latents, factors=None)` | Compare latent mean, standard deviation, total correlation, and optional real MIG. | `dict[str, float]` |
| `evaluate_latent_model(model, loader, ...)` | Collect samples and compute the standard latent-space metrics in one call. | dictionary with `samples`, `metrics`, and `statistics` |
| `evaluate_per_class_generation(...)` | Run a chosen feature metric separately for each non-empty internal class. | `dict[int, dict[str, float]]` |

For the filtered MNIST notebooks, use:

```python
post_eval = evaluate_latent_model(
    f_trained, test_loader, device=device,
    n_classes=NUM_CLASSES, class_to_label=CLASS_TO_DIGIT,
    n_samples=2048, include_mmd=False,
)
```

### Latent metrics

| Function | Use | Returns |
| --- | --- | --- |
| `mutual_information(x, y, n_bins=20)` | Estimate MI after discretizing two variables. `x` and `y` must have the same sample count. | `float` |
| `mutual_information_gap(latents, factors, n_bins=20, return_per_factor=False)` | Estimate MIG using known factors or labels. `latents` is `(n_samples, latent_dim)`; `factors` is `(n_samples,)` or `(n_samples, n_factors)`. | `float`, or result dict when requested |
| `total_correlation(latents, eps=1e-6)` | Estimate dependence between latent dimensions using a Gaussian covariance approximation. | `float` |

Lower `total_correlation` is better under its approximation. MIG requires
meaningful observed factors or labels; it cannot infer semantic
disentanglement without them.

### Embeddings and plots

| Function | Use | Returns |
| --- | --- | --- |
| `pca_embedding(latents, n_components=2)` | Project latents with NumPy-only PCA. | `(n_samples, n_components)` array |
| `umap_embedding(latents, n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0, **kwargs)` | Project latents with optional `umap-learn`. | embedding array |
| `plot_loss_history(history, keys=(...), log_scale=True, figsize=(10, 14))` | Plot training-loss histories. | `(figure, axes)` |
| `plot_embedding(embedding, labels=None, ax=None, title=None, point_size=12.0, alpha=0.7)` | Scatter a two-column PCA/UMAP embedding. | `Axes` |
| `plot_latent_projections(latents, labels=None, methods=("pca", "umap"), figsize=(12, 5))` | Plot PCA and/or UMAP views. | `(figure, axes)` |
| `plot_real_generated_projections(real_latents, generated_latents, ...)` | Make PCA/UMAP plots colored by digit and by real/generated source. | dictionary of `(figure, axes)` pairs |
| `plot_conditional_mnist_results(model, loader, ...)` | Show real MNIST inputs, reconstructions, and conditional generations. | `(figure, axes)` |

Examples:

```python
embedding = pca_embedding(latents)
ax = plot_embedding(embedding, labels=labels, title="Latent space")
fig, axes = plot_loss_history(history)
```

`plot_loss_history` accepts keys such as `total`, `recon`, `kl`, `drift`,
`var`, `covar`, and `force`; it also recognizes the notebook names
`train_losses`, `recon_losses`, `kl_losses`, `drift_losses`, `var_losses`,
`covar_losses`, and `average_V`.

The underscore-prefixed functions in `eval.py` (`_to_numpy`, `_as_2d`,
`_check_pair`, `_pairwise_squared_distances`, `_matrix_sqrt_psd`,
`_labels_to_codes`, `_discretize`, `_entropy_from_codes`, and
`_import_matplotlib`) are implementation details. Call the public functions
above instead.

## Training helpers

Import with:

```python
from src.training import compose_objective, gradient_report, per_loss_gradient_norms
```

| Function | Use |
| --- | --- |
| `compose_objective(losses, weights, lambda_representation=1.0, loss_scale=1.0)` | Build the differentiable total loss and a report of raw, weighted, and actual contribution values. `lambda_representation` scales reconstruction/KL/variance/covariance as one group relative to drift or adversarial terms. |
| `format_epoch_losses(losses, weights, ...)` | Print raw and weighted epoch-average losses in the same convention as `compose_objective`. |
| `gradient_report(loss, model)` | Show which encoder, decoder, generator, or other parameters receive gradient from one loss. |
| `per_loss_gradient_norms(losses, model, weights, ...)` | Compare the weighted gradient norm of each loss term by model component. |

Leave `loss_scale=1.0` unless you are deliberately testing a global gradient
multiplier. It changes the gradient just like changing the learning rate; a
larger displayed scalar loss is not inherently easier to optimize.

## Drift loss helpers

The intended import is:

```python
from src.drift import compute_V, total_loss
```

All latent inputs are 2-D tensors. `y_neg_latent` and `y_pos_latent` must
share their feature dimension; batches may have different sizes.

| Function | Use | Returns |
| --- | --- | --- |
| `softmax(logits, dim)` | Numerically stable softmax along `dim`. | tensor |
| `compute_V(y_neg, y_pos, T, eps=1e-8)` | Compute the drift vector from negative and positive latent samples. | `(n_neg, latent_dim)` tensor |
| `drift_loss(y_neg, y_pos, temp)` | Regress negative latents toward one detached drift step. | `(loss, V)` |
| `recon_loss(y_pos_recon, y_pos)` | Mean-squared reconstruction loss. | scalar tensor |
| `kl_loss(mu, logvar)` | Mean VAE KL loss over the batch. | scalar tensor |
| `variance_loss(mu, gamma=1.5, eps=1e-4)` | VICReg-style penalty for low per-dimension variance. | scalar tensor |
| `covariance_loss(latents)` | Penalize off-diagonal latent covariance. | scalar tensor |
| `total_loss(y_pos_recon, y_pos, y_pos_mu, y_pos_logvar, y_neg_latent, y_pos_latent, temp, ...)` | Combine reconstruction, KL, drift, variance, and covariance losses. | `(total, loss_items)` |

Typical training use:

```python
loss, parts = total_loss(
    y_pos_recon, y_pos, y_pos_mu, y_pos_logvar,
    y_neg_latent, y_pos_latent, temp=1.0,
)
loss.backward()
```

`parts` contains `recon`, `kl`, `drift`, `V`, `var`, `covar`, and the grouped
representation terms. Adjust `lambda_kl`, `lambda_drift`, `lambda_var`,
`lambda_cov`, and optionally `lambda_representation` to change relative
contributions.

## DriftXpress helpers

The functions in `driftXpress.py` are the landmark/Nyström approximation of
the drift calculation:

| Function | Use | Returns |
| --- | --- | --- |
| `select_landmarks(y_pos, num_landmarks=128)` | Randomly select landmark rows. | landmark tensor |
| `kzu(z, landmarks, T)` | Compute the exponential distance kernel. | kernel matrix |
| `build_nystrom_cache(landmarks, T, jitter=1e-5)` | Build and cache the inverse square-root landmark kernel. | cache matrix |
| `nystrom_map(z, landmarks, K_uu_inv_sqrt, T)` | Map samples into the cached Nyström feature space. | mapped tensor |
| `pre_compute_summaries(y_pos, landmarks, K_uu_inv_sqrt, T)` | Precompute positive-sample summaries. | `(Ap, bp)` |
| `attraction(y_neg, landmarks, K_uu_inv_sqrt, Ap, bp, T)` | Compute attraction toward positive samples. | tensor |
| `repulsion(y_neg, T)` | Compute normalized negative-sample repulsion. | tensor |
| `compute_V(y_neg, landmarks, K_uu_inv_sqrt, Ap, bp, T=1.0)` | Return attraction minus repulsion. | tensor |
| `drift_loss(generated, landmarks, K_uu_inv_sqrt, Ap, bp, T)` | Regress generated latents toward one detached Nyström drift step. | `(loss, V)` |
| `recon_loss`, `kl_loss`, `variance_loss`, `covariance_loss` | Same loss terms as above. | scalar tensors |
| `total_loss(..., landmarks, K_uu_inv_sqrt, Ap, bp, ...)` | Combine the loss terms using the Nyström drift. | `(total, loss_items)` |

Prepare the cache once, then reuse it during training:

```python
landmarks = select_landmarks(y_pos)
K_cache = build_nystrom_cache(landmarks, temp)
Ap, bp = pre_compute_summaries(y_pos, landmarks, K_cache, temp)
```
