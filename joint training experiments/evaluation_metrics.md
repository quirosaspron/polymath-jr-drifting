# Evaluation metrics for joint-training algorithms

This file defines the primary metrics used to compare algorithms. The other functions in `src/eval.py` are diagnostic tools for understanding the model and latent space.

## Primary metrics

All algorithms must use the same data split, fixed evaluator, number of generated samples, training budget, and random seeds.

| Metric | Measures | Better |
|---|---|---|
| Sliced Wasserstein distance | Difference between real and generated feature distributions across random projections | Lower |
| Class coverage | Fraction of classes represented by at least the chosen number of generated samples | Higher |
| FID | Difference between the mean and covariance of real and generated evaluator features | Lower |

For MNIST, class labels should come from the same frozen classifier for every algorithm. FID and sliced Wasserstein distance should be computed on the classifier features, not directly on raw pixels.

## Results table

Fill this table after running each algorithm over the same evaluation protocol. Report mean ± standard deviation over multiple seeds.

| Algorithm | Sliced Wasserstein ↓ | Class coverage ↑ | FID ↓ |
|---|---:|---:|---:|
| Algorithm 1 | — | — | — |
| Algorithm 2 | — | — | — |
| Algorithm 3 | — | — | — |

An algorithm should only be called better if it improves the primary distribution metrics without a serious loss in coverage. If one algorithm has the best Wasserstein distance and another has the best coverage, report the trade-off rather than forcing one overall score.

## Algorithm 1

Algorithm 1 jointly trains the encoder, decoder, and latent generator with one weighted objective containing reconstruction, KL, drift, variance, and covariance terms.

Its distinguishing feature is that the drift loss is evaluated in latent space with a detached target: the drift term updates the latent generator, while the encoder and decoder are updated by reconstruction and the non-drift regularizers.

## Diagnostic metrics

These are useful for interpretation but are not primary algorithm-ranking metrics:

- MIG: useful when known factors or labels are available; higher is generally better.
- Total correlation: measures dependence between latent dimensions; lower means less dependence under the current Gaussian approximation.
- PCA and UMAP: visualizations, not scores.
- Latent correlation matrix: shows redundant or strongly coupled latent dimensions.
- Loss curves and latent spread: useful for diagnosing collapse, instability, or optimization problems.
