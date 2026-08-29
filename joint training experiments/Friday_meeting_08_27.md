# Friday meeting — August 27

## Headline

On the six-class MNIST experiment, the strongest saved VAE result is the
**separate-training baseline**: pixel FID **13.62** and pixel
sliced-Wasserstein **0.0570**. The best saved fully-joint result is the staged
pretrained run (pixel FID **23.29**); the from-scratch full-joint run fails
decisively (pixel FID **79.50**).

The evidence this week is more specific than ``joint is better'': **a useful,
mostly fixed representation currently helps DriftXpress more than updating the
representation jointly**. Pretraining makes the joint route viable, but none
of the saved joint variants yet beats the separate baseline.

## Research question

Can joint training learn a representation in which DriftXpress reaches the
data distribution more easily and with fewer optimization steps?

Related question: can a direct pixel-space model, or a frozen/pretrained
representation, avoid the instability of learning an encoder and drift field
simultaneously?

## Experimental setup and metric note

- MNIST digits: **0, 8, 2, 4, 7, 9**; 35,788 training and 6,005 test images.
- Main VAE experiments: 32-dimensional conditional latent, approximately
  357k trainable parameters, and 200 DriftXpress landmarks per class.
- The core table uses the **saved pixel-space** FID and sliced-Wasserstein
  outputs from each notebook. Lower is better. Latent FID is diagnostic only.
- Pixel FID is useful for this within-week comparison, but is not canonical
  Inception FID. Future final comparisons should use the same frozen MNIST
  classifier/evaluator, seeds, sample count, and training budget for every
  method.

## Completed MNIST experiments

| Experiment | Main change | Pixel FID ↓ | Pixel SW ↓ | Latent FID ↓ | Coverage ↑ |
|---|---|---:|---:|---:|---:|
| [Separate baseline](algorithms/separate.ipynb) | Train VAE first, freeze it, then train the conditional generator | **13.62** | **0.0570** | **11.88** | 1.00 |
| [Encoder-free](algorithms/encoder-free.ipynb) | Direct image-space EMF/drift model; no VAE or latent space | 31.28 | 0.1042 | — | — |
| [Full joint](algorithms/full-joint.ipynb) | Joint re-encoded drift from scratch; cache fixed before encoder updates | 79.50 | 0.2022 | 69.21 | 1.00 |
| [Pretrained full joint](algorithms/pretrained-full-joint.ipynb) | 50 VAE-pretrain epochs + generator/drift warm-ups, then joint training | **23.29** | **0.0802** | 25.13 | 1.00 |
| [Cyclic pretrained full joint](algorithms/cyclic-pretrained-full-joint.ipynb) | Three cycles; cache rebuilt between cycles | 26.62 | 0.0939 | **13.78** | 1.00 |
| [Cache refresh](algorithms/do%20you%20need%20to%20resample%20in%20full%20joint/pretrained-full-joint-resample.ipynb) | Rebuild DriftXpress summaries every 25 joint epochs | 37.30 | 0.0913 | 92.90 | 1.00 |
| [Fixed cache](algorithms/do%20you%20need%20to%20resample%20in%20full%20joint/pretrained-full-joint-drift.ipynb) | Keep summaries fixed after pretraining | 43.95 | 0.0964 | 47.47 | 1.00 |
| [Adversarial](algorithms/adversarial.ipynb) | Latent discriminator added to joint objective | 49.50 | 0.1687 | 82.37 | 1.00 |

### What we can say from this table

1. The separate VAE-then-generator baseline is the current result to beat:
   pixel FID **13.62**, substantially below the best saved fully-joint run
   (**23.29**).
2. Full joint training from scratch fails (pixel FID **79.50**). Its UMAP and
   latent spread indicate a moving-representation / stale-cache mismatch.
3. Staged pretraining and warm-ups make fully-joint training much more usable
   than the other saved joint variants, but they have not closed the gap to the
   separate baseline.
4. Cyclic cache rebuilding is close to the staged joint result; the current
   latent adversarial loss is clearly worse.
5. Cache refreshing is directionally better than the fixed-cache run, but it
   requires a controlled repeat before claiming a causal effect.

> Important: these are not all matched ablations. Training budgets, warm-ups,
> and optimizer settings differ across several rows. Use the table to guide the
> next experiment, not to claim a definitive winner beyond the current saved
> baseline.

> Encoder-free has no latent metric by construction. The representation-transfer
> rows are expanded below with their learned-MNIST metrics; their two routes
> share the same evaluator and optimizer-step budget.

## Full-joint failure mode: moving representation, stale DriftXpress cache

The [from-scratch full-joint run](algorithms/full-joint.ipynb) records pixel
FID **79.50**, pixel SW **0.2022**, and latent FID **69.21**. At the final
epoch, the real latent standard deviation is **1.179** while the generated
latent standard deviation is only **0.061**. The generated distribution has
therefore collapsed into a much narrower region of the current latent space.

Its UMAP makes the likely mechanism visible: current real encodings move away
from the generated encodings. In full joint training, the encoder changes the
coordinates of the real distribution, while the DriftXpress landmarks and
attraction summaries were computed in the initial coordinates. The drift loss
then chases stale landmark summaries rather than the current real encoding
distribution. The UMAP is qualitative evidence, but it agrees with the large
latent-spread gap and the poor pixel metrics.

This is why cache refresh is a central ablation, not an incidental detail. A
matched fixed-versus-refresh run is required before deciding whether refreshing
the cache is sufficient to stabilize full joint training.

## Does the DriftXpress cache need to be refreshed?

| Experiment | Cache policy | Pixel FID ↓ | Pixel SW ↓ | Note |
|---|---|---:|---:|---|
| [Resample](algorithms/do%20you%20need%20to%20resample%20in%20full%20joint/pretrained-full-joint-resample.ipynb) | Rebuild summaries every 25 joint epochs | 37.30 | **0.0913** | 50 VAE-pretrain epochs; 200 joint epochs |
| [Fixed cache / normal drift](algorithms/do%20you%20need%20to%20resample%20in%20full%20joint/pretrained-full-joint-drift.ipynb) | Build once after pretraining | 43.95 | 0.0964 | 100 VAE-pretrain epochs; 200 joint epochs |

Refreshing the cache looks directionally helpful in the saved runs, but this
is **not a controlled ablation**: pretraining length and optimizer settings
also differ. The next clean experiment should hold the model, seed, pretrain
budget, joint budget, and learning rates fixed, changing only
`RESAMPLE_EVERY` (for example: never, 25, and 10 epochs).

## Representation-learning experiment

The [representation-transfer run](algorithms/representation%20learning%20for%20easier%20drift/representation_transfer.ipynb) now provides the clearest direct
test of the representation hypothesis. Both routes use 11,160 optimizer steps
and the same frozen MNIST evaluator:

| Route | Pixel FID ↓ | Pixel MMD ↓ | Pixel SW ↓ | Learned MNIST FID ↓ | Conditional accuracy ↑ |
|---|---:|---:|---:|---:|---:|
| Separate representation-transfer | **19.51** | **0.0227** | **0.0634** | **8.00** | 0.998 |
| Joint representation-transfer | 26.54 | 0.0473 | 0.0861 | 21.27 | 0.999 |

In this experiment, updating the representation jointly makes every reported
quality metric worse, despite virtually identical conditional accuracy. This
is the strongest current evidence that a stable pretrained representation is
preferable to co-adapting it with the drift objective.

## Encoder-free experiment

The [encoder-free run](algorithms/encoder-free.ipynb) removes the VAE entirely.
It maps Gaussian-noised images directly to images using EMF plus capped
pixel-, feature-, and characteristic-drift corrections.

| Parameters | Training | Pixel FID ↓ | Pixel MMD ↓ | Pixel SW ↓ | Learned MNIST FID ↓ | Conditional accuracy ↑ |
|---:|---|---:|---:|---:|---:|---:|
| 356,944 | 200 epochs / 14,472 steps; averaged final 5 checkpoints | 31.28 | 0.0551 | 0.1042 | 10.57 | 0.935 |

This is still a useful alternative path: it has essentially the same parameter
count as the 32-D VAE (about 357k) and avoids encoder/decoder co-adaptation.
However, the saved run does not beat the separate baseline and its generations
are visibly blurrier; keep it as a secondary direction rather than the main
Friday result.

## Runs without saved final metrics

- [pretrained-full-joint-Copy1](algorithms/pretrained-full-joint-Copy1.ipynb)
  currently has no saved final evaluation output, so it is not ranked.
- [Anchor experiment](algorithms/anchor.ipynb) has saved pretraining and
  generator-warm-up output, but no completed final evaluation.

These should not be ranked until the same final evaluation is run and saved.

## MoLab CIFAR-10 implementation

The [MoLab CIFAR-10 script](../Paper%20implementations/driftXpress/train_cifar10_driftxpress.py) is a direct
image-space DriftXpress implementation, not a VAE:

- **Model:** approximately 30M-parameter residual U-Net; base width 88,
  three residual blocks per stage, GroupNorm + SiLU, and spatial attention at
  8×8 and 4×4 resolutions.
- **Representation for DriftXpress:** frozen DINOv3 ViT-B/16 features,
  Nyström cache with 512 landmarks, Laplacian temperature 0.05.
- **Configured MoLab run:** 20,000 steps, batch size 256, checkpoints and
  samples every 1,000 steps, and 10k-sample FID every 5,000 steps.

No MoLab `metrics.jsonl`, checkpoint, or saved FID output is present in this
repository, so this should be presented as a **scaling implementation and run
configuration**, not as a reported CIFAR-10 quality result. Exporting the
MoLab results archive is the next requirement for a CIFAR table.

## Suggested Friday presentation flow

1. State the question: can a representation be made easier for DriftXpress to
   traverse?
2. Show the separate-baseline sample grid first: it is the current best
   standard VAE result and visually the clearest grid.
3. Show the full-joint UMAP with the real/generated latent spread (1.179 vs
   0.061): it is the clearest failure-mechanism slide.
4. Show the pretrained fully-joint grid next: it is the best joint result and
   motivates why staged training is still worth pursuing.
5. Show the separate-vs-joint representation-transfer grid side by side. This
   is the clearest answer to the research question so far.
6. Use the cache-refresh table as a supporting ablation; do not use its noisy
   sample grid as the headline visual.
7. Keep encoder-free and CIFAR-10 as one backup/future-work slide.

## Next experiments

1. Controlled pretraining ablation: same seed, optimizer, warm-ups, losses,
   and total steps; vary only VAE-pretraining epochs.
2. Controlled cache-refresh ablation: fixed vs refresh every 25 vs every 10
   joint epochs.
3. Rerun and save the common final evaluation for the
   `pretrained-full-joint-Copy1` replicate.
4. Evaluate every final model with the same frozen MNIST evaluator and at
   least three seeds; report mean ± standard deviation.
5. Recover/export the MoLab CIFAR-10 `metrics.jsonl`, checkpoint metadata,
   and sample grid before making any CIFAR quality claim.
