# Algorithm 1 — Joint VAE–Generator Training

This formulation describes the current `JointModel` training loop in `experiments.ipynb`.

## Variables

For a minibatch

$$
X_B=\{x_i\}_{i=1}^{B},
\qquad
x_i\in\mathbb{R}^{D},
$$

the trainable parameter blocks are:

$$
\phi=\text{encoder parameters},
\qquad
\psi=\text{decoder parameters},
\qquad
\omega=\text{latent-generator parameters}.
$$

The latent dimension is $d$. The encoder produces a mean $\mu_i$ and log-variance $\ell_i$ for each real sample. The symbols $z_i^+$ and $z_i^-$ denote positive and generated latent points, respectively.

For loss calculations, stack the encoder means and generated latents row-wise:

$$
M=
\begin{bmatrix}
\mu_1^\mathsf{T}\\
\vdots\\
\mu_B^\mathsf{T}
\end{bmatrix},
\qquad
Z^-=
\begin{bmatrix}
(z_1^-)^\mathsf{T}\\
\vdots\\
(z_B^-)^\mathsf{T}
\end{bmatrix}
\in\mathbb{R}^{B\times d}.
$$

## Algorithm 1: Joint training

**Require:** minibatch loader $\mathcal{D}$, encoder $E_\phi$, decoder $D_\psi$, generator $G_\omega$, latent dimension $d$, temperature $\tau$, learning rate $\eta$, and loss weights $\lambda_{\mathrm{KL}},\lambda_{\mathrm{drift}},\lambda_{\mathrm{var}},\lambda_{\mathrm{cov}}$.

**Ensure:** jointly updated parameters $(\phi,\psi,\omega)$.

1. Initialize $\phi$, $\psi$, and $\omega$.

2. Initialize one AdamW optimizer over all parameters $(\phi,\psi,\omega)$.

3. For each training epoch:

   1. Set the model to training mode.

   2. For each minibatch $X_B$ from $\mathcal{D}$:

      1. Sample independent Gaussian noise

         $$
         \epsilon_i\sim\mathcal{N}(0,I_d),
         \qquad
         \xi_i\sim\mathcal{N}(0,I_d).
         $$

      2. Encode each real sample:

         $$
         (\mu_i,\ell_i)=E_\phi(x_i).
         $$

      3. Construct the positive latent samples using reparameterization:

         $$
         z_i^+
         =
         \mu_i+
         \exp\left(\frac{1}{2}\ell_i\right)
         \odot\epsilon_i.
         $$

      4. Decode the positive latent samples:

         $$
         \widehat{x}_i^+=D_\psi(z_i^+).
         $$

      5. Generate negative latent samples:

         $$
         z_i^-=G_\omega(\xi_i).
         $$

      6. Compute the generated reconstruction $\widehat{x}_i^-=D_\psi(z_i^-)$.

         This value is currently not used in the loss.

      7. Compute the reconstruction, KL, drift, variance, and covariance losses.

      8. Form the scalarized objective:

         $$
         \mathcal{L}
         =
         \mathcal{L}_{\mathrm{rec}}
         +
         \lambda_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}
         +
         \lambda_{\mathrm{drift}}\mathcal{L}_{\mathrm{drift}}
         +
         \lambda_{\mathrm{var}}\mathcal{L}_{\mathrm{var}}
         +
         \lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}.
         $$

      9. Clear the optimizer gradients.

      10. Backpropagate $\mathcal{L}$.

      11. Update all parameters with AdamW.

   3. If a learning-rate scheduler is used, call `scheduler.step()` once after the minibatches.

4. Return $(\phi,\psi,\omega)$.

## Gradient routes in the current implementation

The current drift loss is computed from latent points. Its drift field and target are detached. Therefore:

| Parameter block | Receives gradients from |
|---|---|
| Encoder $\phi$ | Reconstruction, KL, positive variance, and positive covariance losses |
| Decoder $\psi$ | Positive reconstruction loss only |
| Generator $\omega$ | Drift, generated variance, and generated covariance losses |

Thus, the encoder is **not** trained by the drift loss. The decoder is also **not** trained by the drift loss in the current implementation.

This differs from the teammate's “stop encoder drift-gradient” variant if that variant allows the drift loss to pass through the decoder. The current implementation is more restrictive: the drift update is latent-generator-only.

## Loss definitions

### Reconstruction loss

$$
\mathcal{L}_{\mathrm{rec}}
=
\frac{1}{BD}
\sum_{i=1}^{B}
\left\|
\widehat{x}_i^+-x_i
\right\|_2^2.
$$

### KL loss

$$
\mathcal{L}_{\mathrm{KL}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left[
-\frac{1}{2}
\sum_{j=1}^{d}
\left(
1+\ell_{ij}-\mu_{ij}^{2}-\exp(\ell_{ij})
\right)
\right].
$$

### Variance loss

For any $Z\in\mathbb{R}^{B\times d}$,

$$
s_j(Z)
=
\sqrt{
\operatorname{Var}_{\mathrm{unbiased}}(Z_{:,j})
+
\varepsilon
},
\qquad
\varepsilon=10^{-4},
$$

$$
\mathcal{V}(Z)
=
\frac{1}{d}
\sum_{j=1}^{d}
\operatorname{ReLU}\left(\gamma-s_j(Z)\right),
\qquad
\gamma=1.5.
$$

The current variance loss is

$$
\mathcal{L}_{\mathrm{var}}
=
\mathcal{V}(M)+\mathcal{V}(Z^-),
$$

where $M$ is the matrix whose rows are the encoder means $\mu_i$.

### Covariance loss

For any $Z\in\mathbb{R}^{B\times d}$, define

$$
\widetilde{Z}
=
Z-\mathbf{1}_B\overline{z}^{\mathsf{T}},
\qquad
\overline{z}
=
\frac{1}{B}
\sum_{i=1}^{B}Z_{i,:},
$$

and

$$
C(Z)
=
\frac{1}{B-1}
\widetilde{Z}^{\mathsf{T}}\widetilde{Z}.
$$

Removing the diagonal gives

$$
C_{\mathrm{off}}(Z)
=
C(Z)-\operatorname{diag}(C(Z)).
$$

The covariance penalty is

$$
\mathcal{C}(Z)
=
\frac{1}{d}
\left\|
C_{\mathrm{off}}(Z)
\right\|_F^2.
$$

The current covariance loss is

$$
\mathcal{L}_{\mathrm{cov}}
=
\mathcal{C}(M)+\mathcal{C}(Z^-).
$$

### Drift loss

The fixed DriftXpress routine supplies

$$
\mathcal{L}_{\mathrm{drift}}(Z^-,Z^+;\tau).
$$

Its target is detached during backpropagation, so this term updates the generator through $Z^-$ but does not update the encoder through $Z^+$.

## Optimization problem

The complete training problem is

$$
\min_{\phi,\psi,\omega}
\mathcal{L}(\phi,\psi,\omega),
$$

where

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{rec}}
+
\lambda_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}
+
\lambda_{\mathrm{drift}}\mathcal{L}_{\mathrm{drift}}
+
\lambda_{\mathrm{var}}\mathcal{L}_{\mathrm{var}}
+
\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}.
$$
