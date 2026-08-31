Does a drift-aware, frozen representation reduce the optimization steps and wall-clock time needed for a freshly initialized generator to reach a fixed,Does a drift-aware, frozen representation reduce the optimization steps and wall-clock time needed for a freshly initialized generator to reach a fixed, Does a drift-aware, frozen representation reduce the optimization steps and wall-clock time needed for a freshly initialized generator to reach a fDoes a drift-aware, frozen representation reduce the optimization steps
and wall-clock time needed for a freshly initialized generator to reach a fixed, independently measured distance from the target distribution?

I want to learn the representation explicitly for downstream drift convergance. Then freeze it and discard the representation training generator and prove that the benefit transfers to new generator initializations or architectures. 

The kernel we use for drifting is induced by our representation $\phi(x)$ so we have:
$$
k_\phi(x,y)= exp( \frac{-||\phi(x) - \phi(y)||}{\tau})
$$

So learning and modifying $\phi(x)$ changes the geometry that governs the drift. 

Explain the basics of the paper "Second-Order Drifting Models". 

They argue that kenel spectral stiffness makes fine-scale structure converge slowly. 

I already did an experiment where I tested the transfer protocol. That is the central idea. But as it is, it doesn't make the case that my present joint training objective is helping to produce a better transferable representation. 

Some issuses inherent in that experiment are:
- DriftXpress cache becomes obsolete as it uses landmarks in a space that was moved by the encoder. DriftXpress assumes a fixed encoder. So maybe for the representation fine-tuning I have to use normal drifting and only use driftXpress after we have freezed the representation. 

- We keept the temperature constant at 2 without normalizing feature distances. An encoder can change the latent space and therefore it would also make the intial temperature obsolete. The original paper explicitly normalizes both features and drift magnitude for cross-encoder comparisons.

- I used a latent MSE warm-up. I don't think this adds anything and only makes training more complex. 

- For a fair comparison against separate VAE, we would need the generator to use the same seeds, landmarks, noise, etc..

My first implementation of this experiment will be to freeze the pretrained VAE (encoder and decoder) and learn a metric head over the fixed latent space $\phi$. This will make comparisons between separate and joint more fair and give us insight into how the latent space is being transformed. 

Also before going into MNIST I will start with high dimensional synthetic data. 

The decisive question is: as ambient dimension increases, does the oracle representation reduce time-to-target, and does the learned representation approach the oracle?


What does easier drifting mean?
- Track number of steps/seconds to epsilon. ie the first step where an external discrepancy falls below a threshold. 
- How fast it gets to our discrepancy metrics (fid, mmd, etc..)



For our toy data we'll use $W_2$ sliced Wasserstein distance. For MNIST we'll use MMD/SWD or MNIST FID. 


Your possible contribution is therefore a representation objective specifically designed to reduce the number of drift updates required—not simply “we introduced a neural kernel.”

## Protocol
1. Pretrain and freeze VAE
2. Train $\phi$ using drift related information
3. Freeze $\phi$
4. Initialize a fresh generator
5. Train the generator using DriftXpress with the kernel defined by the frozen $\phi$
6. Compare against an identically initalized generator using the raw latent kernel

