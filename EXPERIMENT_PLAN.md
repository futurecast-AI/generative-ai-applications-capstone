# Experiment plan — recorded before training

Train one unconditional variational autoencoder (VAE) on real MNIST handwritten
digits. MNIST is not the Fashion-MNIST clothing dataset used in the previous
capstone. No external generation API, pretrained weights or AI-created training
images will be used.

- Source: original MNIST IDX files, via Google's public CVDF mirror.
- Reserve the official 10,000-image test set. Split the official 60,000 training
  images into 55,000 training and 5,000 validation images using seed 20261119.
- Convert uint8 pixels to float32 in [0,1], then **binarize at >=0.5**. This makes
  binary cross-entropy an actual Bernoulli reconstruction likelihood, rather
  than treating continuous grayscale targets as Bernoulli observations.
- Encoder: 784 → 256 ReLU, separate 16-dimensional mean and log-variance heads.
  Decoder: 16 → 256 ReLU → 784 logits. Standard normal prior, diagonal Gaussian
  approximate posterior, differentiable reparameterization.
- Loss: sum reconstruction BCE over 784 pixels plus analytic KL over 16 latent
  dimensions, then mean over images. KL coefficient 1, no warm-up.
- Adam lr=0.001, default betas/epsilon, foreach=False; batch 256; 10 epochs;
  CPU float32, two threads, deterministic algorithms.
- All training images participate in every epoch. No augmentation, tuning,
  early stopping or validation-based checkpoint selection. Use final weights.
- Plot training/validation loss components. Validation uses fixed seeded
  posterior noise at every epoch, separately from training randomness.
- Generate and display all 64 prior draws from seed 20261120, in fixed order.
  Display decoder means and also literal Bernoulli pixel samples. Do not
  replace disappointing outputs with another seed or pick only attractive ones.
- Show paired reconstructions of the first test image in each digit class.
  Compute final held-out stochastic reconstruction BCE and KL using one fixed
  posterior draw per image. These are not image quality scores or exact
  marginal likelihood.
- Compare each of the 64 decoder means with its nearest binarized training
  image by pixel RMSE. This is a limited similarity diagnostic, not a privacy,
  originality or copyright guarantee.
- Evaluate concrete visible strengths, ambiguity, noise and variability after
  viewing the generated grid. Do not invent digit labels or a success rate.
- Report actual failures, narrow dataset coverage, synthetic labeling,
  attribution and limitations on authenticity-sensitive use.

Provide an executed notebook, the supplied APA template populated with actual
results, original dataset files and attribution, an actual environment freeze,
tests and a submission ZIP. Disclose AI assistance; Edward must review and
understand the work and follow his course's academic-integrity policy.
