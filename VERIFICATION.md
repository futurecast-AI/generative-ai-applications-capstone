# Verification evidence

## Focused checks

`python -m pytest tests -q`: **12 passed** on the final notebook/report inputs.
The saved output is in `outputs/validation.txt`.

Checks cover:

- Actual source fingerprints, IDX shapes, data attribution.
- Recreated seeded splits, disjoint indices and disclosed binary collisions.
- Independent BCE/KL formulas, finite gradients and zero KL for a normal posterior.
- Architecture, parameter count, correct reparameterization and loss history.
- All 64 prior latents, decoder means and literal Bernoulli samples reproduced
  from the saved weights and seeds.
- Every held-out reconstruction BCE, KL and MSE reproduced for all 10,000 images.
- Global nearest-training indices and direct pixel RMSE for all 64 generations.
- All ten selected test reconstructions reproduced from saved weights.
- Executed notebook, six embedded figures, sample-specific qualitative review,
  four-to-six-sentence ending summary.
- Required report sections, six images, grounded ethics, scholarly citations,
  explicit AI-assistance disclosure.
- Concrete jargon-free non-technical explanation.
- Actual environment freeze and preserved course-template formatting audit.

## Fresh-environment reproduction

Created a separate Python 3.11.18 environment from the actual freeze, installing
the CPU PyTorch wheel from its official index first. Executed only a copy of the
final notebook and original dataset in a clean output directory; no saved model
or prior results were copied there.

Byte-for-byte identical:

- `experiment_summary.json`
- `training_history.csv`
- `test_reconstruction_metrics.csv`
- `nearest_training_images.csv`

Element-for-element identical NumPy arrays:

- `prior_latents.npy`
- `generated_means.npy`
- `generated_binary_samples.npy`
- `reconstructions.npy`

Runtime wall-clock timing was excluded from equality checks because elapsed
time is not deterministic. This verifies this Python/Linux/CPU stack, not
every platform or a universal quality result.

## Visual and preview checks

Viewed all six original figures, including every one of the 64 generated means
and 64 literal binary samples, all ten reconstruction pairs and the first eight
nearest-image pairs. The qualitative review records specific sample IDs,
not a fabricated success rate.

Rendered and visually reviewed all **fourteen PDF pages**, including the
plain-language section on **page 5**. No clipping or missing figure pages found.
The report preserves native course-template geometry, styles and header/footer
parts; Liberation Sans substitutes for Arial in the PDF, explicitly disclosed.

Verified the report preview responds using its actual endpoint hostname and
returns byte-identical PDF content. Verified the exported notebook preview
contains the sample-specific evaluation.

## Remaining limits

No GitHub Actions workflow is configured in this repository. Local verification
is not a claim of successful CI. Draft PR status, branch rules and remote head
are checked separately after publication.

No classroom submission or external reviewer approval has occurred. Edward
must review and understand this AI-assisted work before submitting.
