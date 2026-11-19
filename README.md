# Generative AI Applications capstone

**AI Foundations (Edward Hickman) · Udacity · Project 5**

An unconditional PyTorch variational autoencoder trained from scratch on real
MNIST handwriting, with fixed prior generations, literal Bernoulli samples,
held-out reconstruction diagnostics, specific visual failure analysis and
contextual ethics. MNIST is distinct from the earlier Fashion-MNIST capstone.

## Submission files

- `generative_model.ipynb` — executed, full model/training/generation code,
  six embedded figures, qualitative evaluation and five-sentence final summary.
- `module_summary.pdf` — required report sections, actual examples and metrics,
  plain-language explanation, APA 7 references.
- `requirements.txt` — actual `python -m pip freeze` from the working environment.
- `data/` — all four unmodified MNIST gzip files and data attribution notice.

Additional files include the editable Word report, Markdown report,
experiment plan, provenance, saved weights, numeric arrays, per-image
diagnostics, HTML notebook and focused tests. No external API key is needed.

## Actual results and limits

The 415,024-parameter model trained on 55,000 images for ten epochs. Validation
total objective declined from **148.673 to 97.274 per image**. Final test
reconstruction BCE was **70.348**, KL **25.002** and total **95.350**; these
are reconstruction/regularization diagnostics, **not generation accuracy or
exact marginal likelihood**.

Both full 64-sample grids are shown: conditional mean pixel probabilities and
literal Bernoulli pixel draws from the same latent vectors. S00/S02 have
recognizable digit-like shapes, while S04 is blurred and S06 fragmented.
No classifier or blinded human-rating study was conducted.

Three validation and eight test images become pixel-identical to training
inputs after binarization; no original grayscale cross-set matches were found.
These preprocessing collisions and unknown writer-level overlap are disclosed.
Nearest-image pixel RMSE is not a privacy, originality or copyright guarantee.

## Reproduce the notebook

Tested with **Python 3.11.18 on Linux CPU**. Create an independent environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m jupyter notebook
```

Open `generative_model.ipynb` and choose **Restart Kernel and Run All** from the
repository root. The original data are already included. If a file is missing,
the notebook downloads that exact MNIST source file and verifies its MD5.

For noninteractive execution:

```bash
python -m nbconvert --to notebook --execute --inplace generative_model.ipynb \
  --ExecutePreprocessor.timeout=1200
python -m pytest tests -q
```

Install the CPU PyTorch wheel from its official index **before** the remaining
freeze; `+cpu` wheels are not normally on the default PyPI index. Versions are
not silently replaced. Wheel availability differs across operating systems;
the fully tested reproduction target is the documented Linux/Python stack.
Determinism is not a guarantee of identical results on all machines.

## Rebuild derived files

The supplied notebook is already finalized and does not require these builders.
To intentionally recreate it from source:

```bash
python scripts/build_notebook.py
python -m nbconvert --to notebook --execute --inplace generative_model.ipynb
python scripts/build_notebook.py --finalize
python -m nbconvert --to html generative_model.ipynb --output-dir outputs \
  --output generative_model.html
```

Finalization inserts the recorded visual review and actual summary values
without changing model code or executed outputs. If changing the model or
samples, inspect the new figures and revise `outputs/qualitative_review.json`;
do not carry old sample descriptions into a changed experiment.

To rebuild the report, obtain the original course template:
https://docs.google.com/document/d/1cv_MjQVKnGc9sbaYiOUna2155jMEvUxACwyDax-XYWI/edit

```bash
python scripts/build_report.py /path/to/apa-template.docx
```

Report rebuilding additionally requires LibreOffice and suitable fonts.
The supplied PDF and editable DOCX need no report-builder setup to open.
The report preserves the template's native styles, geometry and headers;
Liberation Sans substitutes for Arial in the PDF. Instructor and due date
were not supplied and are not invented.

## Responsible use and academic integrity

This is an educational demonstration. Generated images are labeled synthetic
and must not be presented as genuine handwriting, signatures or real records.
Source attribution and CC BY-SA 3.0 information are in `data/DATA_NOTICE.md`.
The data notice is not a blanket license for every repository file or a legal
claim about output ownership.

AI assistance was used in implementation, verification and drafting. Edward
must review, understand and personalize the work according to his course's
academic-integrity policy. All reported generations are from the accompanying
trained model, not an external image service. Nothing is automatically
submitted to the classroom.
