import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Inches
from docx.text.paragraph import Paragraph
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
s = json.loads((OUT / "experiment_summary.json").read_text())
a = json.loads((OUT / "data_audit.json").read_text())
review = json.loads((OUT / "qualitative_review.json").read_text())
t = s["test"]
sections = [
    ("Overview",
     "This project trains an unconditional variational autoencoder (VAE) to generate "
     "28 × 28 digit images from real MNIST handwriting. It implements the full "
     "encoder, decoder, reparameterization, training and prior sampling in PyTorch, "
     "rather than using an external AI service or pretrained model. The goal is "
     "an inspectable small-scale generative experiment with honest output evaluation "
     "and responsible-use reasoning, not polished handwriting or a production system."),
    ("Dataset or Prompt Description",
     "MNIST provides 60,000 official training and 10,000 test images, grayscale "
     "uint8 pixels, and digit labels 0–9 (TensorFlow, n.d.). It is real human-created "
     "handwriting, not synthetic input data, and is different from the Fashion-MNIST "
     "clothing dataset used in the preceding capstone. The four unmodified compressed "
     "IDX files are included, downloaded from the public CVDF mirror at "
     "https://storage.googleapis.com/cvdf-datasets/mnist/. Published MD5 values and "
     "locally computed SHA-256 fingerprints, shapes, headers and lengths are checked. "
     "Keras (n.d.) documents MNIST's Creative Commons Attribution-Share Alike 3.0 terms; "
     "the included data notice attributes its creators and distinguishes data rights "
     "from claims about generated outputs.\n\n"
     "A seeded split of the official training set gives 55,000 training and 5,000 "
     "validation images; the official 10,000 test images remain reserved for final "
     "diagnostics. Training digit counts range from 4,982 for 5 to 6,204 for 1, "
     "so the data are not exactly class-balanced. Labels are used only for inspection "
     "and example selection, never as model inputs or training targets. Pixels are "
     "scaled to [0,1], thresholded at >=0.5, converted to float32 and flattened "
     "to 784 values. Figure 1 shows raw versus binarized first examples per digit. "
     "Thresholding supports a Bernoulli likelihood but discards gray edge detail "
     "(Kingma & Welling, 2013).\n\n"
     "No exact grayscale repeats or train/test pixel matches were found. After "
     f"thresholding, {a['binarized_validation_matching_training']} validation images "
     f"(0.06%) and {a['binarized_test_matching_training']} test images (0.08%) match "
     "training inputs exactly. All planned records are retained, and this "
     "preprocessing-induced overlap is disclosed as a limitation of the held-out "
     "diagnostics. Writer identities and demographics are absent from these IDX "
     "files; writer-level independence, near duplicates and population representation "
     "have not been established."),
    ("Model Design and Training Approach",
     "A VAE learns an approximate posterior for each input and a decoder that maps "
     "latent values to a distribution over observations. The reparameterization "
     "z = mu + exp(0.5 × logvar) × epsilon, with epsilon drawn from N(0,I), allows "
     "gradients to reach the encoder through sampled latent values "
     "(Kingma & Welling, 2013). The 784 → 256 ReLU encoder has separate 16-unit "
     "mean and log-variance heads; the decoder is 16 → 256 ReLU → 784 logits. "
     f"There are {s['parameter_count']:,} trainable parameters. The prior is standard "
     "normal and the approximate posterior has diagonal covariance. This small "
     "fully connected design is appropriate for demonstrating digit generation and "
     "latent sampling with a short CPU budget, but no architecture comparison proves "
     "its widths or latent dimension optimal.\n\n"
     "Per image, reconstruction binary cross-entropy is summed across 784 pixels "
     "and analytic KL divergence is summed across 16 latent coordinates. The KL "
     "formula is 0.5 × sum(mu² + exp(logvar) − 1 − logvar); the two terms are added "
     "with coefficient 1, then averaged over the batch. Binary targets make the "
     "reconstruction term a Bernoulli negative log-likelihood. Raw logits go into "
     "BCEWithLogits; sigmoid is used only when displaying or sampling probabilities "
     "(PyTorch contributors, n.d.-a). The training objective estimates a negative "
     "evidence lower bound, not exact marginal negative log-likelihood "
     "(Kingma & Welling, 2013).\n\n"
     "Adam uses learning rate 0.001, betas=(0.9,0.999), epsilon=1e-8, zero weight "
     "decay and foreach=False. Ten full epochs use batch size 256, including the "
     "last partial batch. CPU float32, two threads, seed 20261119 and deterministic "
     "algorithms target repeatability on this environment, not universal hardware "
     "equivalence (PyTorch contributors, n.d.-b). Training noise, epoch order, "
     "validation noise and generation use separated seeded generators. Validation "
     "uses the same posterior noise each epoch, with no gradient updates. Final "
     "epoch-ten weights are used; there is no search, augmentation, early stopping "
     "or validation-based model selection. Finite losses and declining validation "
     "objective show this run learned under the chosen settings; they do not "
     "establish that longer training or another model would be worse."),
    ("Output Evaluation and Interpretation",
     f"Figure 2 shows validation total falling from {s['first_validation_total']:.3f} "
     f"to {s['final_validation_total']:.3f} per image. Reconstruction improves while "
     "KL generally rises, reflecting a tradeoff within the summed objective rather "
     "than a requirement that every component decrease. No non-finite loss or "
     "training crash occurred. Training curves are online metrics while weights "
     "change; validation curves use fixed end-of-epoch weights. The remaining "
     "challenge is output ambiguity and noisy sampling, not an invented optimization "
     "failure.\n\n"
     "Generation uses all 64 fixed standard-normal latent draws from seed 20261120. "
     "No real image or digit label enters this step. Figure 3 shows conditional "
     "mean pixel probabilities; Figure 4 makes literal Bernoulli pixel draws "
     "from those same probabilities with a separate seed. Neither grid is "
     "cherry-picked. " + review["report_observations"] + "\n\n"
     f"On all 10,000 held-out images, one fixed posterior draw per image gives "
     f"mean reconstruction BCE {t['reconstruction_bce']:.3f}, KL {t['kl']:.3f} and "
     f"total {t['total']:.3f}; posterior-mean pixel MSE is "
     f"{t['posterior_mean_mse']:.5f}. These are reconstruction diagnostics, not "
     "generation accuracy, exact marginal likelihood or proof of image realism. "
     "Different generative evaluation criteria need not agree, so task-specific "
     "visual inspection supplements the objective (Theis et al., 2016).\n\n"
     "Figure 6 compares the first eight generated means with their nearest "
     "binarized input across all 55,000 training images using pixel RMSE. "
     "S00's nearest image is training index 2192 at RMSE 0.215; S01's is index "
     "32186 at 0.223; S02's is index 42159 at 0.204. S01 looks more like a 3 "
     "but its nearest training image looks more like a 4, illustrating that "
     "pixel similarity is not semantic agreement. Across all 64 means, nearest "
     f"RMSE ranges from {s['nearest_rmse_min']:.3f} to {s['nearest_rmse_max']:.3f}, "
     f"with median {s['nearest_rmse_median']:.3f}. All 64 quantized mean images "
     "differ from each other. Neither result proves originality, privacy, complete "
     "digit coverage or absence of memorization."),
    ("Interpretation for a Non-Technical Audience",
     "The program learned from pictures of real handwritten numbers and made "
     "64 new small pictures. Some are easy to recognize: S00 looks like a long "
     "looped 0, and S02 looks like a 2 with a broad bottom stroke. Others are "
     "unclear: S04 is a blurred shape and S06 has broken curves. The cleaner "
     "gray pictures can become speckled when the program makes black-and-white "
     "versions, so it is important to show both. A person could use these "
     "pictures to explore how a program learns handwriting patterns, but should "
     "check the pictures and label them as computer-made. They should not be "
     "passed off as someone's real handwriting or used as proof that a person "
     "wrote a number."),
    ("Ethical Considerations and Responsible Use",
     "Dataset composition limits what the generator can represent. Training "
     "frequencies differ by digit, and the labels do not reveal writer demographics "
     "or establish worldwide handwriting coverage. S04 and S06 demonstrate weak "
     "outputs, but they cannot support a claim that a particular demographic is "
     "harmed more often. Documentation of collection, composition and intended use "
     "is needed before extending such conclusions (Gebru et al., 2021). Here the "
     "responsible claim is narrow: educational generation from this dataset, with "
     "unknown writer-level representation and no demographic fairness test.\n\n"
     "Authenticity-sensitive misuse is a contextual risk: an image like S02 could "
     "be shown as genuine handwriting even though it was computer-generated. "
     "This project has not demonstrated document forgery or writer impersonation; "
     "it generates isolated digits, not signatures or complete records. Nonetheless, "
     "every generated grid is labeled synthetic, and financial records, identity "
     "verification, signed documents and evidentiary use are excluded. Ambiguous "
     "S04/S06 outputs should also not be quietly inserted into another labeled "
     "training dataset as correct real examples. Any downstream use needs explicit "
     "synthetic labeling, human review and separate validation.\n\n"
     "Creative ownership and data rights require more than calling an image new. "
     "Preserve MNIST attribution and applicable CC BY-SA terms (Keras, n.d.); "
     "do not claim exclusive ownership of a writer's style or that training "
     "automatically grants every downstream right. The observed nearest-image "
     "distances are not a memorization audit or legal conclusion. No private "
     "documents were used, and writer reidentification or membership inference "
     "was not tested. If extending to personal handwriting, obtain permission "
     "and assess collection and retention risks before use, consistent with "
     "provenance-centered documentation (Gebru et al., 2021)."),
    ("Limitations and Future Improvements",
     "One architecture, one seed and ten epochs do not establish comparative "
     "performance or robustness. Fully connected layers omit explicit spatial "
     "structure; the binary representation loses gray detail, and independent "
     "pixel sampling contributes visible speckles. The unconditional model cannot "
     "request a particular digit, produce meaningful text or authenticate writers. "
     "The small visual review has no blinded raters, classifier-based class "
     "counts or inter-rater agreement. Mean-image similarity is only a limited "
     "diagnostic. The three validation and eight test binarization collisions "
     "and untested writer overlap further qualify held-out interpretation.\n\n"
     "Realistic next steps are a preplanned convolutional VAE comparison, repeated "
     "runs across seeds, longer fixed training budgets and independent human "
     "ratings of legibility and diversity. A future split could group identical "
     "processed images before training; writer-aware validation would require "
     "additional source information. A conditional model could explicitly request "
     "digits, but would be a new experiment. These improvements have not been "
     "run and are not presented as achieved results.\n\n"
     "AI assistance was used in implementation, validation and drafting. Edward "
     "must review, understand and personalize the analysis as required by his "
     "course's academic-integrity policy. The report's results, examples and "
     "diagnostics come from the accompanying actual model executions; no external "
     "AI-generated images replace the VAE's outputs."),
]

references = [
    ("Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., "
     "Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. "
     "Communications of the ACM, 64(12), 86–92. https://doi.org/10.1145/3458723",
     "Communications of the ACM, 64"),
    ("Keras. (n.d.). MNIST digits classification dataset. "
     "https://keras.io/api/datasets/mnist/", "MNIST digits classification dataset"),
    ("Kingma, D. P., & Welling, M. (2013). Auto-encoding variational Bayes "
     "[Preprint]. arXiv. https://doi.org/10.48550/arXiv.1312.6114",
     "Auto-encoding variational Bayes"),
    ("PyTorch contributors. (n.d.-a). "
     "torch.nn.functional.binary_cross_entropy_with_logits. PyTorch 2.15 documentation. "
     "https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.binary_cross_entropy_with_logits.html",
     "torch.nn.functional.binary_cross_entropy_with_logits"),
    ("PyTorch contributors. (n.d.-b). Reproducibility. PyTorch 2.15 documentation. "
     "https://docs.pytorch.org/docs/2.15/notes/randomness.html", "Reproducibility"),
    ("TensorFlow. (n.d.). mnist. TensorFlow Datasets. "
     "https://www.tensorflow.org/datasets/catalog/mnist", "mnist"),
    ("Theis, L., van den Oord, A., & Bethge, M. (2016). "
     "A note on the evaluation of generative models [Preprint]. arXiv. "
     "https://doi.org/10.48550/arXiv.1511.01844",
     "A note on the evaluation of generative models"),
]

parser = argparse.ArgumentParser()
parser.add_argument("template", type=Path)
args = parser.parse_args()
doc = Document(args.template)
patterns = [deepcopy(p._p) for p in doc.paragraphs]
original_styles = doc.styles.element.xml
original_section = etree.tostring(doc.sections[0]._sectPr, method="c14n", exclusive=True)
header_footer = {str(p.partname): p.blob for p in doc.part.package.parts
                 if "header" in p.content_type or "footer" in p.content_type}
body = doc._element.body
for child in list(body):
    if child is not doc.sections[0]._sectPr:
        body.remove(child)


def append_pattern(index, text=""):
    element = deepcopy(patterns[index])
    original = Paragraph(element, doc._body)
    properties = deepcopy(original.runs[0]._r.rPr) if original.runs else None
    for child in list(element):
        if child is not element.pPr:
            element.remove(child)
    body.insert(len(body) - 1, element)
    paragraph = Paragraph(element, doc._body)
    run = paragraph.add_run(text)
    if properties is not None:
        run._r.insert(0, properties)
    return paragraph


def page_break():
    append_pattern(13).runs[0].add_break(WD_BREAK.PAGE)


cover = {4: "Generating Handwritten Digits: VAE Analysis and Responsible Use",
         6: "AI Foundations (Edward Hickman)", 7: "Udacity",
         8: "Project 5 — Generative AI Applications", 9: "", 10: ""}
for index in range(11):
    append_pattern(index, cover.get(index, ""))
page_break()
append_pattern(14, cover[4])
text = f"# {cover[4]}\n\n{cover[6]}\n\n"
for heading, content in sections:
    append_pattern(19, heading)
    for paragraph in content.split("\n\n"):
        append_pattern(15, paragraph)
    text += f"## {heading}\n\n{content}\n\n"

figures = [
    ("figure_1_data_examples.png", "First official training image in each class, "
     "shown before modeling. Top: original grayscale. Bottom: binary threshold >=0.5."),
    ("figure_2_training_curves.png", "Loss terms are per-image means. Dashed: online "
     "training; solid: fixed-weight validation with identical posterior noise each epoch."),
    ("figure_3_generated_means.png", "All 64 prior draws from seed 20261120, in order. "
     "These synthetic images display conditional pixel probabilities, not literal binary samples."),
    ("figure_4_binary_samples.png", "Same 64 latent draws as Figure 3; pixels sampled "
     "with seed 20261121. These synthetic images are literal Bernoulli observations."),
    ("figure_5_reconstructions.png", "First official test image for each digit. Top: "
     "binarized input. Bottom: posterior-mean reconstruction, not unconditional generation."),
    ("figure_6_nearest_examples.png", "First eight prior means, not a quality-selected "
     "subset, paired with their nearest binarized training image across all 55,000 inputs. "
     "Pixel RMSE is not a privacy or originality guarantee."),
]
for number, (filename, caption) in enumerate(figures, 1):
    page_break()
    append_pattern(19, f"Figure {number}")
    append_pattern(1).runs[0].add_picture(str(OUT / filename), width=Inches(6.2))
    append_pattern(15, f"Note. {caption}")

page_break()
append_pattern(37, "References")
text += "## References\n\n"
for reference, italic_span in references:
    p = append_pattern(38)
    props = deepcopy(p.runs[0]._r.rPr)
    for run in list(p.runs):
        p._p.remove(run._r)
    before, after = reference.split(italic_span, 1)
    for value, italic in [(before, False), (italic_span, True), (after, False)]:
        run = p.add_run(value)
        if props is not None:
            run._r.insert(0, deepcopy(props))
        run.italic = italic
    text += f"{reference}\n\n"

doc.core_properties.title = cover[4]
doc.core_properties.author = cover[6]
doc.core_properties.subject = cover[8]
doc.core_properties.comments = "AI assistance disclosed. Instructor and due date not supplied."
assert doc.styles.element.xml == original_styles
assert etree.tostring(doc.sections[0]._sectPr, method="c14n", exclusive=True) == original_section
assert header_footer == {str(p.partname): p.blob for p in doc.part.package.parts
                         if "header" in p.content_type or "footer" in p.content_type}
doc.save(ROOT / "module_summary.docx")
(ROOT / "module_summary.md").write_text(text)
subprocess.run(["libreoffice", "-env:UserInstallation=file:///tmp/genai-report-office",
                "--headless", "--convert-to", "pdf", "--outdir", str(ROOT),
                str(ROOT / "module_summary.docx")], check=True)
(OUT / "report_template_audit.json").write_text(json.dumps({
    "template_url": "https://docs.google.com/document/d/1cv_MjQVKnGc9sbaYiOUna2155jMEvUxACwyDax-XYWI/edit",
    "template_sha256": hashlib.sha256(args.template.read_bytes()).hexdigest(),
    "styles_preserved": True, "section_preserved": True, "headers_preserved": True,
    "omitted_fields": ["Instructor", "Due date"],
    "pdf_font": "Liberation Sans substitutes for retained Arial DOCX styles",
}, indent=2) + "\n")
