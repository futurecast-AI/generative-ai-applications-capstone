import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

from docx import Document
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def read_json(name):
    return json.loads((OUT / name).read_text())


def images(name):
    raw = gzip.decompress((ROOT / "data" / name).read_bytes())
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051 and (rows, cols) == (28, 28)
    assert len(raw) == 16 + count * 784
    return np.frombuffer(raw, np.uint8, offset=16).reshape(count, 784)


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(ROOT / "generative_model.ipynb", as_version=4)


@pytest.fixture(scope="module")
def implementation(notebook):
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np, "pd": pd}
    source = next(c.source for c in notebook.cells if "class DigitVAE" in c.source)
    exec(source.split("torch.manual_seed(SEED)")[0], namespace)
    evaluation = next(c.source for c in notebook.cells if "def evaluate(" in c.source)
    exec(evaluation.split("optimizer =")[0], namespace)
    return namespace


@pytest.fixture(scope="module")
def trained(implementation):
    torch.set_num_threads(2)
    model = implementation["DigitVAE"]()
    model.load_state_dict(torch.load(OUT / "vae_state.pt", map_location="cpu", weights_only=True))
    return model.eval()


def test_original_dataset_and_attribution():
    manifest = read_json("dataset_manifest.json")
    assert len(manifest) == 4
    for record in manifest:
        raw = (ROOT / record["file"]).read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.md5(raw).hexdigest() == record["md5"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    assert images("train-images-idx3-ubyte.gz").shape == (60000, 784)
    assert images("t10k-images-idx3-ubyte.gz").shape == (10000, 784)
    assert "Attribution-Share Alike 3.0" in (ROOT / "data/DATA_NOTICE.md").read_text()


def test_disjoint_seeded_partitions_and_binary_inputs():
    assignments = pd.read_csv(OUT / "split_assignments.csv")
    order = np.random.default_rng(20261119).permutation(60000)
    val = np.sort(order[:5000])
    train = np.sort(order[5000:])
    assert assignments.official_index.tolist() == list(range(60000))
    assert np.array_equal(assignments.index[assignments.split == "training"], train)
    assert np.array_equal(assignments.index[assignments.split == "validation"], val)
    assert not set(train) & set(val)
    audit = read_json("data_audit.json")
    assert hashlib.sha256(train.astype("<i8").tobytes()).hexdigest() == audit["training_index_sha256"]
    assert hashlib.sha256(val.astype("<i8").tobytes()).hexdigest() == audit["validation_index_sha256"]
    binary = (images("train-images-idx3-ubyte.gz").astype(np.float32) / 255 >= 0.5)
    hashes = [hashlib.sha256(row.astype(np.uint8).tobytes()).hexdigest() for row in binary]
    train_hashes = {hashes[i] for i in train}
    assert sum(hashes[i] in train_hashes for i in val) == audit["binarized_validation_matching_training"]
    assert pd.read_csv(OUT / "digit_counts.csv")[["training", "validation", "test"]].sum().tolist() == [55000, 5000, 10000]


def test_vae_loss_matches_independent_formula_and_backpropagates(implementation):
    targets = torch.tensor([[0., 1., 1.], [1., 0., 0.]])
    logits = torch.tensor([[0.2, -0.8, 1.1], [-1., 0.5, 0.3]], requires_grad=True)
    mu = torch.tensor([[0.4, 0.1], [0.2, -0.3]], requires_grad=True)
    logvar = torch.tensor([[-0.2, 0.3], [0., 0.2]], requires_grad=True)
    rec, kl = implementation["loss_components"](logits, targets, mu, logvar)
    l, t, m, v = [a.detach().numpy() for a in [logits, targets, mu, logvar]]
    np.testing.assert_allclose(rec.detach(), (np.logaddexp(0, l) - t * l).sum(1), rtol=1e-6)
    np.testing.assert_allclose(kl.detach(), 0.5 * (m*m + np.exp(v) - 1 - v).sum(1), atol=1e-7)
    (rec + kl).mean().backward()
    for tensor in [logits, mu, logvar]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    zero = torch.zeros(2, 16)
    _, zero_kl = implementation["loss_components"](torch.zeros(2, 784), torch.zeros(2, 784), zero, zero)
    assert torch.equal(zero_kl, torch.zeros(2))


def test_architecture_reparameterization_and_training_history(trained):
    config = read_json("experiment_config.json")
    assert sum(p.numel() for p in trained.parameters()) == config["parameter_count"]
    assert config["epochs"] == 10 and config["kl_coefficient"] == 1
    mu, logvar = torch.ones(2, 16), torch.full((2, 16), 0.6)
    a = torch.Generator().manual_seed(4)
    b = torch.Generator().manual_seed(4)
    expected = mu + (logvar * 0.5).exp() * torch.randn(2, 16, generator=b)
    assert torch.equal(trained.reparameterize(mu, logvar, a), expected)
    history = pd.read_csv(OUT / "training_history.csv")
    assert history.epoch.tolist() == list(range(1, 11))
    assert np.isfinite(history.to_numpy()).all()
    for prefix in ["train_", "validation_"]:
        np.testing.assert_allclose(history[prefix + "total"],
                                   history[prefix + "reconstruction_bce"] + history[prefix + "kl"])
    s = read_json("experiment_summary.json")
    assert history.validation_total.iloc[0] == pytest.approx(s["first_validation_total"])
    assert history.validation_total.iloc[-1] == pytest.approx(s["final_validation_total"])


def test_all_prior_draws_and_literal_samples_reproduce(trained):
    z = torch.randn(64, 16, generator=torch.Generator().manual_seed(20261120))
    np.testing.assert_array_equal(z, np.load(OUT / "prior_latents.npy"))
    with torch.inference_mode():
        means = torch.sigmoid(trained.decoder(z)).reshape(64, 28, 28)
        draws = torch.bernoulli(means, generator=torch.Generator().manual_seed(20261121))
    np.testing.assert_array_equal(means, np.load(OUT / "generated_means.npy"))
    np.testing.assert_array_equal(draws, np.load(OUT / "generated_binary_samples.npy"))
    assert means.min() >= 0 and means.max() <= 1
    assert set(draws.flatten().tolist()) == {0., 1.}
    assert read_json("experiment_summary.json")["sample_count"] == 64


def test_all_heldout_metrics_reproduce(trained, implementation):
    binary = torch.from_numpy((images("t10k-images-idx3-ubyte.gz").astype(np.float32) / 255 >= 0.5).astype(np.float32))
    metrics, records = implementation["evaluate"](trained, binary, 20261419)
    saved = pd.read_csv(OUT / "test_reconstruction_metrics.csv")
    assert saved.official_test_index.tolist() == list(range(10000))
    np.testing.assert_allclose(records, saved[records.columns], rtol=1e-12, atol=1e-12)
    for key, value in metrics.items():
        assert value == pytest.approx(read_json("experiment_summary.json")["test"][key], abs=1e-12)


def test_nearest_training_images_are_global_minima():
    order = np.random.default_rng(20261119).permutation(60000)
    indices = np.sort(order[5000:])
    train = torch.from_numpy((images("train-images-idx3-ubyte.gz")[indices].astype(np.float32) / 255 >= 0.5).astype(np.float32))
    means = torch.from_numpy(np.load(OUT / "generated_means.npy").reshape(64, 784))
    nearest = pd.read_csv(OUT / "nearest_training_images.csv")
    with torch.inference_mode():
        distances = torch.cdist(means, train, compute_mode="use_mm_for_euclid_dist")
        positions = distances.argmin(1)
        np.testing.assert_array_equal(positions, nearest.training_position)
        np.testing.assert_array_equal(indices[positions], nearest.official_training_index)
        direct = ((means - train[positions]).square().mean(1)).sqrt()
    np.testing.assert_allclose(direct, nearest.pixel_rmse, rtol=1e-7)
    assert nearest.sample_id.tolist() == [f"S{i:02d}" for i in range(64)]


def test_reconstruction_examples_reproduce(trained):
    raw = gzip.decompress((ROOT / "data/t10k-labels-idx1-ubyte.gz").read_bytes())
    labels = np.frombuffer(raw, np.uint8, offset=8)
    indices = [int(np.flatnonzero(labels == d)[0]) for d in range(10)]
    assert indices == read_json("experiment_summary.json")["reconstruction_test_indices"]
    binary = (images("t10k-images-idx3-ubyte.gz")[indices].astype(np.float32) / 255 >= .5).astype(np.float32)
    with torch.inference_mode():
        mu, _ = trained.encode(torch.from_numpy(binary))
        reconstructed = torch.sigmoid(trained.decoder(mu)).reshape(10, 28, 28)
    np.testing.assert_array_equal(reconstructed, np.load(OUT / "reconstructions.npy"))


def test_notebook_is_executed_and_evaluates_specific_samples(notebook):
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    assert sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs) == 6
    source = "\n".join(c.source for c in notebook.cells)
    assert "Qualitative review pending" not in source
    review = next(c.source for c in notebook.cells if c.metadata.get("role") == "qualitative")
    assert len(set(re.findall(r"S\d{2}", review))) >= 3
    summary = notebook.cells[-1].source.split("\n\n", 1)[1]
    assert len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", summary)) in range(4, 7)
    assert "posterior" in source and "Bernoulli" in source and "no external" in source.lower()


def test_report_required_sections_images_and_grounded_ethics():
    doc = Document(ROOT / "module_summary.docx")
    headings = ["Overview", "Dataset or Prompt Description", "Model Design and Training Approach",
                "Output Evaluation and Interpretation", "Ethical Considerations and Responsible Use",
                "Limitations and Future Improvements", "References"]
    text = "\n".join(p.text for p in doc.paragraphs)
    for heading in headings:
        assert heading in text
    assert len(doc.inline_shapes) == 6
    assert "S00" in text and "synthetic" in text and "not" in text
    assert "Kingma" in text and "Gebru" in text and "Theis" in text
    assert "AI assistance" in text
    pdf = (ROOT / "module_summary.pdf").read_bytes()
    assert pdf.startswith(b"%PDF")
    assert "pending" not in (ROOT / "module_summary.md").read_text().lower()


def test_plain_language_is_concrete_and_jargon_free():
    text = (ROOT / "module_summary.md").read_text()
    plain = text.split("## Interpretation for a Non-Technical Audience\n\n", 1)[1].split("\n\n## ", 1)[0]
    for word in ["latent", "kl", "elbo", "likelihood", "posterior", "variational", "bernoulli"]:
        assert not re.search(r"\b" + word + r"\b", plain.lower())
    assert "64" in plain and "S00" in plain and "person" in plain
    assert "et al." not in plain and "n.d." not in plain


def test_actual_environment_freeze_and_template_audit():
    actual = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    assert actual == (ROOT / "requirements.txt").read_text()
    audit = read_json("report_template_audit.json")
    assert audit["styles_preserved"] and audit["section_preserved"] and audit["headers_preserved"]
    assert audit["omitted_fields"] == ["Instructor", "Due date"]
    assert "torch==2.15.0+cpu" in actual
