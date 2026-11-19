# Dataset provenance and use

The real human-created input is **MNIST handwritten digits**, not Fashion-MNIST.
Prior capstones used Titanic, Student Performance, Bike Sharing and Fashion-MNIST;
none of those files is reused here.

- Original creators: Yann LeCun, Corinna Cortes and Christopher J. C. Burges.
- Original source: http://yann.lecun.com/exdb/mnist/
- Verified public catalog: https://www.tensorflow.org/datasets/catalog/mnist
- Verified rights/attribution: https://keras.io/api/datasets/mnist/
- Actual downloads: https://storage.googleapis.com/cvdf-datasets/mnist/
  plus each of the four filenames in `outputs/dataset_manifest.json`.
- Original files are included unchanged. Manifest records published MD5 checks,
  computed SHA-256, bytes and exact download URL.

60,000 training and 10,000 test images are 28×28 grayscale uint8 with labels 0–9.
Seeded split gives 55,000 training, 5,000 validation and 10,000 official test.
Labels inspect coverage and select reconstruction illustrations only.
The unconditional VAE never uses labels as inputs or loss targets.

Preprocessing divides by 255 then thresholds at >=0.5 to form binary float32
observations. No resizing, augmentation, imputation, learned normalization or
synthetic inputs. Original compressed files are not modified.

The notebook found no exact repeats in the original official sets and no
grayscale cross-set matches. Binarization makes **three validation** and
**eight test** images match training inputs exactly. These are disclosed
preprocessing collisions, not deleted records. Writer identities and demographic
attributes are absent from these IDX files; writer independence, near-duplicate
overlap and population representation were not tested.

Generated output arrays, images and weights are model results, not original
dataset observations. Both the 64 decoder-mean images and the 64 literal
binary samples are clearly identified as synthetic. Preserve source
attribution and follow CC BY-SA 3.0 terms for data redistribution/adaptation.
