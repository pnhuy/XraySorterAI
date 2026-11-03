---
license: mit
library_name: onnx
pipeline_tag: image-classification
tags:
  - onnx
  - onnxruntime
  - radiology
  - musculoskeletal
  - x-ray
  - xception
---

# XraySorterAI (ONNX)

ONNX export of the XraySorterAI Xception classifiers for **projection** (45 classes) and **body side** in
musculoskeletal radiographs. Original work: <https://github.com/maxrusse/XraySorterAI>.

> ⚠️ **Not for medical use.** Research/demonstration only; not for clinical or diagnostic decisions.
> Make sure you have the rights to any images you use and that they contain no personal identifiers.

## Files

| File | Description |
|---|---|
| `onnx/projection.onnx` | projection classifier, 45 outputs |
| `onnx/side.onnx` | body-side classifier, 12 outputs (labels alternate right/left, as in the original code) |
| `config.json` | labels, I/O spec, SHA-256 of each model |
| `inference.py` | TensorFlow-free inference (ONNX Runtime + OpenCV + NumPy) |

## Model I/O

- Input `pixels`: `float32 [N, 256, 256, 3]` (NHWC), values 0..255, dynamic batch.
  Xception normalisation (`x/127.5 - 1`) is **inside** the graph.
- Output: softmax probabilities.
- The 3 channels are produced by `inference.preprocess` (adaptive threshold, raw, inverted) — use it rather than a plain RGB image.

## Usage

```bash
pip install -r requirements.txt
python inference.py scan.dcm scan.png scan.nii.gz
```

```python
from inference import XraySorter
sorter = XraySorter(".")
print(sorter.predict("scan.dcm"))   # also .png/.nii.gz, PIL.Image, pydicom Dataset, NumPy array
# {'projection': {'label': 'ap_knee', 'confidence': 0.99}, 'side': {'label': 'bodyside_left', 'confidence': 0.98}}
```

Efficiency notes: sessions are built once, preprocessing is shared by both models, batches are supported,
no TensorFlow / tf-explain dependency (Grad-CAM is not included in this export).

## Conversion

Produced with `export.py` (tf2onnx, opset 17); outputs were verified against the Keras models.
