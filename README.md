
# A Deep Learning Approach for Projection and Body Side Classification in Musculoskeletal Radiographs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxrusse/XraySorterAI/blob/master/XraySorterAI_Demo.ipynb)


## Brief:
This project introduces a deep learning solution to classify radiographic projection and body side in musculoskeletal radiographs. Utilizing the Xception architecture, the models achieve high accuracy, offering potential improvements in radiologic workflow and patient care.

## ⚠️ Important Disclaimer:
Usage Rights and Privacy Concerns: Ensure you have the appropriate rights to use any X-ray images before uploading or using them with this model. Remove any personally identifiable information from the images. Be aware of the ethical and legal implications related to patient privacy and data protection.

Not for Medical Use: The predictions and visualizations provided by this model are strictly for demonstration and research purposes. They should not be used for clinical or diagnostic purposes and do not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding any medical condition.

## Download and Installation:

- **Clone the GitHub repository**:
```bash
git clone https://github.com/maxrusse/XraySorterAI.git
```

- **Download the Repository**:
  1. Navigate to the main page of the repository at https://github.com/maxrusse/XraySorterAI.
  2. Click "Code", then "Download ZIP".
  3. Extract the ZIP to your desired location.

## Usage:


**Install required libarys**:
- TensorFlow version: 2.13.0
- tf-explain version: 0.3.1
- nibabel version: 4.0.2
- cv2 version: 4.8.0
- numpy version: 1.23.5
- Python version: 3.10.12

## Integration

1. Navigate to the "models" directory to access the pretrained models.
2. Integrate the models as per your requirements, exemplified by the "visualize_prediction" functions in "analyse.py".

## Quick Start with Google Colab:

For a hands-on demo, [click here](https://colab.research.google.com/github/maxrusse/XraySorterAI/blob/master/XraySorterAI_Demo.ipynb)

## ONNX export, inference and Hugging Face upload

The Keras models in `models/` can be converted to ONNX for fast, TensorFlow-free inference. The converted,
upload-ready package lives in `dist/` and is published at <https://huggingface.co/pnhuy/xraysorterai>.

```
export.py              Keras -> ONNX converter, builds dist/
hf_assets/             hand-written files copied into dist/ (inference.py, model card, requirements, license)
requirements-export.txt  dependencies for export.py only
upload.sh              uploads dist/ to Hugging Face
samples/               local test radiographs (git-ignored)
dist/                  generated output (git-ignored): onnx/, config.json, inference.py, README.md, ...
```

### 1. Create `dist/`

Needs Python 3.10/3.11 (TensorFlow 2.15). Skip this step if you only want to run or upload an existing `dist/`.

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements-export.txt -r dist/requirements.txt   # or: python -m pip install ...
python export.py            # optional: --opset 17 --out dist
```

`export.py` loads `models/model_proj` and `models/model_side`, folds Xception preprocessing into the graph,
converts with tf2onnx (opset 17, dynamic batch), checks each ONNX model against Keras, then writes:

```
dist/onnx/projection.onnx   dist/onnx/side.onnx   dist/config.json (labels, I/O, SHA-256)
dist/inference.py  dist/README.md (model card)  dist/requirements.txt  dist/LICENSE  dist/.gitattributes
dist/samples/              (if ./samples exists)
```

`dist/` is wiped and rebuilt on every run, so edit `hf_assets/` (not `dist/`) and rerun `export.py`.

### 2. Run inference

```bash
uv pip install -r dist/requirements.txt          # NOTE: a plain `pip` inside a uv venv may install elsewhere
python dist/inference.py scan.dcm scan.png scan.nii.gz
```

```python
import sys; sys.path.insert(0, "dist")
from inference import XraySorter

sorter = XraySorter("dist")
sorter.predict("scan.dcm")            # path: .dcm, .png, .jpg, .nii/.nii.gz
sorter.predict(pil_image)             # PIL.Image
sorter.predict(pydicom.dcmread(p))    # pydicom Dataset
sorter.predict(np_array)              # 2-D (or HxWx3) NumPy array
sorter.predict_batch([a, b, c])       # any mix, one preprocessing pass shared by both models
```

Output: `{"projection": {"label": ..., "confidence": ...}, "side": {...}}`.

Inputs are handled for you: 8-bit images are promoted to the ~12-bit range the models were trained on, and
upright images are converted to the NIfTI (x, y) orientation. Without these conversions AP views and
right/left are unreliable (details in `dist/README.md`).

Smoke test on the sample radiographs (15/16 projections and 4/4 sides expected):

```bash
python samples/run_samples.py
```

### 3. Upload to Hugging Face

```bash
pip install -U huggingface_hub
hf auth login                  # token with write access (older CLI: huggingface-cli login), or export HF_TOKEN=...
./upload.sh --dry-run          # show the exact upload command, upload nothing
./upload.sh                    # upload dist/ to pnhuy/xraysorterai
./upload.sh -m "Update inference"
HF_REPO=<user>/<other-repo> ./upload.sh
```

`upload.sh` checks that `dist/` is complete, removes `__pycache__`, and publishes only the samples listed in
`samples/manifest.json` (those have a documented source and license); other files in `samples/` stay local.
Before publishing, put the real copyright holder in `hf_assets/LICENSE` and rerun `export.py`.

### 4. Load from Hugging Face and run inference

```python
import sys
from huggingface_hub import snapshot_download

repo_dir = snapshot_download("pnhuy/xraysorterai")     # cached after the first call
sys.path.insert(0, repo_dir)
from inference import XraySorter

print(XraySorter(repo_dir).predict("scan.png"))
```

Dependencies: `pip install huggingface_hub onnxruntime numpy opencv-python-headless pydicom nibabel Pillow`.

## License:

This project is licensed under the MIT License. Please refer to the LICENSE file in the repository for more details.


[![DOI](https://zenodo.org/badge/691215990.svg)](https://zenodo.org/badge/latestdoi/691215990)
