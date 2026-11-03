"""TensorFlow-free inference for XraySorterAI (ONNX Runtime + OpenCV + NumPy).

    from inference import XraySorter
    sorter = XraySorter(".")                  # folder containing config.json + onnx/
    sorter.predict("scan.dcm")                # path: .dcm, .png, .nii/.nii.gz, jpg, ...
    sorter.predict(pil_image)                 # PIL.Image
    sorter.predict(pydicom.dcmread("a.dcm"))  # pydicom Dataset
    sorter.predict(np_array)                  # 2-D (or HxWx3 RGB) NumPy array
    sorter.predict_batch([img1, img2])        # any mix of the above

CLI:  python inference.py scan1.dcm scan2.png ...

Images are preprocessed once and shared by both models; sessions are created once.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
import onnxruntime as ort

cv2.setNumThreads(1)  # avoid oversubscription with ORT threads
SIZE = 256

ImageLike = Union[str, os.PathLike, np.ndarray]


# --------------------------------------------------------------------------- IO
def _align_nifti(image):
    """Reorient a NIfTI volume to a canonical axis order (same logic as the original code)."""
    import nibabel as nib
    affine = image.affine
    max_idx = np.argmax(np.abs(affine[0:3, 0:3]), axis=0)
    signs = np.sign(affine[max_idx, np.arange(3)])
    inv = list(np.argsort(max_idx))
    if len(image.shape) > 3:
        inv += list(range(3, len(image.shape)))
    data = np.transpose(image.get_fdata(), inv)
    for i in range(3):
        if signs[inv[i]] < 0:
            data = np.flip(data, axis=i)
    return data


def _dicom_to_array(ds) -> np.ndarray:
    """pydicom Dataset -> 2-D float32 (first frame, rescaled, MONOCHROME1 inverted)."""
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":   # invert over full stored range
        arr = float(2 ** int(getattr(ds, "BitsStored", 16)) - 1) - arr
    if arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 1:      # multi-frame -> first frame
        arr = arr[0]
    elif arr.ndim == 3:                                               # colour -> gray
        arr = arr.mean(axis=-1)
    arr = arr * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    return arr


def load_image(src: ImageLike) -> np.ndarray:
    """Return a 2-D float32 array of raw pixel values.

    Accepts a file path (.nii/.nii.gz, .dcm/.dicom, .png and other OpenCV formats),
    a PIL.Image, a pydicom Dataset, or a NumPy array (2-D, HxWx1, or HxWx3 RGB).
    """
    if isinstance(src, (str, os.PathLike)):
        path = str(src)
        low = path.lower()
        if low.endswith((".nii", ".nii.gz")):
            import nibabel as nib
            arr = _align_nifti(nib.load(path))
            if arr.ndim == 4:
                arr = arr[:, :, :, 1]
        elif low.endswith((".dcm", ".dicom", ".ima")):
            import pydicom
            arr = _dicom_to_array(pydicom.dcmread(path))
        else:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keeps 16-bit PNGs
            if arr is None:
                raise ValueError(f"cannot read image: {path}")
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY if arr.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    elif isinstance(src, np.ndarray):
        arr = src
    elif type(src).__module__.startswith("PIL."):
        arr = np.asarray(src.convert("L") if src.mode in ("RGB", "RGBA", "P", "LA", "CMYK") else src)
    elif hasattr(src, "pixel_array"):  # pydicom Dataset
        arr = _dicom_to_array(src)
    else:
        raise TypeError(f"unsupported image type: {type(src)!r}")
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and not isinstance(src, (str, os.PathLike)):
        arr = cv2.cvtColor(np.ascontiguousarray(arr[..., :3]).astype(np.float32), cv2.COLOR_RGB2GRAY)
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {arr.shape}")
    return arr


# ------------------------------------------------------------------ preprocessing
def preprocess(img: np.ndarray) -> np.ndarray:
    """2-D raw image -> (256, 256, 3) float32 in 0..255 (the ONNX graph normalises further)."""
    u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    thr = cv2.adaptiveThreshold(cv2.medianBlur(u8, 3), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 15, 2)
    rgb = np.dstack((thr.astype(np.float32), img, 255.0 - img))
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    return cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).astype(np.float32)


# ------------------------------------------------------------------------ model
class XraySorter:
    def __init__(self, repo_dir: str | os.PathLike = ".", providers: Sequence[str] | None = None,
                 intra_op_threads: int = 0):
        repo = Path(repo_dir)
        cfg = json.loads((repo / "config.json").read_text())
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = intra_op_threads
        providers = providers or ["CPUExecutionProvider"]
        self.labels, self.sessions = {}, {}
        for name, m in cfg["models"].items():
            self.sessions[name] = ort.InferenceSession(str(repo / m["file"]), opts, providers=providers)
            self.labels[name] = m["labels"]
        self._input = {n: s.get_inputs()[0].name for n, s in self.sessions.items()}

    def predict_batch(self, images: Sequence[ImageLike]) -> list[dict]:
        batch = np.stack([preprocess(load_image(i)) for i in images])  # shared by both models
        out = [{} for _ in images]
        for name, sess in self.sessions.items():
            probs = sess.run(None, {self._input[name]: batch})[0]
            for k, p in enumerate(probs):
                j = int(p.argmax())
                out[k][name] = {"label": self.labels[name][j], "confidence": float(p[j])}
        return out

    def predict(self, image: ImageLike) -> dict:
        return self.predict_batch([image])[0]


if __name__ == "__main__":
    import sys
    s = XraySorter(Path(__file__).parent)
    for f in sys.argv[1:]:
        print(f, json.dumps(s.predict(f)))
