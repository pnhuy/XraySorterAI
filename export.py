"""Convert the XraySorterAI Keras SavedModels to ONNX and assemble ./dist
as a ready-to-upload Hugging Face repository.

Usage:
    python export.py [--opset 17] [--out dist]

Requires: tensorflow==2.15, tf2onnx, onnx==1.16.2, onnxruntime, numpy<2
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

ROOT = Path(__file__).resolve().parent

TAGS_PROJ = ['ap_clavicle', 'obl_clavicle', 'ap_ac', 'ap_shoulder', 'y-view', 'axial_shoulder', 'ap_elbow',
             'lat_elbow', 'radial_head', 'ap_wrist', 'lat_wrist', 'ap_hand', 'obl_hand', 'ap_finger', 'lat_finger',
             'ap_thumb', 'lat_thumb', 'ap_leg', 'ap_pelvis', 'ap_hip', 'Lauenstein', 'ap_knee', 'lat_knee',
             'defile', 'ap_ankle', 'lat_ankle', 'lat_calcaneus', 'axial_calcaneus', 'ap_foot', 'lat_foot',
             'obl_foot', 'ap_forefoot', 'obl_forefoot', 'ap_toe', 'lat_toe', 'ap_btoe', 'lat_btoe', 'ap_cspine',
             'lat_cspine', 'dens', 'ap_tspine', 'lat_tspine', 'ap_lspine', 'lat_lspine', 'nasal_bone']
# The side model has 12 outputs; the original code maps them as ['right', 'left'] * 6.
TAGS_SIDE = ['bodyside_right', 'bodyside_left'] * 6

MODELS = {"projection": ("models/model_proj", TAGS_PROJ),
          "side": ("models/model_side", TAGS_SIDE)}


def build_inference_model(saved_model_dir: Path) -> tf.keras.Model:
    """Wrap the trained model so the graph takes raw 0..255 pixels (NHWC float32)
    and has Xception's `preprocess_input` (x / 127.5 - 1) folded in as constants.
    Inference mode is forced (Dropout / BatchNorm frozen)."""
    base = tf.keras.models.load_model(str(saved_model_dir), compile=False)
    inp = tf.keras.Input(shape=(256, 256, 3), dtype=tf.float32, name="pixels")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="xception_preprocess")(inp)
    out = base(x, training=False)
    return tf.keras.Model(inp, out, name=base.name)


def convert(model: tf.keras.Model, path: Path, opset: int) -> None:
    spec = (tf.TensorSpec((None, 256, 256, 3), tf.float32, name="pixels"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset, output_path=str(path))
    m = onnx.load(str(path))  # tf2onnx already folds constants / BN and optimises transposes
    onnx.checker.check_model(m)
    # tag metadata for self-describing files
    m.doc_string = "XraySorterAI - input: float32 NHWC [N,256,256,3] pixel values 0..255; output: softmax probabilities"
    onnx.save(m, str(path))


def verify(keras_model, onnx_path: Path, rtol=1e-3, atol=1e-4) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, size=(4, 256, 256, 3)).astype("float32")
    ref = keras_model(x, training=False).numpy()
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    # dynamic batch check
    got1 = sess.run(None, {sess.get_inputs()[0].name: x[:1]})[0]
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(got1, ref[:1], rtol=rtol, atol=atol)
    assert (got.argmax(1) == ref.argmax(1)).all()
    return float(np.abs(got - ref).max())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    out = (ROOT / args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)

    files = {}
    for name, (rel, tags) in MODELS.items():
        print(f"[{name}] loading {rel}")
        km = build_inference_model(ROOT / rel)
        p = out / "onnx" / f"{name}.onnx"
        print(f"[{name}] converting -> {p.relative_to(ROOT)}")
        convert(km, p, args.opset)
        err = verify(km, p)
        print(f"[{name}] verified vs Keras, max abs diff = {err:.2e}, size = {p.stat().st_size/1e6:.1f} MB")
        files[name] = {"file": f"onnx/{name}.onnx", "labels": tags, "sha256": sha256(p)}

    config = {
        "architecture": "Xception",
        "input": {"name": "pixels", "layout": "NHWC", "shape": ["N", 256, 256, 3], "dtype": "float32",
                  "range": "0..255 (Xception normalisation is inside the graph)",
                  "preprocessing": "see inference.py:preprocess"},
        "output": "softmax probabilities",
        "models": files,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # static repo assets
    assets = ROOT / "hf_assets"
    for f in ("inference.py", "README.md", "requirements.txt", "LICENSE", ".gitattributes"):
        shutil.copy(assets / f, out / f)
    if (ROOT / "samples").is_dir():  # optional, local-only (git-ignored)
        shutil.copytree(ROOT / "samples", out / "samples")
    print("done ->", out)


if __name__ == "__main__":
    main()
