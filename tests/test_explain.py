"""Asama 4 testleri: Grad-CAM (hedef katman, cam, overlay, explain_image)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from foodlens.explain import (
    compute_gradcam,
    explain_image,
    overlay_cam,
    resolve_target_layer,
)
from foodlens.model import build_model


def _model(num_classes: int = 5):
    """Kucuk, indirmesiz EfficientNet-B0 (testler hizli ve cevrimdisi)."""
    torch.manual_seed(0)
    return build_model(num_classes=num_classes, weights=None, dropout=0.2).eval()


def _classes(n: int = 5) -> list[str]:
    return [f"class_{i:02d}" for i in range(n)]


# ---------------------------------------------------------------------------
# Hedef katman cozumleme: varsayilan + nokta/indeks yollari
# ---------------------------------------------------------------------------
def test_resolve_target_layer_default_is_last_conv_block():
    m = _model()
    assert resolve_target_layer(m) is m.features[-1]
    # EfficientNet-B0'da features[-1] == features[8] (son konvolusyon blogu)
    assert m.features[-1] is m.features[8]


def test_resolve_target_layer_supports_paths():
    m = _model()
    assert resolve_target_layer(m, "features.8") is m.features[8]
    assert resolve_target_layer(m, "features[-1]") is m.features[-1]


# ---------------------------------------------------------------------------
# Grad-CAM: sekil, aralik, olasiliklar, sinif secimi
# ---------------------------------------------------------------------------
def test_compute_gradcam_shape_range_and_probs():
    m = _model()
    x = torch.randn(1, 3, 64, 64)
    cam, idx, probs = compute_gradcam(m, x)

    assert cam.shape == (64, 64)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
    assert float(cam.max()) > 0.99  # normalize edildi, tamamen bos degil

    assert probs.shape == (5,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert idx == int(probs.argmax())  # varsayilan: top-1


def test_compute_gradcam_accepts_3d_input_and_explicit_class():
    m = _model()
    cam, idx, probs = compute_gradcam(m, torch.randn(3, 64, 64), class_idx=2)
    assert idx == 2
    assert cam.shape == (64, 64)
    assert probs.shape == (5,)


def test_compute_gradcam_does_not_pollute_param_grads():
    """autograd.grad(score, act) parametre .grad'larini kirletmemeli."""
    m = _model()
    compute_gradcam(m, torch.randn(1, 3, 64, 64))
    assert all(p.grad is None for p in m.parameters())


def test_compute_gradcam_restores_train_mode():
    m = _model()
    m.train()
    compute_gradcam(m, torch.randn(1, 3, 64, 64))
    assert m.training is True  # eval'e alip geri birakti


# ---------------------------------------------------------------------------
# Overlay: RGB + orijinal boyut korunur
# ---------------------------------------------------------------------------
def test_overlay_cam_rgb_and_original_size():
    img = Image.new("RGB", (120, 80), (128, 128, 128))
    cam = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    out = overlay_cam(img, cam, alpha=0.5)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == (120, 80)  # cam kucuk olsa da gorsel boyutu korunur


# ---------------------------------------------------------------------------
# explain_image: top-k + overlay + tutarlilik
# ---------------------------------------------------------------------------
def test_explain_image_returns_sorted_topk_and_overlay():
    m = _model()
    img = Image.new("RGB", (300, 200), (200, 60, 30))
    out = explain_image(m, img, _classes(), image_size=64, topk=5)

    assert set(out) == {"overlay", "cam", "class_idx", "predictions"}
    assert out["cam"].shape == (64, 64)
    assert out["overlay"].size == (300, 200)
    assert out["overlay"].mode == "RGB"

    preds = out["predictions"]
    assert len(preds) == 5
    olasiliklar = [p for _, p in preds]
    assert olasiliklar == sorted(olasiliklar, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in olasiliklar)

    # Grad-CAM'in acikladigi sinif = top-1 tahmin
    assert preds[0][0] == _classes()[out["class_idx"]]
