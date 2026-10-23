"""Asama 2-API testleri: FastAPI /predict (mock mod + checkpoint'li gercek mod)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `import app.api` icin proje kokunu yola ekle
    sys.path.insert(0, str(ROOT))

from app import api  # noqa: E402
from foodlens.data import preprocess_image  # noqa: E402


@pytest.fixture()
def client():
    api.reset_state()
    with TestClient(api.app) as c:
        yield c
    api.reset_state()


def _png_bytes(size: int = 300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (200, 60, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _post(client: TestClient, data: bytes, ctype: str, name: str = "gorsel.png"):
    return client.post("/predict", files={"file": (name, data, ctype)})


# ---------------------------------------------------------------------------
# / ve format kontrolu
# ---------------------------------------------------------------------------
def test_root_returns_mesaj(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "mesaj" in r.json()


def test_predict_rejects_unsupported_type(client):
    r = _post(client, b"birsey", "application/pdf", name="x.pdf")
    assert r.status_code == 415
    assert "hata" in r.json()


def test_predict_invalid_image_bytes_gives_400(client, monkeypatch):
    monkeypatch.setenv("FOODLENS_CHECKPOINT", str(Path(__file__).parent / "nope.pt"))
    api.reset_state()
    r = _post(client, b"gecersiz-gorsel-verisi", "image/png")
    assert r.status_code == 400
    assert "hata" in r.json()


# ---------------------------------------------------------------------------
# MOCK mod (checkpoint yok) - arkadasinin iskelet cevabi
# ---------------------------------------------------------------------------
def test_predict_mock_when_no_checkpoint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("FOODLENS_CHECKPOINT", str(tmp_path / "yok.pt"))  # dosya yok
    api.reset_state()
    r = _post(client, _png_bytes(), "image/png")
    assert r.status_code == 200
    body = r.json()
    # arkadasinin birebir alanlari
    assert body["dosya_adi"] == "gorsel.png"
    assert body["tahmin_edilen_yemek"] == "Pizza"
    assert body["dogruluk_orani"] == 94.5
    assert body["kaynak"] == "mock"
    assert "tahminler" not in body


# ---------------------------------------------------------------------------
# GERCEK mod (checkpoint var) - contract + top-5
# ---------------------------------------------------------------------------
def _make_checkpoint(tmp_path: Path, num_classes: int = 101) -> Path:
    from foodlens.model import build_model

    classes = [f"class_{i:03d}" for i in range(num_classes)]
    m = build_model(num_classes=num_classes, weights=None, dropout=0.2)
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": m.state_dict(),
            "classes": classes,
            "arch": "efficientnet_b0",
            "num_classes": num_classes,
            "image_size": 224,
            "config": {},
        },
        ckpt,
    )
    return ckpt


def test_predict_real_top5_with_checkpoint(client, monkeypatch, tmp_path):
    ckpt = _make_checkpoint(tmp_path, num_classes=101)
    monkeypatch.setenv("FOODLENS_CHECKPOINT", str(ckpt))
    api.reset_state()
    r = _post(client, _png_bytes(), "image/png")
    assert r.status_code == 200
    body = r.json()
    assert body["kaynak"] == "model"
    assert body["durum"] == "Basarili"
    assert body["checkpoint"] == str(ckpt)
    assert body["tahmin_edilen_yemek"].startswith("class_")
    # top-5
    assert len(body["tahminler"]) == 5
    top5 = body["tahminler"]
    assert top5[0]["sinif"] == body["tahmin_edilen_yemek"]
    # olasiliklar azalan + yuzde
    ol = [t["olasilik"] for t in top5]
    assert ol == sorted(ol, reverse=True)
    assert all(0.0 <= o <= 100.0 for o in ol)
    # olasiliklar toplami (tum 101 sinif) ~%100 icin ilk5 <= 100
    assert top5[0]["sinif"] in [f"class_{i:03d}" for i in range(101)]


# ---------------------------------------------------------------------------
# /explain (Asama 4): Grad-CAM - checkpoint yoksa 503, varsa base64 overlay
# ---------------------------------------------------------------------------
def test_explain_returns_503_without_checkpoint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("FOODLENS_CHECKPOINT", str(tmp_path / "yok.pt"))
    api.reset_state()
    r = client.post(
        "/explain", files={"file": ("yemek.png", _png_bytes(), "image/png")}
    )
    assert r.status_code == 503
    body = r.json()
    assert "hata" in body
    assert body["durum"] == "Model yok"


def test_explain_real_returns_top5_and_base64_overlay(client, monkeypatch, tmp_path):
    import base64

    ckpt = _make_checkpoint(tmp_path, num_classes=5)
    monkeypatch.setenv("FOODLENS_CHECKPOINT", str(ckpt))
    api.reset_state()

    r = client.post(
        "/explain", files={"file": ("yemek.png", _png_bytes(300), "image/png")}
    )
    assert r.status_code == 200
    body = r.json()

    assert body["kaynak"] == "model"
    assert body["durum"] == "Basarili"
    assert body["gradcam_katmani"] == "features.8"
    assert len(body["tahminler"]) == 5
    assert body["tahminler"][0]["sinif"] == body["tahmin_edilen_yemek"]
    assert 0.0 <= body["dogruluk_orani"] <= 100.0

    # base64 PNG gercekten acilabilir ve giris boyutunu korur
    assert body["gradcam_gorsel"].startswith("data:image/png;base64,")
    raw = base64.b64decode(body["gradcam_gorsel"].split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    assert img.mode == "RGB"
    assert img.size == (300, 300)


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------
def test_preprocess_image_shape_and_rgb():
    x = preprocess_image(Image.new("RGB", (300, 200), (10, 20, 30)), image_size=224)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 224, 224)
    # RGBA -> RGB cevrilir
    x2 = preprocess_image(Image.new("RGBA", (64, 64)), image_size=224)
    assert x2.shape == (3, 224, 224)
