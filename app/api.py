"""FoodLens - FastAPI mikroservis (Asama 2-API).

Arkadasinin iskeleti (yemek_api.txt -> FastAPI) korunur; /predict gercek foodlens
akisina baglanir:

    bytes -> PIL -> foodlens.data.preprocess_image (eval transform, augment'siz)
          -> EfficientNet-B0 (best.pt) -> top-5 softmax -> JSON

best.pt YOKSA: arkadasinin dondugu gibi MOCK mod - servis yine ayakta calisir.
Checkpoints otomatik bulunur: env FOODLENS_CHECKPOINT, yoksa runs/*/best.pt (en yeni).

Checkpoint contract (Aşama 3 train.py bunu yazar):
    {"model_state", "classes", "arch", "num_classes", "image_size", "config"}

Calistirma:
    uvicorn app.api:app --host 127.0.0.1 --port 8000
    python app/api.py
Swagger UI: /docs
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# app/ altinda dogrudan calistirilinca src/ yolunu ekle (pip install -e gerekmez)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from foodlens.config import load_config  # noqa: E402
from foodlens.data import preprocess_image  # noqa: E402
from foodlens.explain import explain_image, resolve_target_layer  # noqa: E402

# --- MOCK cevabi (arkadasinin iskeletindeki birebir ayni alanlar) ---
_MOCK = {
    "tahmin_edilen_yemek": "Pizza",
    "dogruluk_orani": 94.5,
    "durum": "Basarili (mock)",
    "kaynak": "mock",
}

app = FastAPI(
    title="Yemek Sınıflandırma API",
    description="Yemek fotoğraflarını EfficientNet-B0 modeliyle sınıflandıran mikroservis",
    version="1.0.0",
)

# --- Kabuk durumu (tembel yukleme) ---
_STATE: dict[str, Any] = {
    "loaded": False,
    "model": None,
    "classes": None,
    "image_size": 224,
    "checkpoint": None,
}


def find_checkpoint() -> Path | None:
    """Checkpoint'i bul: env FOODLENS_CHECKPOINT -> runs/*/best.pt (en yeni run)."""
    env = os.environ.get("FOODLENS_CHECKPOINT")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    runs = _ROOT / "runs"
    if runs.is_dir():
        candidates = sorted(runs.glob("*/best.pt"))
        if candidates:
            return candidates[-1]  # run adlari <tarih>_<ad> -> leksikografik = kronolojik
    return None


def load_state(ckpt: Path | None) -> dict[str, Any]:
    """Checkpoint'i yukler; None ise mock durumu dondurur (model=None)."""
    from foodlens.model import build_model

    if ckpt is None or not ckpt.exists():
        return {"model": None, "classes": None, "image_size": 224, "checkpoint": None}

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    num_classes = int(blob.get("num_classes", len(blob.get("classes", []))))
    model = build_model(num_classes=num_classes, weights=None, dropout=0.2)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return {
        "model": model,
        "classes": list(blob.get("classes", [])),
        "image_size": int(blob.get("image_size", 224)),
        "checkpoint": str(ckpt),
    }


def get_state() -> dict[str, Any]:
    """Tembel yukleme: ilk cagrida checkpoint'i arar/yukler, sonra onbellegi dondurur."""
    if not _STATE["loaded"]:
        _STATE.update(load_state(find_checkpoint()))
        _STATE["loaded"] = True
    return _STATE


def reset_state() -> None:
    """Testler icin onbellegi sifirla (sonraki cagrida yeniden yukler)."""
    _STATE.update(
        loaded=False, model=None, classes=None, image_size=224, checkpoint=None
    )


def _predict(img: Image.Image, st: dict[str, Any]) -> dict[str, Any]:
    """Gercek tahmin: preprocess -> model -> top-5 softmax (JSON'a hazir dict)."""
    x = preprocess_image(img, st["image_size"]).unsqueeze(0)  # (1,C,H,W)
    with torch.no_grad():
        probs = torch.softmax(st["model"](x), dim=1)[0]
    k = min(5, len(st["classes"]))
    top = torch.topk(probs, k=k)
    tahminler = [
        {"sinif": st["classes"][int(i)], "olasilik": round(float(probs[int(i)]) * 100, 2)}
        for i in top.indices
    ]
    best_i = int(top.indices[0])
    return {
        "tahmin_edilen_yemek": st["classes"][best_i],
        "dogruluk_orani": round(float(top.values[0]) * 100, 2),
        "durum": "Basarili",
        "kaynak": "model",
        "tahminler": tahminler,
        "checkpoint": st["checkpoint"],
    }


def _gradcam_layer_name() -> str:
    """Grad-CAM hedef katmani: config.gradio.gradcam_layer -> 'features.8'."""
    try:
        cfg = load_config(_ROOT / "configs" / "default.yaml")
        return str(cfg.get("gradio", {}).get("gradcam_layer", "features.8"))
    except Exception:
        return "features.8"


def _explain(img: Image.Image, st: dict[str, Any]) -> dict[str, Any]:
    """Grad-CAM + top-5: overlay PNG'yi base64 olarak dondurur."""
    layer_name = _gradcam_layer_name()
    layer = resolve_target_layer(st["model"], layer_name)
    out = explain_image(
        st["model"],
        img,
        st["classes"],
        image_size=st["image_size"],
        target_layer=layer,
        topk=5,
    )

    buf = io.BytesIO()
    out["overlay"].save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    tahminler = [
        {"sinif": name, "olasilik": round(prob * 100, 2)}
        for name, prob in out["predictions"]
    ]
    en_iyi, en_iyi_p = out["predictions"][0]
    return {
        "tahmin_edilen_yemek": en_iyi,
        "dogruluk_orani": round(en_iyi_p * 100, 2),
        "durum": "Basarili",
        "kaynak": "model",
        "tahminler": tahminler,
        "gradcam_katmani": layer_name,
        "gradcam_gorsel": f"data:image/png;base64,{png_b64}",
        "checkpoint": st["checkpoint"],
    }


async def _read_upload(file: UploadFile):
    """Dosyayi dogrular ve PIL RGB'ye cevirir. Hata: (None, JSONResponse)."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        return None, JSONResponse(
            status_code=415,
            content={"hata": "Desteklenmeyen dosya formatı. Lütfen JPG veya PNG yükleyin."},
        )

    data = await file.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB"), None
    except Exception:
        return None, JSONResponse(
            status_code=400,
            content={"hata": "Görsel açılamadı. Geçerli bir görüntü dosyası yükleyin."},
        )


@app.get("/")
async def root():
    return {
        "mesaj": (
            "Yemek Sınıflandırma API'sine Hoş Geldiniz! "
            "Swagger UI için /docs adresine gidin."
        )
    }


@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    img, err = await _read_upload(file)
    if err is not None:
        return err

    st = get_state()
    if st["model"] is None:
        body = dict(_MOCK)  # checkpoint yok -> arkadasinin mock donusu
    else:
        body = _predict(img, st)

    body["dosya_adi"] = file.filename
    return body


@app.post("/explain")
async def explain_food(file: UploadFile = File(...)):
    """Grad-CAM aciklamasi: top-5 + isi haritasi (base64 PNG). Model yoksa 503."""
    img, err = await _read_upload(file)
    if err is not None:
        return err

    st = get_state()
    if st["model"] is None:
        return JSONResponse(
            status_code=503,
            content={
                "hata": (
                    "Grad-CAM için eğitilmiş model gerekli. "
                    "runs/<tarih>_<ad>/best.pt indirin veya FOODLENS_CHECKPOINT ayarlayın."
                ),
                "durum": "Model yok",
                "kaynak": "mock",
            },
        )

    body = _explain(img, st)
    body["dosya_adi"] = file.filename
    return body


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
