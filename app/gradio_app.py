"""FoodLens - Gradio UI (Asama 4).

Tek gorsel yukle -> top-5 tahmin + Grad-CAM isi haritasi (overlay).
Batch YOK (istek: tek gorsel). Model/checkpoint yuklemesi app.api ile ayni
mantigi kullanir: env FOODLENS_CHECKPOINT -> runs/*/best.pt (en yeni).

Checkpoint yoksa arayuz yine acilir; bilgi mesaji gosterir (API'deki mock
fallback'in UI karsiligi).

Calistirma:
    python app/gradio_app.py
    # veya: python -m app.gradio_app
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from app.api import find_checkpoint, load_state  # noqa: E402
from foodlens.config import load_config  # noqa: E402
from foodlens.explain import explain_image, resolve_target_layer  # noqa: E402

_CONFIG_PATH = _ROOT / "configs" / "default.yaml"

# Tembel yuklenen model durumu (api.py ile ayni sozlesme)
_STATE: dict[str, Any] = {"loaded": False}


def _config() -> dict[str, Any]:
    """Gradio ayarlarini config'ten okur; dosya yoksa guvenli varsayilanlar."""
    try:
        return load_config(_CONFIG_PATH)
    except Exception:
        return {}


def get_state() -> dict[str, Any]:
    """Checkpoint'i ilk cagrida yukler, sonra onbellegi dondurur."""
    if not _STATE["loaded"]:
        _STATE.update(load_state(find_checkpoint()))
        _STATE["loaded"] = True
    return _STATE


def reset_state() -> None:
    """Testler icin onbellegi sifirlar."""
    _STATE.clear()
    _STATE["loaded"] = False


def _target_layer_name() -> str:
    """config: gradio.gradcam_layer -> varsayilan 'features.8'."""
    gcfg = _config().get("gradio", {})
    return str(gcfg.get("gradcam_layer", "features.8"))


def _topk() -> int:
    """config: gradio.topk -> varsayilan 5."""
    gcfg = _config().get("gradio", {})
    return int(gcfg.get("topk", 5))


def predict(pil_img: Image.Image | None):
    """UI callback: tek gorsel -> (top-k sozlugu, overlay PIL).

    Gradio'da Label bekledigi icin {sinif: olasilik} dondurur.
    """
    st = get_state()
    if st["model"] is None:
        return (
            {"Checkpoint bulunamadi": 1.0},
            None,
            "⚠️ Eğitilmiş model bulunamadı. `runs/<tarih>_<ad>/best.pt` indirin "
            "veya `FOODLENS_CHECKPOINT` ortam değişkenini ayarlayın. "
            "(Şu an yalnızca API'de mock yanıt var.)",
        )

    if pil_img is None:
        return {}, None, "Lütfen bir yemek fotoğrafı yükleyin."

    img = pil_img.convert("RGB")
    layer = resolve_target_layer(st["model"], _target_layer_name())
    out = explain_image(
        st["model"],
        img,
        st["classes"],
        image_size=st["image_size"],
        target_layer=layer,
        topk=_topk(),
    )

    label = {name: prob for name, prob in out["predictions"]}
    en_iyi, en_iyi_p = out["predictions"][0]
    bilgi = (
        f"**Tahmin:** {en_iyi}  \n"
        f"**Güven:** {en_iyi_p * 100:.2f}%  \n"
        f"**Grad-CAM katmanı:** {_target_layer_name()}"
    )
    return label, out["overlay"], bilgi


def build_demo() -> gr.Blocks:
    """Gradio arayuzunu kurar (tek gorsel; batch yok)."""
    with gr.Blocks(title="FoodLens — Yemek Sınıflandırma") as demo:
        gr.Markdown(
            "# 🍽️ FoodLens\n"
            "Yemek fotoğrafı yükleyin: **top-5 tahmin** + **Grad-CAM** "
            "ısı haritası (modelin nereye baktığı)."
        )
        with gr.Row():
            with gr.Column():
                giris = gr.Image(type="pil", label="Yemek fotoğrafı (tek görsel)")
                btn = gr.Button("Tahmin Et", variant="primary")
                temizle = gr.Button("Temizle")
            with gr.Column():
                etiket = gr.Label(num_top_classes=_topk(), label="Top-5 tahmin")
                cikti = gr.Image(type="pil", label="Grad-CAM (ısı haritası)")
        bilgi = gr.Markdown()

        btn.click(predict, inputs=giris, outputs=[etiket, cikti, bilgi])
        giris.change(predict, inputs=giris, outputs=[etiket, cikti, bilgi])
        temizle.click(
            lambda: (None, None, None, ""),
            outputs=[giris, etiket, cikti, bilgi],
        )
    return demo


def main() -> int:
    """Arayuzu baslatir (config: gradio.share)."""
    share = bool(_config().get("gradio", {}).get("share", False))
    demo = build_demo()
    demo.launch(share=share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
