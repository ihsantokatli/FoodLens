"""Asama 4 UI testleri: Gradio callback (checkpoint yok -> bilgi mesaji)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import gradio_app  # noqa: E402


def test_predict_without_checkpoint_returns_info(monkeypatch, tmp_path):
    """Checkpoint yoksa arayuz cokmez; aciklayici mesaj + bos cikti doner."""
    monkeypatch.setenv("FOODLENS_CHECKPOINT", str(tmp_path / "yok.pt"))
    gradio_app.reset_state()

    label, overlay, info = gradio_app.predict(None)

    assert overlay is None
    assert "Checkpoint bulunamadi" in label
    assert "model bulunamadı" in info
    gradio_app.reset_state()


def test_build_demo_creates_blocks():
    demo = gradio_app.build_demo()
    assert type(demo).__name__ == "Blocks"
