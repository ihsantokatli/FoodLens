"""Test ortami: src/ ve scripts/ yolunu ekle + toy Food-101 fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from make_toy_food101 import make_toy_food101  # noqa: E402


@pytest.fixture(scope="session")
def toy_food101(tmp_path_factory) -> Path:
    """Kucuk toy Food-101 (food-101/ baz klasoru) - yerel, hizli, deterministik."""
    out = tmp_path_factory.mktemp("toy")
    return make_toy_food101(
        out_dir=out,
        num_classes=5,
        train_per_class=30,
        test_per_class=10,
        image_size=64,
        seed=42,
    )


@pytest.fixture()
def augment_cfg() -> dict:
    return {
        "hflip_p": 0.5,
        "rrc_scale": [0.8, 1.0],
        "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
    }
