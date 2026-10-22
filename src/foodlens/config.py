"""FoodLens - konfigurasyon yukleme yardimcilari.

YAML config dosyalarindaki tum yollar "/" ayraciyla yazilir; burada Path nesnesine
cevrilir (Windows + Colab uyumu). Yollar proje kokune goreli cozulur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML konfigurasyonu yukler ve dict olarak dondurur."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Konfigurasyon bulunamadi: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Konfigurasyon bir dict degil: {path}")
    return cfg


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    """Config'teki '/' ayracili yolu Path'e cevirir; verilmisse base'e goreli cozer.

    Ornek: ("data/val_indices.json", "C:/proj") -> Windows'ta C:\\proj\\data\\val_indices.json
    """
    p = Path(str(value))
    if not p.is_absolute() and base is not None:
        p = Path(base) / p
    return p


def project_root_from(config_path: str | Path) -> Path:
    """Config dosyasinin ust klasoru kullanici tarafindan proje koku verilirse
    yardimci. Config configs/ altinda oldugu icin bir ust klasor kokdur."""
    return Path(config_path).resolve().parent.parent
