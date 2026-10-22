"""FoodLens - Colab on-dogrulama (Asama 3 oncesi "preflight").

Tam Food-101 indirildikten SONRA, egitim baslamadan ONCE beklentileri kontrol eder.
Tutmazsa Food101ValidationError firlatir -> egitim baslatilmaz.

Beklenen (dengeli Food-101):
    101 sinif, sinif basina 750 resmi train + 250 test
    resmi train toplam = 101 * 750 = 75.750
    resmi test  toplam = 101 * 250 = 25.250
    train icinden stratified %10 val -> 101 * 75  =  7.575
    kalan train                          -> 101 * 675 = 68.175

Dogrulama META uzerinden calisir (gorsel yuklemez) -> saniyeler sürer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .data import read_food101_classes, read_food101_split, samples_from_meta, stratified_split

# Gercek Food-101 icin beklenen sayilar (dengeli).
FOOD101_EXPECTED: dict[str, int] = {
    "classes": 101,
    "train_total": 75_750,
    "test_total": 25_250,
    "val": 7_575,
    "train_remaining": 68_175,
}


class Food101ValidationError(RuntimeError):
    """Beklenen Food-101 sayilari tutmadiginda firlatilir."""


def _resolve_base(root: Path) -> Path:
    """root/food-101 varsa onu, yoksa root'un kendisini (meta/ + images/ iceren) kullan."""
    root = Path(root)
    nested = root / "food-101"
    return nested if nested.exists() else root


def inspect_food101(root: str | Path, val_ratio: float = 0.10, seed: int = 42) -> dict[str, Any]:
    """Food-101 meta'ini okuyup toplamlari + per-class sayilari dondurur (gorsel yuklemez)."""
    base = _resolve_base(Path(root))
    meta_dir = base / "meta"
    images_dir = base / "images"

    classes = read_food101_classes(meta_dir, "train")
    train_meta = read_food101_split(meta_dir, "train")
    test_meta = read_food101_split(meta_dir, "test")

    _, tr_labels, _ = samples_from_meta(train_meta, images_dir, classes)
    _, te_labels, _ = samples_from_meta(test_meta, images_dir, classes)
    train_idx, val_idx = stratified_split(tr_labels, val_ratio, seed)

    return {
        "base": str(base),
        "num_classes": len(classes),
        "classes": classes,
        "train_total": len(tr_labels),
        "test_total": len(te_labels),
        "val_total": len(val_idx),
        "train_remaining": len(train_idx),
        "val_ratio": val_ratio,
        "seed": seed,
        "per_class_official_train": dict(Counter(tr_labels)),
        "per_class_test": dict(Counter(te_labels)),
        "per_class_val": dict(Counter(tr_labels[i] for i in val_idx)),
        "per_class_remaining": dict(Counter(tr_labels[i] for i in train_idx)),
    }


def validate_food101(
    root: str | Path,
    val_ratio: float = 0.10,
    seed: int = 42,
    expected: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Beklenen sayilari kontrol eder; tutmazsa Food101ValidationError firlatir.

    expected ile (testlerde oyuncak veri icin) varsayilanlar ezilebilir.
    Basarili olursa inspect sonucu dondurur.
    """
    exp = {**FOOD101_EXPECTED, **(expected or {})}
    info = inspect_food101(root, val_ratio, seed)

    checks: list[tuple[str, int, int]] = [
        ("sinif sayisi", info["num_classes"], exp["classes"]),
        ("resmi train toplam", info["train_total"], exp["train_total"]),
        ("resmi test toplam", info["test_total"], exp["test_total"]),
        ("val (stratified)", info["val_total"], exp["val"]),
        ("kalan train", info["train_remaining"], exp["train_remaining"]),
    ]

    failures: list[str] = []
    for name, got, want in checks:
        if got != want:
            failures.append(f"  - {name}: beklenen {want:,}, bulunan {got:,}")

    # Per-class denge: her sinif ayni sayida ornek icermeli (Food-101 dengelidir).
    balance_keys = [
        ("per_class_official_train", "sinif basina resmi train"),
        ("per_class_test", "sinif basina test"),
        ("per_class_val", "sinif basina val"),
        ("per_class_remaining", "sinif basina kalan train"),
    ]
    for key, label in balance_keys:
        vals = sorted(set(info[key].values()))
        if len(vals) != 1:
            failures.append(f"  - {label}: DENGESIZ degerler {vals}")

    if failures:
        raise Food101ValidationError(
            "Food-101 on-dogrulama BASARISIZ (egitim baslatilmadi):\n" + "\n".join(failures)
        )
    return info


def format_validation_report(info: dict[str, Any]) -> str:
    """Dogrulama sonucunu insan-okur metin olarak formatlar."""
    pcv = sorted(set(info["per_class_val"].values()))
    pct = sorted(set(info["per_class_test"].values()))
    lines = [
        f"Food-101 on-dogrulama OK  (base={info['base']})",
        f"  sinif sayisi        : {info['num_classes']}",
        f"  resmi train toplam  : {info['train_total']:,}",
        f"  resmi test  toplam  : {info['test_total']:,}",
        f"  val (stratified %)  : {info['val_total']:,}   (val_ratio={info['val_ratio']}, seed={info['seed']})",
        f"  kalan train         : {info['train_remaining']:,}",
        f"  sinif basina val    : {pcv}",
        f"  sinif basina test   : {pct}",
    ]
    return "\n".join(lines)
