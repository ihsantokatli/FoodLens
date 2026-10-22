"""Food-101 biciminde KUCUK oyuncak (toy) kaynak veri uretir - YEREL TEST ICIN.

Gerçek Food-101'i (5.4 GB) indirmeden veri katmaninin akisini calistirmak icin
ayni klasor/meta bicimini kullanir:

    <out>/food-101/
        meta/train.json   {sinif: ["<sinif>/<ad>", ...]}
        meta/test.json    {sinif: ["<sinif>/<ad>", ...]}
        images/<sinif>/<ad>.jpg

Bu kaynak; make_subset.py girdisi, explore_data.py ve testler icin kullanilir.
Kalici/degerlendirme amaci TAŞIMAZ; tam egitim Colab'da gercek Food-101 ile yapilir.

Kullanim:
    python scripts/make_toy_food101.py --out data/toy --num-classes 12 \
        --train-per-class 105 --test-per-class 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Gercek Food-101 sinif adlarini andiran sahte adlar (grid raporu okunur olsun diye)
_TOY_CLASSES = [
    "apple_pie", "burger", "cake", "donut", "ice_cream", "pizza",
    "ramen", "salad", "sushi", "steak", "taco", "pasta",
]


def _make_image(path: Path, base_color: tuple[int, int, int], size: int, rng: random.Random) -> None:
    """Sinif bazli renk + rastgele dikdortgen gorseli (hizli JPEG)."""
    img = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(img)
    for _ in range(6):
        x0 = rng.randint(0, size - 1)
        y0 = rng.randint(0, size - 1)
        x1 = min(size - 1, x0 + rng.randint(size // 6, size // 2))
        y1 = min(size - 1, y0 + rng.randint(size // 6, size // 2))
        shade = tuple(max(0, min(255, c + rng.randint(-70, 70))) for c in base_color)
        draw.rectangle([x0, y0, x1, y1], fill=shade)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=80)


def make_toy_food101(
    out_dir: str | Path,
    num_classes: int = 12,
    train_per_class: int = 105,
    test_per_class: int = 30,
    image_size: int = 128,
    seed: int = 42,
    classes: list[str] | None = None,
) -> Path:
    """Toy Food-101 uretir; food-101 baz klasorunu (meta/ + images/) dondurur."""
    out_dir = Path(out_dir)
    base = out_dir / "food-101"
    meta_dir = base / "meta"
    images_dir = base / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if classes is None:
        classes = _TOY_CLASSES[:num_classes]
        if len(classes) < num_classes:
            classes = classes + [f"class_{i:02d}" for i in range(len(classes), num_classes)]
    classes = classes[:num_classes]

    rng = random.Random(seed)
    train_meta: dict[str, list[str]] = {}
    test_meta: dict[str, list[str]] = {}

    for i, cls in enumerate(classes):
        base_color = ((i * 47) % 200 + 40, (i * 89) % 200 + 30, (i * 131) % 200 + 50)
        tr_rels, te_rels = [], []
        for j in range(train_per_class):
            name = f"{cls}_{j:04d}"
            _make_image(images_dir / cls / f"{name}.jpg", base_color, image_size, rng)
            tr_rels.append(f"{cls}/{name}")
        for j in range(test_per_class):
            name = f"{cls}_test_{j:04d}"
            _make_image(images_dir / cls / f"{name}.jpg", base_color, image_size, rng)
            te_rels.append(f"{cls}/{name}")
        train_meta[cls] = tr_rels
        test_meta[cls] = te_rels

    with (meta_dir / "train.json").open("w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=1)
    with (meta_dir / "test.json").open("w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=1)

    return base


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Food-101 biciminde toy kaynak veri uretir (yerel test)")
    p.add_argument("--out", default="data/toy", help="Cikti kok (icinde food-101/ olusturulur)")
    p.add_argument("--num-classes", type=int, default=12)
    p.add_argument("--train-per-class", type=int, default=105)
    p.add_argument("--test-per-class", type=int, default=30)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    base = make_toy_food101(
        out_dir=args.out,
        num_classes=args.num_classes,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Toy Food-101 hazir: {base}")
    print(f"  meta/train.json, meta/test.json, images/  -> {args.out}/food-101/")


if __name__ == "__main__":
    main()
