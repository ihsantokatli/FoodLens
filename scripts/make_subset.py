"""Food-101'den kucuk smoke alt kume uretir (Asama 1).

- Kaynaktan meta/train.json okur; test split'ine DOKUNMAZ (test degerlendirmesi
  Colab'da tam Food-101 test split'i ile yapilir).
- N sinif x (train_per_class + val_per_class) gorseli deterministik secer ve
  <out>/images/<sinif>/ altina kopyalar.
- val_indices.json YAZAR (foodlens.data.collect_pool ayni sirayi kullanir),
  boylece %80/20 stratified split kalici ve tekrar uretilebilir olur.

Kullanim:
    python scripts/make_subset.py --src data/food-101 \
        --out data/subsets/smoke_10cls --num-classes 10 \
        --train-per-class 80 --val-per-class 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# script'i dogrudan calistirinca src/ yolunu ekle (pip install -e . gerekmez)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodlens.data import (  # noqa: E402
    collect_pool,
    load_or_create_val_indices,
    read_food101_split,
)


def read_train_meta(src_base: Path) -> dict[str, list[str]]:
    """Kaynak food-101 klasorunden meta/train okur (test'e dokunmaz).

    meta/train.txt ONCELIKLI (gercek arsip satir formati: 'sinif/id', uzantisiz);
    yoksa meta/train.json. data.py ile AYNI okuma yolunu kullanir.
    """
    meta_dir = Path(src_base) / "meta"
    if not (meta_dir / "train.txt").exists() and not (meta_dir / "train.json").exists():
        raise FileNotFoundError(
            f"meta/train.txt veya meta/train.json bulunamadi: {meta_dir}\n"
            "Kaynak food-101 klasoru (meta/ + images/ iceren) gerekli."
        )
    return read_food101_split(meta_dir, "train")


def select_pool(
    train_meta: dict[str, list[str]],
    num_classes: int,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> dict[str, list[str]]:
    """Siniflari ve sinif basina gorselleri deterministik secer.

    Degerer (kaynak meta'daki gibi) sinif on eki icerir, uzantisiz:
    "apple_pie/1005649".
    """
    classes = sorted(train_meta.keys())
    if num_classes > len(classes):
        raise ValueError(f"Istenen {num_classes} sinif, kaynakta {len(classes)} var")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(classes, num_classes))

    need = train_per_class + val_per_class
    pool: dict[str, list[str]] = {}
    for cls in chosen:
        rels = list(train_meta[cls])
        if len(rels) < need:
            raise ValueError(
                f"{cls}: kaynakta {len(rels)} gorsel var, {need} gerekli"
            )
        rng2 = random.Random(f"{seed}:{cls}")
        pool[cls] = rng2.sample(rels, need)
    return pool


def build_subset(
    src_base: Path,
    out_base: Path,
    num_classes: int,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> dict:
    """Havuzu kopyalar ve val_indices.json ile %80/20 split'i kalici yapar."""
    src_base = Path(src_base)
    out_base = Path(out_base)
    train_meta = read_train_meta(src_base)
    pool = select_pool(train_meta, num_classes, train_per_class, val_per_class, seed)

    src_images = src_base / "images"
    out_images = out_base / "images"
    if out_images.exists():
        shutil.rmtree(out_images)
    out_images.mkdir(parents=True, exist_ok=True)

    copied = 0
    for cls, rels in pool.items():
        cls_dir = out_images / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for rel in rels:
            # rel = "<sinif>/<ad>" (uzantisiz); kaynak gorsel: images/<rel>.jpg
            src_img = src_images / f"{rel}.jpg"
            if not src_img.exists():
                # bazi varyantlarda uzanti zaten olabilir
                alt = src_images / rel
                if alt.exists():
                    src_img = alt
                else:
                    raise FileNotFoundError(f"Kaynak gorsel yok: {src_img}")
            dst = cls_dir / Path(rel).name
            if not dst.suffix:
                dst = dst.with_suffix(".jpg")
            shutil.copy2(src_img, dst)
            copied += 1

    # val_indices.json'foodlens.data ile AYNI kanonik sira uzerinden yaz
    paths, labels, classes = collect_pool(out_images)
    total_per_class = train_per_class + val_per_class
    val_ratio = val_per_class / total_per_class
    val_indices_path = out_base / "val_indices.json"
    train_idx, val_idx = load_or_create_val_indices(
        val_indices_path, labels, val_ratio, seed
    )

    summary = {
        "out": str(out_base),
        "num_classes": len(classes),
        "classes": classes,
        "copied_images": copied,
        "pool_per_class": total_per_class,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "val_ratio": val_ratio,
        "val_indices_path": str(val_indices_path),
        "seed": seed,
        "test_split_touched": False,
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Food-101'den smoke alt kume uretir")
    p.add_argument("--src", default="data/food-101", help="Kaynak food-101 klasoru (meta/ + images/)")
    p.add_argument("--out", default="data/subsets/smoke_10cls", help="Cikti alt kume klasoru")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--train-per-class", type=int, default=80)
    p.add_argument("--val-per-class", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    summary = build_subset(
        src_base=Path(args.src),
        out_base=Path(args.out),
        num_classes=args.num_classes,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
