"""FoodLens - veri katmani (Asama 1).

Food-101 (veya ondan uretilen alt kume) uzerinde:
- meta/train.json okuma (torchvision Food101 birebir formati: {sinif: [yol, ...]})
- stratified %N val split (seed ile); val indeksleri JSON'a kaydedilir (tekrar uretilebilir)
- train: augmentation'li; val/test: augmentation'siz AYRI dataset
- per-class dagilim raporu, ornek grid gorseli

Not: Indirme yapmaz. Tam Food-101 indirmesi Colab'da yapilir; yerelde alt kume/oyuncak veri kullanilir.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from .config import resolve_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transform'lar
# ---------------------------------------------------------------------------
def build_transforms(
    image_size: int, augment: dict[str, Any] | None, train: bool
) -> transforms.Compose:
    """train=True -> augmentation'li; train=False -> yalnizca Resize+CenterCrop (temiz)."""
    if train:
        augment = augment or {}
        cj = augment.get("color_jitter", {}) or {}
        ops: list = [
            transforms.RandomResizedCrop(
                image_size, scale=tuple(augment.get("rrc_scale", (0.8, 1.0)))
            ),
            transforms.RandomHorizontalFlip(p=float(augment.get("hflip_p", 0.5))),
            transforms.ColorJitter(
                brightness=float(cj.get("brightness", 0.0)),
                contrast=float(cj.get("contrast", 0.0)),
                saturation=float(cj.get("saturation", 0.0)),
                hue=float(cj.get("hue", 0.0)),
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        return transforms.Compose(ops)
    # degerlendirme: kucult + merkez kirp + normalize (augmentation YOK)
    resize_to = int(round(image_size / 0.875))  # 224 -> 256
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Meta okuma
# ---------------------------------------------------------------------------
def read_food101_meta(meta_dir: Path) -> dict[str, list[str]]:
    """meta/train.json (veya test.json) okur. Dusuk seviye; sadece dict dondurur."""
    path = Path(meta_dir)
    if not path.exists():
        raise FileNotFoundError(f"Food-101 meta bulunamadi: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"meta JSON dict degil: {path}")
    return data


def read_food101_split(meta_dir: Path, split: str) -> dict[str, list[str]]:
    """meta/<split>.txt ONCELIKLI; yoksa <split>.json. Ikcisi ayni dict'i dondurur.

    Gercek arsip satir formati (meta/train.txt, meta/test.txt):
        apple_pie/1005649        <- "sinif/id", UZANTISIZ, dosya satiri basina bir ornek
    meta/train.json ayni bilgiyi dict olarak tutar ve degerler de sinif on eki tasir:
        {"apple_pie": ["apple_pie/1005649", ...]}
    Burada .txt satirini "sinif/id" olarak okuyup JSON degeriyle AYNI bicime
    (sinif on ekli, uzantisiz) ceviririz; boylece iki kaynak birebir eslesir.
    """
    meta_dir = Path(meta_dir)
    txt_path = meta_dir / f"{split}.txt"
    if txt_path.exists():
        meta: dict[str, list[str]] = defaultdict(list)
        for lineno, line in enumerate(
            txt_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            cls, sep, rest = line.partition("/")
            if not sep or not cls or not rest:
                raise ValueError(
                    f"meta/{split}.txt {lineno}. satir gecersiz "
                    f"( beklenen 'sinif/id' ): {line!r}"
                )
            meta[cls].append(f"{cls}/{rest}")
        return dict(meta)
    return read_food101_meta(meta_dir / f"{split}.json")


def read_food101_classes(meta_dir: Path, split: str = "train") -> list[str]:
    """Sinif listesi: meta/classes.txt varsa ONU (gercekte 101 sirali ad); yoksa sorted."""
    meta_dir = Path(meta_dir)
    classes_txt = meta_dir / "classes.txt"
    if classes_txt.exists():
        return [
            ln.strip()
            for ln in classes_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    return sorted(read_food101_split(meta_dir, split).keys())


def samples_from_meta(
    meta: dict[str, list[str]],
    images_dir: Path,
    classes: list[str] | None = None,
) -> tuple[list[str], list[int], list[str]]:
    """Meta dict'ten (yol, hedef) ornegi listesi + sinif adlari uretir.

    Donus: (image_paths, labels, classes)
    - classes: verilirse O kullanilir (meta/classes.txt); verilmezse sorted(meta)
    - labels: class_to_idx eslemesi
    - image_paths: images_dir/<yol>.jpg   (yol zaten sinif on eki tasir, uzantisiz)

    KANONIK SIRA: siniflar sirali, her sinif icin ornekler de rel-path ile sirali.
    Bu yuzden ayni meta hem .txt hem .json'dan okunsa BIREBIR ayni ornek sirasi
    uretilir (val_indices.json indeksleri kaynak formatindan bagimsiz olur).
    """
    if classes is None:
        classes = sorted(meta.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    image_paths: list[str] = []
    labels: list[int] = []
    for cls in classes:
        idx = class_to_idx[cls]
        for rel in sorted(str(r).replace("\\", "/") for r in meta.get(cls, [])):
            image_paths.append(str(images_dir / Path(rel).with_suffix(".jpg")))
            labels.append(idx)
    return image_paths, labels, list(classes)


# ---------------------------------------------------------------------------
# Stratified val split + val_indices.json
# ---------------------------------------------------------------------------
def stratified_split(
    labels: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Sinif bazli dengeli train/val indeksleri uretir (deterministik).

    Her sinif icin ayri RNG (seed + sinif) kullanilir; boylece sinif ekleme/cikarma
    diger siniflarin bolunmesini DEGISTIRMEZ (tekrar uretilebilirlik icin onemli).
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio (0,1) araliginda olmali: {val_ratio}")
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[lab].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in sorted(by_class):
        idxs = sorted(by_class[cls])
        rng = random.Random(f"{seed}:{cls}")
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def load_or_create_val_indices(
    path: Path, labels: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """val indekslerini JSON'dan yukler; yoksa stratified split ile uretip kaydeder.

    JSON mevcut ve gecerliyse YENI split uretilmez (tekrar uretilebilirlik).
    Donus: (train_idx, val_idx)
    """
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        val_idx = [int(i) for i in payload["val_indices"]]
        # gecerlilik: indeksler menzil disinda VEYA ratio/seed farkliysa yeniden uret
        valid = (
            all(0 <= i < len(labels) for i in val_idx)
            and len(set(val_idx)) == len(val_idx)
            and abs(float(payload.get("val_ratio", -1)) - val_ratio) < 1e-9
            and int(payload.get("seed", -1)) == seed
            and int(payload.get("num_samples", -1)) == len(labels)
        )
        if valid:
            val_set = set(val_idx)
            train_idx = [i for i in range(len(labels)) if i not in val_set]
            return train_idx, val_idx

    train_idx, val_idx = stratified_split(labels, val_ratio, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "val_ratio": val_ratio,
                "seed": seed,
                "num_samples": len(labels),
                "val_indices": val_idx,
            },
            f,
            indent=2,
        )
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageListDataset(Dataset):
    """(image_path, label) listesinden basit dataset. Transform'suz ham PIL dondurur."""

    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = default_loader(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def build_dataset(
    image_paths: list[str], labels: list[int], train: bool, image_size: int, augment
) -> ImageListDataset:
    tf = build_transforms(image_size, augment, train=train)
    return ImageListDataset(image_paths, labels, transform=tf)


# ---------------------------------------------------------------------------
# DataLoader seti
# ---------------------------------------------------------------------------
def build_dataloaders(
    train_ds, val_ds, test_ds, batch_size: int, num_workers: int = 0, seed: int = 42
) -> dict[str, DataLoader]:
    """train (karistir), val/test (sabit sirada) loader'lari olusturur."""
    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loaders = {
        "train": DataLoader(
            train_ds, shuffle=True, generator=g, drop_last=False, **common
        ),
        "val": DataLoader(val_ds, shuffle=False, **common),
    }
    if test_ds is not None:
        loaders["test"] = DataLoader(test_ds, shuffle=False, **common)
    return loaders


# ---------------------------------------------------------------------------
# Tam Food-101 yukleyici (Colab) + alt kume yukleyici
# ---------------------------------------------------------------------------
def load_food101(
    root: Path,
    val_ratio: float,
    seed: int,
    val_indices_path: Path | None,
    image_size: int = 224,
    augment=None,
    download: bool = False,
) -> dict[str, Any]:
    """Tam Food-101 (data/food-101). train split'inden stratified val ayirir.

    download=True sadece Colab'da; yerelde indirme YAPMAZ (Runtime hatasi verir).
    """
    root = Path(root)
    base = root / "food-101"
    meta_dir = base / "meta"
    images_dir = base / "images"

    if download:
        from torchvision.datasets import Food101 as _F101

        _F101(root=str(root), split="train", download=True)

    # Gercek arsipte meta hem .txt hem .json icerir; .txt ONCELIKLI (satir: sinif/id).
    classes = read_food101_classes(meta_dir, "train")
    train_meta = read_food101_split(meta_dir, "train")
    test_meta = read_food101_split(meta_dir, "test")

    unknown = set(train_meta) - set(classes)
    if unknown:
        raise ValueError(
            f"train meta'da classes.txt'te OLMAYAN siniflar var: {sorted(unknown)[:5]}"
        )

    tr_paths, tr_labels, _ = samples_from_meta(train_meta, images_dir, classes)
    te_paths, te_labels, _ = samples_from_meta(test_meta, images_dir, classes)

    if val_indices_path is None:
        val_indices_path = root / "val_indices.json"

    bundle = _split_into_datasets(
        tr_paths, tr_labels, val_ratio, seed, val_indices_path, image_size, augment
    )
    bundle.update(
        {
            "classes": classes,
            "test": build_dataset(te_paths, te_labels, False, image_size, augment),
            "test_size": len(te_paths),
        }
    )
    return bundle


def collect_pool(images_dir: Path) -> tuple[list[str], list[int], list[str]]:
    """Alt kume havuzunu kanonik sirada okur: (image_paths, labels, classes).

    images/<sinif>/*.jpg duzeninde; siniflar sorted, icindekiler filename ile
    sirali -> DETERMINISTIK. make_subset.py ile data.py AYNIsini kullanir,
    boylece val_indices.json'daki indeksler her iki tarafta da tutarli olur.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Alt kume images klasoru yok: {images_dir}")
    classes = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    if not classes:
        raise RuntimeError(f"Alt kumede sinif klasoru bulunamadi: {images_dir}")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths: list[str] = []
    labels: list[int] = []
    for cls in classes:
        for p in sorted((images_dir / cls).glob("*.jpg")):
            paths.append(str(p))
            labels.append(class_to_idx[cls])
    if not paths:
        raise RuntimeError(f"Alt kumede gorsel bulunamadi: {images_dir}")
    return paths, labels, classes


def _split_into_datasets(
    paths: list[str],
    labels: list[int],
    val_ratio: float,
    seed: int,
    val_indices_path: Path,
    image_size: int,
    augment,
) -> dict[str, Any]:
    """Tek split mekanizmasi: val indekslerini JSON'dan yukle/uret, train+val dataset kur.

    train: augmentation'li, val: augmentation'siz (AYRI dataset).
    """
    train_idx, val_idx = load_or_create_val_indices(
        Path(val_indices_path), labels, val_ratio, seed
    )
    s_train = [paths[i] for i in train_idx]
    l_train = [labels[i] for i in train_idx]
    s_val = [paths[i] for i in val_idx]
    l_val = [labels[i] for i in val_idx]
    return {
        "train": build_dataset(s_train, l_train, True, image_size, augment),
        "val": build_dataset(s_val, l_val, False, image_size, augment),
        "train_size": len(s_train),
        "val_size": len(s_val),
        "val_indices_path": str(val_indices_path),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def load_subset(
    root: Path,
    val_ratio: float,
    seed: int,
    val_indices_path: Path | None,
    image_size: int = 224,
    augment=None,
) -> dict[str, Any]:
    """Alt kume (data/subsets/<ad>). Havuz images/<sinif>/*.jpg.

    Yerel smoke test icin; val_indices.json ile %80/20 split (make_subset.py uretir).
    Test split'i YOK (make_subset test'e dokunmaz; test degerlendirmesi Colab'da
    tam Food-101 test split'i ile yapilir).
    """
    root = Path(root)
    paths, labels, classes = collect_pool(root / "images")
    if val_indices_path is None:
        val_indices_path = root / "val_indices.json"

    bundle = _split_into_datasets(
        paths, labels, val_ratio, seed, val_indices_path, image_size, augment
    )
    bundle["classes"] = classes
    bundle["test"] = None
    bundle["test_size"] = 0
    return bundle


def build_from_config(config: dict[str, Any], download: bool = False) -> dict[str, Any]:
    """Config'ten dogru yukleyiciyi secip dataset bundle dondurur.

    config["data"]["kind"]: "food101" (tam) | "subset" (alt kume)
    """
    d = config["data"]
    kind = d.get("kind", "food101")
    root = resolve_path(d["root"])
    augment = d.get("augment")
    image_size = int(d.get("image_size", 224))

    if kind == "food101":
        val_indices = d.get("val_indices_path")
        return load_food101(
            root=root,
            val_ratio=float(d.get("val_ratio", 0.1)),
            seed=int(config.get("seed", 42)),
            val_indices_path=resolve_path(val_indices) if val_indices else None,
            image_size=image_size,
            augment=augment,
            download=download,
        )
    if kind == "subset":
        val_indices = d.get("val_indices_path")
        return load_subset(
            root=root,
            val_ratio=float(d.get("val_ratio", 0.2)),
            seed=int(config.get("seed", 42)),
            val_indices_path=resolve_path(val_indices) if val_indices else None,
            image_size=image_size,
            augment=augment,
        )
    raise ValueError(f"Bilinmeyen data.kind: {kind!r}")


# ---------------------------------------------------------------------------
# Raporlar
# ---------------------------------------------------------------------------
def per_class_distribution(bundle: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Her split icin sinif basina ornek sayisi. {split: {sinif: adet}}"""
    splits = ["train", "val", "test"]
    classes = bundle["classes"]
    out: dict[str, dict[str, int]] = {}
    for sp in splits:
        ds = bundle.get(sp)
        if ds is None:
            continue
        counts = {c: 0 for c in classes}
        for lab in ds.labels:
            counts[classes[lab]] += 1
        out[sp] = counts
    return out


def format_distribution_report(bundle: dict[str, Any]) -> str:
    """Per-class dagilim raporu (metin)."""
    dist = per_class_distribution(bundle)
    classes = bundle["classes"]
    lines = ["Per-class ornek sayilari (min/max + ornek):"]
    for sp, counts in dist.items():
        vals = list(counts.values())
        if not vals:
            continue
        lines.append(
            f"  {sp:5s}: n={sum(vals):6d}  sinif={len(vals):3d}  "
            f"min={min(vals)}  max={max(vals)}  "
            f"{'dengeli' if min(vals) == max(vals) else 'DENGESIZ'}"
        )
    # ilk 3 sinifin detayi (test dengesini gosterir)
    for sp, counts in dist.items():
        detail = ", ".join(f"{c}={counts[c]}" for c in classes[:3])
        lines.append(f"  {sp:5s} ilk 3: {detail}")
    return "\n".join(lines)


def save_sample_grid(
    ds,
    classes: list[str],
    out_path: Path,
    n: int = 16,
    ncol: int = 4,
    title: str = "ornekler",
) -> Path:
    """Dataset'ten rastgele n ornegi grid olarak kaydeder (normalize geri alinmis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(n, len(ds))
    idxs = random.Random(0).sample(range(len(ds)), n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.4 * nrow))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idxs):
        img, lab = ds[i]
        if isinstance(img, torch.Tensor):
            img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(str(classes[lab]), fontsize=7)
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
