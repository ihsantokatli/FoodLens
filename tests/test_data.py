"""Asama 1 testleri: veri katmani (split, tekrar uretilebilirlik, transform, subset)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from foodlens.data import (
    build_from_config,
    collect_pool,
    format_distribution_report,
    load_or_create_val_indices,
    per_class_distribution,
    stratified_split,
)


# ---------------------------------------------------------------------------
# stratified split + tekrar uretilebilirlik
# ---------------------------------------------------------------------------
def test_stratified_split_disjoint_and_ratio():
    labels = [c for c in range(10) for _ in range(50)]  # 10 sinif x 50
    tr, va = stratified_split(labels, val_ratio=0.1, seed=42)
    assert set(tr).isdisjoint(va)
    assert sorted(tr + va) == list(range(len(labels)))
    # her sinif icin ~%10 (50 * 0.1 = 5)
    from collections import Counter
    val_per_class = Counter(labels[i] for i in va)
    assert all(v == 5 for v in val_per_class.values())


def test_stratified_split_deterministic_same_seed():
    labels = [c for c in range(10) for _ in range(50)]
    a = stratified_split(labels, 0.1, seed=42)
    b = stratified_split(labels, 0.1, seed=42)
    assert a == b


def test_val_indices_reproducible_and_persisted(tmp_path: Path):
    labels = [c for c in range(7) for _ in range(20)]
    path = tmp_path / "val_indices.json"

    tr1, va1 = load_or_create_val_indices(path, labels, 0.1, seed=42)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 42 and abs(payload["val_ratio"] - 0.1) < 1e-9
    assert payload["num_samples"] == len(labels)

    # ikinci cagri: JSON'dan AYNI indeksleri yukler (yeniden uretmez)
    tr2, va2 = load_or_create_val_indices(path, labels, 0.1, seed=42)
    assert va1 == va2 and tr1 == tr2

    # farkli seed farkli split uretmeli (JSON guncellenir)
    tr3, va3 = load_or_create_val_indices(path, labels, 0.1, seed=7)
    assert va3 != va1


def test_val_indices_json_matches_split(tmp_path: Path):
    labels = [c for c in range(4) for _ in range(25)]
    path = tmp_path / "v.json"
    _, va = load_or_create_val_indices(path, labels, 0.2, seed=42)
    stored = json.loads(path.read_text(encoding="utf-8"))["val_indices"]
    assert stored == va


# ---------------------------------------------------------------------------
# tam Food-101 akisi (toy kaynak): train/val/test KESISMEMELI
# ---------------------------------------------------------------------------
def _food101_cfg(toy_food101: Path, tmp_path: Path, augment: dict) -> dict:
    return {
        "seed": 42,
        "data": {
            "kind": "food101",
            "root": str(toy_food101.parent),  # .../toy -> toy/food-101/
            "num_classes": 5,
            "val_ratio": 0.1,
            "val_indices_path": str(tmp_path / "val_indices.json"),
            "image_size": 64,
            "batch_size": 8,
            "num_workers": 0,
            "augment": augment,
        },
    }


def test_food101_no_overlap(toy_food101, tmp_path, augment_cfg):
    cfg = _food101_cfg(toy_food101, tmp_path, augment_cfg)
    b = build_from_config(cfg)

    tr = set(b["train"].image_paths)
    va = set(b["val"].image_paths)
    te = set(b["test"].image_paths)

    assert tr.isdisjoint(va), "train/val kesişmemeli"
    assert tr.isdisjoint(te), "train/test kesişmemeli"
    assert va.isdisjoint(te), "val/test kesişmemeli"
    # butunluk: train+val == tam train split, test ayri split
    assert len(tr) + len(va) == b["train_size"] + b["val_size"]
    assert b["test_size"] > 0


def test_food101_augmented_vs_clean_transform(toy_food101, tmp_path, augment_cfg):
    cfg = _food101_cfg(toy_food101, tmp_path, augment_cfg)
    b = build_from_config(cfg)

    train_ops = [type(t).__name__ for t in b["train"].transform.transforms]
    val_ops = [type(t).__name__ for t in b["val"].transform.transforms]
    test_ops = [type(t).__name__ for t in b["test"].transform.transforms]

    # train: augmentation var
    assert "RandomResizedCrop" in train_ops
    assert "RandomHorizontalFlip" in train_ops
    # val/test: augmentation YOK (yalnizca Resize + CenterCrop + ToTensor + Normalize)
    for ops in (val_ops, test_ops):
        assert "RandomResizedCrop" not in ops
        assert "RandomHorizontalFlip" not in ops
        assert "ColorJitter" not in ops
        assert "Resize" in ops and "CenterCrop" in ops


def test_val_deterministic_train_not(toy_food101, tmp_path, augment_cfg):
    cfg = _food101_cfg(toy_food101, tmp_path, augment_cfg)
    b = build_from_config(cfg)

    # val: ayni indeks iki kez -> AYNI tensor (augmentation yok)
    x1, _ = b["val"][0]
    x2, _ = b["val"][0]
    assert torch.equal(x1, x2)

    # train: augmentation nedeniyle (genelde) farkli olmali
    diffs = 0
    for _ in range(5):
        xa, _ = b["train"][0]
        xb, _ = b["train"][0]
        if not torch.equal(xa, xb):
            diffs += 1
    assert diffs > 0, "train transform'i rastgele olmali"


def test_image_shape(toy_food101, tmp_path, augment_cfg):
    cfg = _food101_cfg(toy_food101, tmp_path, augment_cfg)
    cfg["data"]["image_size"] = 64
    b = build_from_config(cfg)
    x, y = b["train"][0]
    assert x.shape == (3, 64, 64)
    assert x.dtype == torch.float32


# ---------------------------------------------------------------------------
# alt kume (subset) + make_subset test'e DOKUNMAZ
# ---------------------------------------------------------------------------
def test_make_subset_no_test_touch_and_split(toy_food101, tmp_path, augment_cfg):
    from make_subset import build_subset

    out = tmp_path / "subset"
    # test.json'i once hash'le
    test_json = toy_food101 / "meta" / "test.json"
    before = test_json.read_bytes()

    summary = build_subset(
        src_base=toy_food101,
        out_base=out,
        num_classes=3,
        train_per_class=20,
        val_per_class=5,
        seed=42,
    )

    assert test_json.read_bytes() == before, "test split'e dokunulmamali"
    assert summary["test_split_touched"] is False
    # cikti klasorunde test goruntusu YOK (sadece havuz)
    assert summary["num_classes"] == 3
    assert summary["copied_images"] == 3 * 25
    assert summary["train_size"] == 3 * 20
    assert summary["val_size"] == 3 * 5
    assert (out / "val_indices.json").exists()

    # test images/ klasoru alt kumede olmamali (havuz sadece gorsel)
    # (test split'i kopyalanmadi)
    all_names = [p.name for p in (out / "images").rglob("*.jpg")]
    assert not any("_test_" in n for n in all_names), "test gorselleri kopyalanmamali"


def test_subset_no_overlap_and_reproducible(toy_food101, tmp_path, augment_cfg):
    from make_subset import build_subset

    out = tmp_path / "subset2"
    build_subset(toy_food101, out, 3, 20, 5, seed=42)

    cfg = {
        "seed": 42,
        "data": {
            "kind": "subset",
            "root": str(out),
            "num_classes": 3,
            "val_ratio": 0.20,  # 5/25
            "val_indices_path": str(out / "val_indices.json"),
            "image_size": 64,
            "batch_size": 8,
            "num_workers": 0,
            "augment": augment_cfg,
        },
    }
    b = build_from_config(cfg)
    tr = set(b["train"].image_paths)
    va = set(b["val"].image_paths)
    assert tr.isdisjoint(va)
    assert b["train_size"] == 60  # 3*20
    assert b["val_size"] == 15  # 3*5
    assert b["test"] is None

    # tekrar cagir -> AYNI split (val_indices.json yeniden kullanilir)
    b2 = build_from_config(cfg)
    assert b2["train"].image_paths == b["train"].image_paths
    assert b2["val"].image_paths == b["val"].image_paths


def test_collect_pool_canonical_order(toy_food101, tmp_path, augment_cfg):
    from make_subset import build_subset

    out = tmp_path / "subset3"
    build_subset(toy_food101, out, 4, 10, 5, seed=42)
    p1, l1, c1 = collect_pool(out / "images")
    p2, l2, c2 = collect_pool(out / "images")
    assert p1 == p2 and l1 == l2 and c1 == c2  # deterministik
    assert c1 == sorted(c1)


# ---------------------------------------------------------------------------
# rapor
# ---------------------------------------------------------------------------
def test_distribution_report_runs(toy_food101, tmp_path, augment_cfg):
    cfg = _food101_cfg(toy_food101, tmp_path, augment_cfg)
    b = build_from_config(cfg)
    dist = per_class_distribution(b)
    assert set(dist) == {"train", "val", "test"}
    for split_counts in dist.values():
        assert sum(split_counts.values()) > 0
    report = format_distribution_report(b)
    assert "Per-class" in report
