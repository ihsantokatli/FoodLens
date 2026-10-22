"""Asama 1 testleri: veri katmani (split, tekrar uretilebilirlik, transform, subset).

Ayrica Asama 2 oncesi iki dogrulama:
  1) toy uretici GERCEK Food-101 meta bicimiyle birebir mi?  (.txt/.json esitligi)
  2) default.yaml (kind: food101, val_ratio 0.10) yolu sinif basina %10 val veriyor mu?
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from foodlens.config import load_config
from foodlens.data import (
    build_from_config,
    collect_pool,
    format_distribution_report,
    load_or_create_val_indices,
    per_class_distribution,
    read_food101_classes,
    read_food101_split,
    samples_from_meta,
    stratified_split,
)
from foodlens.validate import (
    FOOD101_EXPECTED,
    Food101ValidationError,
    inspect_food101,
    validate_food101,
)

ROOT = Path(__file__).resolve().parents[1]


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


# ===========================================================================
# DOGRULAMA 1) toy uretici gercek Food-101 meta bicimiyle BIREBIR mi?
# ===========================================================================
def test_toy_meta_matches_real_food101_format(toy_food101: Path):
    """Toy arsiv gercekteki dosya/satir bicimini birebir uretmeli.

    Gercek Food-101 (alpapado/food-101 kopyasiyla dogrulandi):
      meta/classes.txt : sirali sinif adlari, satir satir
      meta/train.txt   : satir basina "sinif/id" (UZANTISIZ)
      meta/test.txt    : ayni
      meta/train.json  : {sinif: ["sinif/id", ...]}
      images/<sinif>/<id>.jpg
    """
    meta = toy_food101 / "meta"
    images = toy_food101 / "images"

    # 1) beklenen dosyalar mevcut
    for name in ("classes.txt", "labels.txt", "train.txt", "test.txt", "train.json", "test.json"):
        assert (meta / name).exists(), f"eksik meta dosyasi: {name}"

    # 2) classes.txt sirali, satir sayisi == json anahtar sayisi
    classes_txt = [ln for ln in (meta / "classes.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert classes_txt == sorted(classes_txt)
    train_json = json.loads((meta / "train.json").read_text(encoding="utf-8"))
    assert classes_txt == sorted(train_json.keys())

    # 3) train.txt / test.txt satir formati: "sinif/id", UZANTISIZ, bos satir yok
    for split in ("train", "test"):
        lines = (meta / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in lines if ln.strip()]
        j = json.loads((meta / f"{split}.json").read_text(encoding="utf-8"))
        assert len(lines) == sum(len(v) for v in j.values()), f"{split}.txt satir sayisi != json toplami"
        for ln in lines:
            cls, sep, rest = ln.partition("/")
            assert sep == "/", f"{split}.txt satirinda '/' yok: {ln!r}"
            assert cls in j, f"{split}.txt satirinda bilinmeyen sinif: {cls!r}"
            assert not rest.lower().endswith((".jpg", ".jpeg", ".png")), f"{split}.txt satiri uzantili: {ln!r}"
            assert ln in j[cls], f"{split}.txt satiri json degerlerinde yok: {ln!r}"

    # 4) her meta girdisi icin gercek gorsel dosyasi var: images/<sinif>/<id>.jpg
    for split in ("train", "test"):
        j = json.loads((meta / f"{split}.json").read_text(encoding="utf-8"))
        for cls, rels in j.items():
            for rel in rels:
                assert (images / f"{rel}.jpg").exists(), f"gorsel yok: images/{rel}.jpg"


def test_txt_and_json_meta_produce_identical_samples(tmp_path: Path):
    """data.py ayni split'i .txt ve .json'dan okuyup BIREBIR ayni ornegi uretmeli.

    Kanonik sira (siniflar sirali + her sinif icin rel-path sirali) sayesinde
    val_indices.json indeksleri meta formatindan bagimsiz olur.
    """
    from make_toy_food101 import make_toy_food101

    base = make_toy_food101(
        out_dir=tmp_path / "eq", num_classes=4, train_per_class=9, test_per_class=4, image_size=32, seed=7
    )
    meta_dir, images_dir = base / "meta", base / "images"
    classes = read_food101_classes(meta_dir, "train")
    assert (meta_dir / "classes.txt").exists()

    for split in ("train", "test"):
        # .json'u gecici olarak kaldir -> .txt'den oku
        jp = meta_dir / f"{split}.json"
        backup = meta_dir / f"{split}.json.bak"
        jp.rename(backup)
        try:
            meta_txt = read_food101_split(meta_dir, split)
            p_txt, l_txt, c_txt = samples_from_meta(meta_txt, images_dir, classes)
        finally:
            backup.rename(jp)

        # simdi .json'dan oku
        meta_json = read_food101_split(meta_dir, split)
        p_json, l_json, c_json = samples_from_meta(meta_json, images_dir, classes)

        assert p_txt == p_json, f"{split}: .txt ve .json farkli GORSEL SIRASI verdi"
        assert l_txt == l_json, f"{split}: .txt ve .json farkli ETIKET verdi"
        assert c_txt == c_json == classes
        assert len(p_txt) > 0


def test_data_py_prefers_txt_and_is_equivalent(tmp_path: Path, augment_cfg):
    """build_from_config(kind=food101) .txt varken de .json'a dusunce AYNI sonucu vermeli."""
    from make_toy_food101 import make_toy_food101

    base = make_toy_food101(
        out_dir=tmp_path / "eq2", num_classes=3, train_per_class=10, test_per_class=4, image_size=32, seed=11
    )

    def _cfg(vi_name: str) -> dict:
        return {
            "seed": 42,
            "data": {
                "kind": "food101",
                "root": str(base.parent),
                "val_ratio": 0.2,
                "val_indices_path": str(tmp_path / vi_name),
                "image_size": 32,
                "batch_size": 4,
                "num_workers": 0,
                "augment": augment_cfg,
            },
        }

    # .txt + .json birlikte -> .txt oncelikli
    b_both = build_from_config(_cfg("vi_both.json"))

    # .txt'leri gecici kaldir -> .json fallback
    meta = base / "meta"
    moved = []
    for split in ("train", "test"):
        src = meta / f"{split}.txt"
        dst = meta / f"{split}.txt.bak"
        src.rename(dst)
        moved.append((dst, src))
    try:
        b_json = build_from_config(_cfg("vi_json.json"))
    finally:
        for dst, src in moved:
            dst.rename(src)

    assert b_both["train"].image_paths == b_json["train"].image_paths
    assert b_both["val"].image_paths == b_json["val"].image_paths
    assert b_both["test"].image_paths == b_json["test"].image_paths
    assert b_both["train"].labels == b_json["train"].labels
    assert b_both["classes"] == b_json["classes"]


# ===========================================================================
# DOGRULAMA 2) default.yaml (kind: food101, val_ratio 0.10) -> sinif basina %10 val
# ===========================================================================
def test_default_yaml_val_ratio_per_class(toy_food101: Path, tmp_path: Path, augment_cfg):
    """default.yaml yolu toy veride SINIF BASINA tam %10 val uretmeli."""
    cfg = load_config(ROOT / "configs" / "default.yaml")

    # default.yaml'in gercek ayarlari korunur; sadece kok ve indeks yolu toy'a cevrilir
    assert cfg["data"]["kind"] == "food101"
    val_ratio = float(cfg["data"]["val_ratio"])
    assert val_ratio == 0.10, "default.yaml val_ratio 0.10 olmali"
    seed = int(cfg["seed"])
    assert seed == 42

    cfg["data"]["root"] = str(toy_food101.parent)
    cfg["data"]["val_indices_path"] = str(tmp_path / "val_indices.json")
    cfg["data"]["num_workers"] = 0
    cfg["data"]["augment"] = augment_cfg

    b = build_from_config(cfg)
    dist = per_class_distribution(b)

    # toy train: her sinif 30 ornek -> val = round(30*0.10) = 3, kalan train = 27
    meta_train = read_food101_split(toy_food101 / "meta", "train")
    per_class_train_pool = {cls: len(v) for cls, v in meta_train.items()}
    assert set(per_class_train_pool.values()) == {30}

    for cls, pool_n in per_class_train_pool.items():
        expected_val = max(1, round(pool_n * val_ratio))  # stratified_split ile ayni formul
        assert dist["val"][cls] == expected_val, f"{cls}: val {dist['val'][cls]} != {expected_val}"
        assert dist["train"][cls] == pool_n - expected_val
        # gercekten %10
        assert dist["val"][cls] / pool_n == pytest.approx(val_ratio)

    assert sum(dist["val"].values()) == 15  # 5 sinif * 3
    assert sum(dist["train"].values()) == 135  # 5 sinif * 27
    assert sum(dist["test"].values()) == 50  # 5 sinif * 10


# ===========================================================================
# DOGRULAMA 3) Colab on-dogrulama (validate.py) - toy ile mekanizma testi
# ===========================================================================
def test_validate_food101_passes_on_toy_with_toy_expectations(toy_food101: Path):
    """Dogrulayici dogru beklentilerle toy uzerinden GECER (mekanizma calisiyor)."""
    toy_expected = {
        "classes": 5,
        "train_total": 150,  # 5 * 30
        "test_total": 50,  # 5 * 10
        "val": 15,  # 5 * round(30*0.1)
        "train_remaining": 135,  # 5 * 27
    }
    info = validate_food101(toy_food101.parent, val_ratio=0.10, seed=42, expected=toy_expected)
    assert info["num_classes"] == 5
    assert info["val_total"] == 15
    assert info["train_remaining"] == 135

    # per-class dengeli
    assert set(info["per_class_val"].values()) == {3}
    assert set(info["per_class_test"].values()) == {10}


def test_validate_food101_fails_when_counts_mismatch(toy_food101: Path):
    """Beklenen tutmazsa Food101ValidationError -> egitim baslatilmaz."""
    # toy veride GERCEK (101/75750/...) beklentileriyle basarisiz olmali
    with pytest.raises(Food101ValidationError) as exc:
        validate_food101(toy_food101.parent, val_ratio=0.10, seed=42, expected=FOOD101_EXPECTED)
    msg = str(exc.value)
    assert "egitim baslatilmadi" in msg
    assert "sinif sayisi" in msg  # 101 beklentisi tutmadi

    # yanlis val_ratio ile de basarisiz olmali (val sayisi kayar)
    with pytest.raises(Food101ValidationError):
        validate_food101(
            toy_food101.parent,
            val_ratio=0.5,
            seed=42,
            expected={"classes": 5, "train_total": 150, "test_total": 50, "val": 15, "train_remaining": 135},
        )


def test_validate_food101_inspect_reads_txt_meta(toy_food101: Path):
    """inspect_food101 gercek arsip bicimindeki (txt oncelikli) meta'dan sayiyor."""
    info = inspect_food101(toy_food101.parent, val_ratio=0.10, seed=42)
    assert info["train_total"] == 150
    assert info["test_total"] == 50
    assert (toy_food101 / "meta" / "train.txt").exists()
    assert (toy_food101 / "meta" / "classes.txt").exists()
