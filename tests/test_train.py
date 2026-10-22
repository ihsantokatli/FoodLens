"""Asama 3 testleri: train.py (run klasoru, checkpoint contract, epoch metrikleri, e2e)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from foodlens.train import (
    build_argparser,
    make_run_dir,
    run_epoch,
    save_checkpoint,
    train_run,
)

# ---------------------------------------------------------------------------
# run klasoru: ASLA ezme
# ---------------------------------------------------------------------------
def _run_cfg(tmp_path: Path) -> dict:
    return {
        "experiment_name": "unit_exp",
        "seed": 42,
        "logging": {
            "runs_dir": str(tmp_path / "runs"),
            "history_csv": "history.csv",
            "curves_png": "curves.png",
            "summary_csv": "summary.csv",
        },
    }


def test_make_run_dir_never_overwrites(tmp_path):
    cfg = _run_cfg(tmp_path)
    d1 = make_run_dir(cfg)
    (d1 / "best.pt").write_bytes(b"eski")  # dolu klasor isareti
    d2 = make_run_dir(cfg)
    assert d1.exists() and d2.exists()
    assert d1 != d2
    assert (d1 / "best.pt").read_bytes() == b"eski"  # ezkalmadi
    assert d1.is_dir() and d2.is_dir()


# ---------------------------------------------------------------------------
# checkpoint contract (api.py ile ayni)
# ---------------------------------------------------------------------------
def test_save_checkpoint_contract(tmp_path):
    from foodlens.model import build_model

    m = build_model(num_classes=7, weights=None, dropout=0.2)
    classes = [f"c{i}" for i in range(7)]
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, classes, "efficientnet_b0", 64, {"k": 1})

    blob = torch.load(p, map_location="cpu", weights_only=False)
    assert set(blob) == {
        "model_state",
        "classes",
        "arch",
        "num_classes",
        "image_size",
        "config",
    }
    assert blob["num_classes"] == 7
    assert blob["classes"] == classes
    assert blob["arch"] == "efficientnet_b0"
    assert blob["image_size"] == 64
    # state_dict geri yuklenebilir
    from foodlens.model import build_model as bm

    m2 = bm(num_classes=7, weights=None, dropout=0.2)
    m2.load_state_dict(blob["model_state"])


# ---------------------------------------------------------------------------
# run_epoch: metrik araliklari + top5 >= top1 + optimizer adimi
# ---------------------------------------------------------------------------
class _Tiny(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, n)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_run_epoch_metrics_and_step():
    torch.manual_seed(0)
    ds = TensorDataset(torch.randn(8, 3, 8, 8), torch.randint(0, 10, (8,)))
    loader = DataLoader(ds, batch_size=4)
    model = _Tiny()
    before = model.fc.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()

    m = run_epoch(model, loader, crit, torch.device("cpu"), opt, None, grad_clip=1.0)
    assert 0.0 <= m["top1"] <= 1.0
    assert 0.0 <= m["top5"] <= 1.0
    assert m["top5"] >= m["top1"]  # top5 asla top1'den kucuk olamaz
    assert m["loss"] > 0.0
    assert not torch.allclose(before, model.fc.weight.detach())  # optimizer adim atti


def test_run_epoch_eval_mode_no_step():
    torch.manual_seed(0)
    ds = TensorDataset(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,)))
    loader = DataLoader(ds, batch_size=4)
    model = _Tiny()
    before = model.fc.weight.detach().clone()
    m = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), None, None)
    assert torch.allclose(before, model.fc.weight.detach())  # egitim yok -> agirlik sabit
    assert 0.0 <= m["top1"] <= 1.0


# ---------------------------------------------------------------------------
# e2e: train_run toy veride iki asamayi da calistirir + tum ciktilari uretir
# ---------------------------------------------------------------------------
def test_train_run_end_to_end(toy_food101, tmp_path, augment_cfg):
    cfg = {
        "experiment_name": "unit_e2e",
        "seed": 42,
        "data": {
            "kind": "food101",
            "root": str(toy_food101.parent),
            "val_ratio": 0.2,
            "val_indices_path": str(tmp_path / "vi.json"),
            "image_size": 64,
            "batch_size": 4,
            "num_workers": 0,
            "augment": augment_cfg,
        },
        "model": {
            "arch": "efficientnet_b0",
            "pretrained_weights": None,  # indirmeyi atla (hizli, cevrimdisi)
            "dropout": 0.2,
        },
        "train": {
            "amp": False,
            "label_smoothing": 0.1,
            "grad_clip_norm": 1.0,
            "stage1": {"epochs": 1, "head_lr": 1e-3, "backbone_lr": 0.0},
            "stage2": {"epochs": 1, "head_lr": 1e-3, "backbone_lr": 1e-4, "scheduler": "cosine"},
            "unfreeze_from_block": -1,
        },
        "eval": {"topk": [1, 5]},
        "logging": {
            "runs_dir": str(tmp_path / "runs"),
            "history_csv": "history.csv",
            "curves_png": "curves.png",
            "summary_csv": "summary.csv",
        },
    }
    args = build_argparser().parse_args(["--device", "cpu", "--limit-batches", "2"])
    out = train_run(cfg, args)

    run_dir = Path(out["run_dir"])
    # tum beklenen ciktilar
    for name in ("config.json", "history.csv", "curves.png", "best.pt", "last.pt"):
        assert (run_dir / name).exists(), f"eksik cikti: {name}"

    # history.csv stage sutunu + iki asama
    hist = (run_dir / "history.csv").read_text(encoding="utf-8")
    lines = [ln for ln in hist.splitlines() if ln.strip()]
    assert lines[0].startswith("stage,")
    stages = [ln.split(",")[0] for ln in lines[1:]]
    assert stages == ["stage1", "stage2"]

    # runs/summary.csv
    summary = (tmp_path / "runs" / "summary.csv").read_text(encoding="utf-8")
    assert summary.startswith("timestamp,")
    assert "unit_e2e" in summary

    # ozet degerleri tutarli
    s = out["summary"]
    assert s["num_classes"] == 5  # toy fixture 5 sinif
    assert 0.0 <= float(s["best_val_top1"]) <= 1.0
    # toy'da test split var -> test degerlendirmesi yapildi
    assert s["test_top1"] != ""

    # checkpoint gercekten yuklenebilir + contract saglam
    blob = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert blob["num_classes"] == 5 and len(blob["classes"]) == 5
