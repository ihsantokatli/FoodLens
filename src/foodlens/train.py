"""FoodLens - egitim dongusu (Asama 3).

İki asamali ince ayar (fine-tuning) + runs/ logging + checkpoint contract.

  ASAMA 1 : freeze_backbone -> sadece head (LR head_lr, sabit)
  ASAMA 2 : unfreeze_from_block(model, n) -> discriminative LR
            (backbone backbone_lr, head head_lr) + cosine scheduler

Metrikler: loss + top1 + top5 (train/val her epoch; test sadece sonunda).
Ciktilar runs/<tarih>_<deney_ad>/ altina yazilir (mevcut klasor ASLA ezilmez):
  config.json, history.csv (stage sutunu), curves.png, best.pt, last.pt
  + runs/summary.csv (kosu basina ozet satiri)

Checkpoint contract (api.py ile ayni):
  {"model_state", "classes", "arch", "num_classes", "image_size", "config"}

Kullanim (ince sarmalayici):  python scripts/train.py --config configs/smoke.yaml
Modul olarak:                 python -m foodlens.train --config ...
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import load_config, resolve_path
from .data import build_dataloaders, build_from_config
from .model import (
    build_model,
    discriminative_param_groups,
    freeze_backbone,
    trainable_frozen_summary,
    unfreeze_from_block,
)

_HISTORY_FIELDS = [
    "stage",
    "epoch",
    "lr",
    "train_loss",
    "train_top1",
    "train_top5",
    "val_loss",
    "val_top1",
    "val_top5",
    "seconds",
]


# ---------------------------------------------------------------------------
# Run klasoru + checkpoint
# ---------------------------------------------------------------------------
def make_run_dir(cfg: dict[str, Any]) -> Path:
    """runs/<tarih>_<deney_ad> olusturur; mevcut ise _1, _2... ekler (ASLA ezmez)."""
    runs_root = resolve_path(cfg["logging"]["runs_dir"])
    base = f"{datetime.now():%Y%m%d_%H%M%S}_{cfg['experiment_name']}"
    run_dir = runs_root / base
    i = 1
    while run_dir.exists():
        run_dir = runs_root / f"{base}_{i}"
        i += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    path: Path,
    model: nn.Module,
    classes: list[str],
    arch: str,
    image_size: int,
    config: dict[str, Any],
) -> None:
    """Checkpoint contract ile kaydeder (api.py/train ayni sozlesme)."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": list(classes),
            "arch": arch,
            "num_classes": len(classes),
            "image_size": int(image_size),
            "config": config,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler=None,
    grad_clip: float = 0.0,
    limit_batches: int = 0,
) -> dict[str, float]:
    """Tek epoch: egitim (optimizer varsa) veya degerlendirme. Metrikleri dondurur."""
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None

    tot_loss = 0.0
    c1 = 0
    c5 = 0
    n = 0

    for bi, (xb, yb) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        bs = xb.size(0)
        tot_loss += loss.item() * bs
        k = min(5, logits.size(1))
        topk = logits.topk(k, dim=1).indices  # (bs, k) -> SINIF indeksleri
        c1 += (topk[:, 0] == yb).sum().item()
        # gercek etiket top-k icinde mi? (topk sutunlari SINIF indeksi, rank degil)
        c5 += topk.eq(yb.view(-1, 1)).any(dim=1).sum().item()
        n += bs

    n = max(n, 1)
    return {
        "loss": tot_loss / n,
        "top1": c1 / n,
        "top5": c5 / n,
    }


# ---------------------------------------------------------------------------
# Gorsel: egriler
# ---------------------------------------------------------------------------
def plot_curves(history: list[dict[str, Any]], out_path: Path, title: str) -> Path:
    """history satirlarindan loss + top1 egrileri cizer (Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = list(range(1, len(history) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(xs, [r["train_loss"] for r in history], marker="o", label="train")
    axes[0].plot(xs, [r["val_loss"] for r in history], marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("adim (epoch)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, [r["train_top1"] for r in history], marker="o", label="train top1")
    axes[1].plot(xs, [r["val_top1"] for r in history], marker="o", label="val top1")
    axes[1].set_title("Top-1 dogruluk")
    axes[1].set_xlabel("adim (epoch)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # asama gecisini goster
    for i in range(1, len(history)):
        if history[i]["stage"] != history[i - 1]["stage"]:
            for ax in axes:
                ax.axvline(i + 0.5, color="red", linestyle="--", alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CSV yardimcilari
# ---------------------------------------------------------------------------
def append_history(path: Path, row: dict[str, Any]) -> None:
    """history.csv'ye satir ekler (ilk seferde baslik yazar)."""
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _HISTORY_FIELDS})


def append_summary(runs_root: Path, row: dict[str, Any]) -> None:
    """runs/summary.csv'ye kosu ozeti satiri ekler (ilk seferde baslik)."""
    path = runs_root / "summary.csv"
    runs_root.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Asamalar
# ---------------------------------------------------------------------------
def _train_stage(
    model,
    loaders,
    criterion,
    device,
    scaler,
    cfg,
    stage: str,
    epochs: int,
    optimizer,
    scheduler,
    history_path: Path,
    history: list[dict[str, Any]],
    run_dir: Path,
    classes,
    arch,
    image_size,
    snapshot,
    grad_clip,
    limit_batches,
    best: dict[str, float],
) -> dict[str, float]:
    """Bir asamayi epoch epoch calistirir; en iyi val top1 icin best.pt gunceller."""
    best_val = best
    for ep in range(1, epochs + 1):
        t0 = time.time()
        lr_now = max(g["lr"] for g in optimizer.param_groups)

        tr = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler,
            grad_clip, limit_batches,
        )
        va = run_epoch(
            model, loaders["val"], criterion, device, None, None, 0, limit_batches
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "stage": stage,
            "epoch": ep,
            "lr": round(lr_now, 8),
            "train_loss": round(tr["loss"], 5),
            "train_top1": round(tr["top1"], 5),
            "train_top5": round(tr["top5"], 5),
            "val_loss": round(va["loss"], 5),
            "val_top1": round(va["top1"], 5),
            "val_top5": round(va["top5"], 5),
            "seconds": round(time.time() - t0, 1),
        }
        append_history(history_path, row)
        history.append(row)

        # en iyi val top1 -> best.pt
        if va["top1"] > best_val.get("val_top1", -1.0):
            best_val = {"val_top1": va["top1"], "val_top5": va["top5"], "stage": stage, "epoch": ep}
            save_checkpoint(
                run_dir / "best.pt", model, classes, arch, image_size, snapshot
            )

        # her epoch sonunda last.pt (kopek/kopma guvenligi)
        save_checkpoint(run_dir / "last.pt", model, classes, arch, image_size, snapshot)

        print(
            f"  [{stage}] epoch {ep}/{epochs}  "
            f"lr={lr_now:.1e}  "
            f"train loss={tr['loss']:.4f} top1={tr['top1']:.3f}  |  "
            f"val loss={va['loss']:.4f} top1={va['top1']:.3f} top5={va['top5']:.3f}  "
            f"({row['seconds']}s)"
        )
    return best_val


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------
def train_run(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Tum egitimi yonetir; koza ozeti dict dondurur."""
    tcfg = cfg.get("train", {})
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    # --- veri ---
    print("[1/5] Veri yukleniyor ...")
    bundle = build_from_config(cfg)
    classes = bundle["classes"]
    loaders = build_dataloaders(
        bundle["train"],
        bundle["val"],
        bundle["test"],
        batch_size=int(dcfg.get("batch_size", 32)),
        num_workers=int(dcfg.get("num_workers", 0)),
        seed=seed,
    )
    print(
        f"      sinif={len(classes)}  train={bundle['train_size']}  "
        f"val={bundle['val_size']}  test={bundle.get('test_size', 0)}  device={device}"
    )

    # --- model ---
    print("[2/5] Model kuruluyor (EfficientNet-B0) ...")
    image_size = int(dcfg.get("image_size", 224))
    model = build_model(
        num_classes=len(classes),
        weights=mcfg.get("pretrained_weights", "IMAGENET1K_V1"),
        dropout=float(mcfg.get("dropout", 0.2)),
    ).to(device)

    # --- ayarlar ---
    epochs1 = int(args.stage1_epochs if args.stage1_epochs is not None else tcfg.get("stage1", {}).get("epochs", 5))
    epochs2 = int(args.stage2_epochs if args.stage2_epochs is not None else tcfg.get("stage2", {}).get("epochs", 12))
    if args.no_stage2:
        epochs2 = 0
    unfreeze_n = int(
        args.unfreeze_from_block
        if args.unfreeze_from_block is not None
        else tcfg.get("unfreeze_from_block", -1)
    )
    head_lr1 = float(tcfg.get("stage1", {}).get("head_lr", 1e-3))
    head_lr2 = float(tcfg.get("stage2", {}).get("head_lr", 1e-3))
    bb_lr2 = float(tcfg.get("stage2", {}).get("backbone_lr", 1e-4))
    grad_clip = float(tcfg.get("grad_clip_norm", 0.0))
    label_smoothing = float(tcfg.get("label_smoothing", 0.0))
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    weight_decay = float(tcfg.get("weight_decay", 1e-4))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    # --- run klasoru + config snapshot ---
    run_dir = make_run_dir(cfg)
    snapshot = {
        **cfg,
        "_effective": {
            "device": str(device),
            "amp": use_amp,
            "stage1_epochs": epochs1,
            "stage2_epochs": epochs2,
            "unfreeze_from_block": unfreeze_n,
            "limit_batches": args.limit_batches,
            "num_classes": len(classes),
            "image_size": image_size,
        },
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    history_path = run_dir / cfg["logging"]["history_csv"]
    history: list[dict[str, Any]] = []
    best: dict[str, float] = {}

    print(f"[3/5] Run klasoru: {run_dir}")
    print(f"      AMP={use_amp}  stage1={epochs1}ep  stage2={epochs2}ep  unfreeze_from_block={unfreeze_n}")

    # --- ASAMA 1 ---
    if epochs1 > 0:
        print("[4/5] ASAMA 1 — donuk backbone, sadece head")
        freeze_backbone(model)
        s = trainable_frozen_summary(model)
        print(f"      egitilebilir {s['trainable']/1e6:.2f}M / toplam {s['total']/1e6:.2f}M")
        opt1 = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=head_lr1,
            weight_decay=weight_decay,
        )
        best = _train_stage(
            model, loaders, criterion, device, scaler, cfg, "stage1", epochs1,
            opt1, None, history_path, history, run_dir, classes,
            mcfg.get("arch", "efficientnet_b0"), image_size, snapshot,
            grad_clip, args.limit_batches, best,
        )

    # --- ASAMA 2 ---
    if epochs2 > 0:
        print(f"[4/5] ASAMA 2 — backbone aciliyor (from_block={unfreeze_n}), discriminative LR + cosine")
        unfreeze_from_block(model, unfreeze_n)
        s = trainable_frozen_summary(model)
        print(f"      egitilebilir {s['trainable']/1e6:.2f}M / toplam {s['total']/1e6:.2f}M")
        groups = discriminative_param_groups(model, head_lr=head_lr2, backbone_lr=bb_lr2)
        opt2 = torch.optim.AdamW(groups, weight_decay=weight_decay)
        sched = None
        if str(tcfg.get("stage2", {}).get("scheduler", "")).lower() == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(epochs2, 1))
        best = _train_stage(
            model, loaders, criterion, device, scaler, cfg, "stage2", epochs2,
            opt2, sched, history_path, history, run_dir, classes,
            mcfg.get("arch", "efficientnet_b0"), image_size, snapshot,
            grad_clip, args.limit_batches, best,
        )

    # --- TEST (sadece son degerlendirme) ---
    test_top1 = test_top5 = ""
    if "test" in loaders:
        print("[5/5] TEST degerlendirmesi (resmi test split) ...")
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model_state"])
        te = run_epoch(model, loaders["test"], criterion, device, None, None, 0, args.limit_batches)
        test_top1, test_top5 = round(te["top1"], 5), round(te["top5"], 5)
        print(f"      test top1={test_top1}  top5={test_top5}")

    # --- egriler + ozet ---
    plot_curves(
        history,
        run_dir / cfg["logging"]["curves_png"],
        f"{cfg['experiment_name']}  (en iyi val top1={best.get('val_top1', 0):.3f})",
    )

    summary = {
        "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "run_dir": run_dir.name,
        "experiment": cfg["experiment_name"],
        "device": str(device),
        "data_kind": dcfg.get("kind", "food101"),
        "num_classes": len(classes),
        "train_size": bundle["train_size"],
        "val_size": bundle["val_size"],
        "test_size": bundle.get("test_size", 0),
        "stage1_epochs": epochs1,
        "stage2_epochs": epochs2,
        "unfreeze_from_block": unfreeze_n,
        "best_val_top1": best.get("val_top1", ""),
        "best_val_top5": best.get("val_top5", ""),
        "test_top1": test_top1,
        "test_top5": test_top5,
    }
    append_summary(resolve_path(cfg["logging"]["runs_dir"]), summary)

    print(f"\nBitti. Run: {run_dir}")
    print(f"  en iyi val top1 = {summary['best_val_top1']}")
    if test_top1 != "":
        print(f"  test top1/top5  = {test_top1} / {test_top5}")
    return {"run_dir": str(run_dir), "summary": summary, "history": history}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FoodLens egitim (iki asamali fine-tune)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--unfreeze-from-block", type=int, default=None,
                   help="features indeksi (-1 tum backbone); config'i ezer")
    p.add_argument("--stage1-epochs", type=int, default=None)
    p.add_argument("--stage2-epochs", type=int, default=None)
    p.add_argument("--no-stage2", action="store_true", help="sadece asama 1")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="epoch basina batch siniri (yerel smoke; 0=tum)")
    p.add_argument("--device", default=None, help="cpu|cuda (varsayilan otomatik)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    train_run(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
