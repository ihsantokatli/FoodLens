"""Veri katmani ciktisi: per-class dagilim raporu + ornek grid gorseli (Asama 1).

Config'i yukler, dogru dataset'i kurar, dagilim raporunu basar ve grid kaydeder.

Kullanim:
    python scripts/explore_data.py --config configs/smoke.yaml
    python scripts/explore_data.py --config configs/default.yaml --grid data/toy_grid.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodlens.config import load_config  # noqa: E402
from foodlens.data import (  # noqa: E402
    build_from_config,
    format_distribution_report,
    save_sample_grid,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Per-class dagilim + ornek grid uretir")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--grid", default=None, help="Grid cikti yolu (varsayilan: data/_preview/<ad>_grid.png)")
    p.add_argument("--grid-n", type=int, default=16)
    p.add_argument("--download", action="store_true", help="food101 icin indirme ac (Colab)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    bundle = build_from_config(cfg, download=args.download)

    print(f"config      : {args.config}")
    print(f"kind        : {cfg['data'].get('kind')}")
    print(f"classes     : {len(bundle['classes'])}")
    print(f"train/val/test : {bundle['train_size']}/{bundle['val_size']}/{bundle['test_size']}")
    if bundle.get("val_indices_path"):
        print(f"val_indices : {bundle['val_indices_path']}")
    print("-" * 60)
    print(format_distribution_report(bundle))
    print("-" * 60)

    grid_path = Path(args.grid) if args.grid else Path("data/_preview") / f"{cfg.get('experiment_name', 'exp')}_grid.png"
    out = save_sample_grid(
        bundle["train"],
        bundle["classes"],
        grid_path,
        n=args.grid_n,
        title=f"train ornekleri ({cfg.get('experiment_name', '')})",
    )
    print(f"grid kaydedildi: {out}")


if __name__ == "__main__":
    main()
