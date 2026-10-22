"""Food-101 on-dogrulama CLI (Colab). Tutmazsa cikis kodu 1 -> egitim baslatilmez.

Colab'da tam indirme SONRASI calistirin (notebook hucresi veya %run):
    python scripts/validate_food101.py --config configs/default.yaml

Beklenen: 101 sinif, resmi train 75.750, test 25.250, val 7.575, kalan train 68.175.
Basarisizlikta stderr'e yazar ve exit 1 doner; boylece egitim adimi gecmez.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodlens.config import load_config, resolve_path  # noqa: E402
from foodlens.validate import (  # noqa: E402
    Food101ValidationError,
    format_validation_report,
    validate_food101,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Food-101 indirme sonrasi on-dogrulama")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--root", default=None, help="Override: food-101 kok dizini")
    p.add_argument("--val-ratio", type=float, default=None, help="Override: val orani")
    p.add_argument("--seed", type=int, default=None, help="Override: split seed")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    d = cfg["data"]
    root = Path(args.root) if args.root else resolve_path(d["root"])
    val_ratio = args.val_ratio if args.val_ratio is not None else float(d.get("val_ratio", 0.10))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))

    try:
        info = validate_food101(root, val_ratio=val_ratio, seed=seed)
    except FileNotFoundError as exc:
        print(f"Food-101 on-dogrulama BASARISIZ (egitim baslatilmadi):\n  - {exc}", file=sys.stderr)
        return 1
    except Food101ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_validation_report(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
