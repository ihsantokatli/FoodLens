"""FoodLens egitim CLI (ince sarmalayici) - Asama 3.

Is mantigi src/foodlens/train.py icinde; burada sadece src/ yolunu ekleyip
main() cagirilir (pip install -e gerekmez).

Kullanim:
    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/default.yaml --unfreeze-from-block -1
    # yerel hizli smoke (epoch/batch sinirli):
    python scripts/train.py --config configs/smoke.yaml --stage1-epochs 1 --stage2-epochs 1 --limit-batches 3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodlens.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
