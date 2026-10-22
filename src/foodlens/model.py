"""FoodLens - model katmani (Asama 2).

EfficientNet-B0 + ImageNet-1K (IMAGENET1K_V1) agirliklari + Food-101 (101 sinif) head.

İki asamali ince ayar (fine-tuning) icin dondurme/cozme yardimcilari:
    freeze_backbone(model)          -> sadece head egitilir (asama 1)
    unfreeze_all(model)             -> tum katmanlar acilir (asama 2)
    unfreeze_from_block(model, n)   -> features[n:] + head; n=-1 tum backbone

Discriminative LR (asama 2) icin param grup uretimi:
    discriminative_param_groups(model, head_lr, backbone_lr)

Backbone = model.features (features[0]=stem, [1..7]=MBConv, [8]=conv head).
Head     = model.classifier (Dropout + Linear(1280, num_classes)).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

_FEAT = "features"
_CLF = "classifier"


# ---------------------------------------------------------------------------
# Kurulum
# ---------------------------------------------------------------------------
def build_model(
    num_classes: int = 101,
    weights: str | EfficientNet_B0_Weights | None = "IMAGENET1K_V1",
    dropout: float = 0.2,
) -> nn.Module:
    """EfficientNet-B0 kurar ve Food-101 head'ini takar.

    weights: EfficientNet_B0_Weights enum'u, isim string'i ("IMAGENET1K_V1") veya None.
    dropout: head'deki Dropout olasiligi (config'ten).
    """
    if isinstance(weights, str):
        weights = EfficientNet_B0_Weights[weights]
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _features(model: nn.Module) -> nn.Module:
    return getattr(model, _FEAT)


def _head(model: nn.Module) -> nn.Module:
    return getattr(model, _CLF)


def _set_requires_grad(module: nn.Module, flag: bool) -> int:
    """require_grad'i ayarlar; toplam parametre sayisini dondurur."""
    n = 0
    for p in module.parameters():
        p.requires_grad = flag
        n += p.numel()
    return n


# ---------------------------------------------------------------------------
# Dondurme / cozme (iki asamali ince ayar)
# ---------------------------------------------------------------------------
def freeze_backbone(model: nn.Module) -> None:
    """Backbone'u dondurur; head acik birakilir (asama 1: sadece head)."""
    _set_requires_grad(_features(model), False)
    _set_requires_grad(_head(model), True)


def unfreeze_all(model: nn.Module) -> None:
    """Tum katmanlari (backbone + head) acar (asama 2)."""
    _set_requires_grad(_features(model), True)
    _set_requires_grad(_head(model), True)


def unfreeze_from_block(model: nn.Module, n: int) -> None:
    """Backbone'u dondurup features[n:] + head'i acar.

    n: features modul indeksi (0..8; 0=stem, 1..7=MBConv, 8=conv head).
    n=-1 -> tum backbone (tum bloklar). Ornek: n=5 -> features[5..8] + head.
    Head her durumda acik birakilir.
    """
    feats = _features(model)
    length = len(feats)
    _set_requires_grad(feats, False)
    if n == -1:
        _set_requires_grad(feats, True)
    else:
        if not (0 <= n < length):
            raise ValueError(
                f"blok indeksi -1 veya [0, {length - 1}] araliginda olmali; bulundu: {n}"
            )
        for mod in list(feats)[n:]:
            _set_requires_grad(mod, True)
    _set_requires_grad(_head(model), True)


# ---------------------------------------------------------------------------
# Discriminative LR param gruplari + ozet
# ---------------------------------------------------------------------------
def discriminative_param_groups(
    model: nn.Module, head_lr: float, backbone_lr: float
) -> list[dict[str, Any]]:
    """Discriminative LR icin optimizer param gruplari uretir.

    Sadece require_grad=True olan parametreler eklenir (donukler optimizer'a girmez).
    Donus: [{"params": [...], "lr": backbone_lr}, {"params": [...], "lr": head_lr}]
    """
    backbone = [p for p in _features(model).parameters() if p.requires_grad]
    head = [p for p in _head(model).parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    if head:
        groups.append({"params": head, "lr": head_lr})
    if not groups:
        raise ValueError("egitilecek parametre yok (tumu donuk); once unfreeze_* cagirin")
    return groups


def trainable_frozen_summary(model: nn.Module) -> dict[str, int]:
    """Backbone/head icin egitilebilir/donuk parametre sayilarini ozetler."""
    bb = list(_features(model).parameters())
    hd = list(_head(model).parameters())
    bb_total = sum(p.numel() for p in bb)
    bb_train = sum(p.numel() for p in bb if p.requires_grad)
    hd_total = sum(p.numel() for p in hd)
    hd_train = sum(p.numel() for p in hd if p.requires_grad)
    return {
        "backbone_total": bb_total,
        "backbone_trainable": bb_train,
        "head_total": hd_total,
        "head_trainable": hd_train,
        "total": bb_total + hd_total,
        "trainable": bb_train + hd_train,
        "frozen": (bb_total - bb_train) + (hd_total - hd_train),
    }


def format_summary(s: dict[str, int]) -> str:
    """Ozet dict'ini insan-okur metin olarak formatlar."""

    def _m(n: int) -> str:
        return f"{n / 1e6:.2f}M"

    return (
        f"  backbone : toplam {_m(s['backbone_total'])}, egitilebilir {_m(s['backbone_trainable'])}\n"
        f"  head     : toplam {_m(s['head_total'])}, egitilebilir {_m(s['head_trainable'])}\n"
        f"  toplam   : {_m(s['total'])} | egitilebilir {_m(s['trainable'])} | donuk {_m(s['frozen'])}"
    )
