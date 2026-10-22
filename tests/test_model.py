"""Asama 2 testleri: model (EfficientNet-B0 + 101 head, dondurme, discriminative LR)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from foodlens.model import (
    build_model,
    discriminative_param_groups,
    format_summary,
    freeze_backbone,
    trainable_frozen_summary,
    unfreeze_all,
    unfreeze_from_block,
)

NUM_CLASSES = 101


@pytest.fixture(scope="module")
def model() -> nn.Module:
    """Pretrained (cached) EfficientNet-B0 + 101 head. Her test kendi durumunu kurar."""
    return build_model(num_classes=NUM_CLASSES, weights="IMAGENET1K_V1", dropout=0.2)


def _trainable_indices(module: nn.Module) -> set[int]:
    """require_grad=True olan parametrelerin kimliklerini (id) dondurur."""
    return {id(p) for p in module.parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# (N, 101) cikti + head kurulumu
# ---------------------------------------------------------------------------
def test_build_model_output_shape(model):
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, NUM_CLASSES)


def test_head_is_101_way(model):
    lin = model.classifier[1]
    assert isinstance(lin, nn.Linear)
    assert lin.in_features == 1280
    assert lin.out_features == NUM_CLASSES
    # Dropout config ile ayni (0.2)
    assert model.classifier[0].p == 0.2


def test_build_model_weights_none_no_download():
    m = build_model(num_classes=NUM_CLASSES, weights=None)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, NUM_CLASSES)


# ---------------------------------------------------------------------------
# dondurme / cozme
# ---------------------------------------------------------------------------
def test_freeze_backbone_only_head_trainable(model):
    freeze_backbone(model)
    assert _trainable_indices(model.features) == set()  # backbone tam donuk
    assert _trainable_indices(model.classifier) != set()  # head acik
    # head'de sadece Linear agirliklar + bias (Dropout parametresiz)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == (
        model.classifier[1].weight.numel() + model.classifier[1].bias.numel()
    )


def test_unfreeze_all_everything_trainable(model):
    unfreeze_all(model)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert train == total


def test_unfreeze_from_block_opens_suffix_and_head(model):
    unfreeze_from_block(model, 5)
    feats = list(model.features)
    # features[:5] donuk
    for mod in feats[:5]:
        assert _trainable_indices(mod) == set()
    # features[5:] acik (MBConv 5,6,7 + conv head 8)
    for mod in feats[5:]:
        assert _trainable_indices(mod) != set()
    # head acik
    assert _trainable_indices(model.classifier) != set()


def test_unfreeze_from_block_minus1_opens_all(model):
    unfreeze_from_block(model, -1)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert train == total  # tum bloklar


def test_unfreeze_from_block_invalid_raises(model):
    with pytest.raises(ValueError):
        unfreeze_from_block(model, 9)  # 0..8 disi
    with pytest.raises(ValueError):
        unfreeze_from_block(model, -2)


# ---------------------------------------------------------------------------
# discriminative LR param gruplari
# ---------------------------------------------------------------------------
def test_discriminative_param_groups_respect_trainable_and_lr(model):
    freeze_backbone(model)  # sadece head trainable
    groups = discriminative_param_groups(model, head_lr=1e-3, backbone_lr=1e-4)
    # sadece head oldugu icin tek grup
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3
    head_params = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in head_params}

    unfreeze_all(model)  # hepsi trainable -> 2 grup
    groups = discriminative_param_groups(model, head_lr=1e-3, backbone_lr=1e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4  # backbone
    assert groups[1]["lr"] == 1e-3  # head
    # gruplar toplami = tum trainable parametreler (donuk yok)
    n_group = sum(len(g["params"]) for g in groups)
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_group == n_train


def test_discriminative_param_groups_all_frozen_raises(model):
    freeze_backbone(model)
    for p in model.classifier.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError):
        discriminative_param_groups(model, head_lr=1e-3, backbone_lr=1e-4)
    # geri al (diger testler)
    for p in model.classifier.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# ozet
# ---------------------------------------------------------------------------
def test_trainable_frozen_summary_counts(model):
    freeze_backbone(model)
    s = trainable_frozen_summary(model)
    assert s["backbone_trainable"] == 0
    assert s["head_trainable"] == s["head_total"]  # Linear 1280*101 + 101
    assert s["head_total"] == 1280 * NUM_CLASSES + NUM_CLASSES
    assert s["frozen"] == s["backbone_total"]
    assert s["trainable"] == s["head_total"]

    txt = format_summary(s)
    assert "backbone" in txt and "head" in txt and "egitilebilir" in txt
