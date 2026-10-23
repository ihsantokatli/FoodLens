"""FoodLens - aciklanabilirlik (Asama 4): Grad-CAM.

Hedef katman: `features[-1]` (= `features.8`, son konvolusyon blogu).
Top-1 sinif icin isi haritasi (heatmap) uretir ve PIL gorsel uzerine bindirir.
Gradio UI ve /explain endpoint'i bu modulu kullanir.

Grad-CAM (Selvaraju et al., 2017): hedef katman aktivasyonlari A ve sinif
skorunun A'ya gore gradyani kullanilir:
    weight = mean(grad over spatial)   -> kanal onemi
    cam    = ReLU(sum(weight * A))     -> isi haritasi
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import preprocess_image


def resolve_target_layer(model, name: str | None = None):
    """Hedef katmani cozer.

    name=None -> model.features[-1] (son konvolusyon = features.8).
    name="features.8" / "features[-1]" gibi nokta/indeks yolu desteklenir.
    """
    if not name:
        return model.features[-1]
    obj = model
    for part in name.split("."):
        token = part.strip()
        # "features[-1]" -> getattr(model, "features")[-1]
        if token.endswith("]") and "[" in token:
            base, idx = token[:-1].split("[", 1)
            obj = getattr(obj, base)[int(idx)]
        elif token.lstrip("-").isdigit():
            obj = obj[int(token)]
        else:
            obj = getattr(obj, token)
    return obj


def compute_gradcam(
    model,
    input_tensor: torch.Tensor,
    target_layer=None,
    class_idx: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Tek goruntu icin Grad-CAM hesaplar.

    input_tensor: (1, C, H, W) ya da (C, H, W). class_idx=None -> top-1.
    Donus: (cam [H,W] 0..1 numpy, class_idx, probs [num_classes] numpy)
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    if target_layer is None:
        target_layer = model.features[-1]

    activations: list[torch.Tensor] = []
    hook = target_layer.register_forward_hook(
        lambda _m, _i, out: activations.append(out)
    )
    was_training = model.training
    model.eval()
    try:
        x = input_tensor.detach().clone().to(device)
        x.requires_grad_(True)  # gradyan grafigi (parametreler donuk olsa da)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if class_idx is None:
            class_idx = int(logits[0].argmax().item())

        act = activations[-1]  # (1, C, h, w)
        # sadece aktivasyona gore gradyan al -> parametre .grad'larini kirletmez
        grads = torch.autograd.grad(logits[0, class_idx], act, retain_graph=False)[0]
    finally:
        hook.remove()
        if was_training:
            model.train()

    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = torch.relu((weights * act).sum(dim=1))[0]  # (h, w)
    cam = cam - cam.min()
    # goreli normalizasyon: mutlak epsilon kullanma (cok kucuk degerlerde
    # epsilon baskin cikip cam'i sifira yaklastirir)
    cam_max = cam.max()
    if float(cam_max.detach()) > 0:
        cam = cam / cam_max
    cam = F.interpolate(
        cam[None, None],
        size=tuple(input_tensor.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return cam.detach().cpu().numpy(), int(class_idx), probs


def overlay_cam(
    pil_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Isi haritasini PIL gorsel uzerine bindirir (orijinal boyutu korur)."""
    import matplotlib

    cmap = matplotlib.colormaps[colormap]
    base = pil_img.convert("RGB")
    w, h = base.size

    # cam model giris boyutunda (or. 224x224); goruntuye geri olcekle
    if cam.shape != (h, w):
        cam_u8 = (np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)
        cam = np.asarray(Image.fromarray(cam_u8).resize((w, h), Image.BILINEAR))
        cam = cam.astype(np.float32) / 255.0

    heat = (cmap(np.clip(cam, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)
    arr = (1.0 - alpha) * np.asarray(base).astype(np.float32) + alpha * heat.astype(
        np.float32
    )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def explain_image(
    model,
    pil_image: Image.Image,
    classes: list[str],
    image_size: int = 224,
    target_layer=None,
    topk: int = 5,
) -> dict:
    """Tek goruntu: on isleme -> Grad-CAM -> top-k tahmin + overlay.

    Donus: {"overlay": PIL, "cam": np[H,W], "class_idx": int,
            "predictions": [(sinif_adi, olasilik), ...] (azalan)}
    """
    x = preprocess_image(pil_image, image_size).unsqueeze(0)
    cam, class_idx, probs = compute_gradcam(
        model, x, target_layer=target_layer, class_idx=None
    )
    overlay = overlay_cam(pil_image, cam)
    k = min(topk, len(classes))
    order = np.argsort(probs)[::-1][:k]
    predictions = [(classes[int(i)], float(probs[int(i)])) for i in order]
    return {
        "overlay": overlay,
        "cam": cam,
        "class_idx": class_idx,
        "predictions": predictions,
    }
