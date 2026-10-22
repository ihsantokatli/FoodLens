# FoodLens

Food-101 veri seti üzerinde EfficientNet-B0 ile transfer learning kullanarak yemek
fotograflarindan yemek turunu siniflandiran proje.

- **Egitim:** PyTorch + torchvision (Colab Free T4)
- **Arayuz:** Gradio (top-5 tahmin + Grad-CAM)
- **Paket:** `foodlens` (src/ altinda)

## Kurulum

```bash
pip install -r requirements.txt
```

## Yapilandirma

- `configs/default.yaml` — tam Food-101 egitimi (101 sinif, Colab)
- `configs/smoke.yaml` — yerel kucuk test (10 sinif, CPU)

## Asamalar

| Asama | Icerik | Durum |
|-------|--------|-------|
| 0 | Proje iskeleti | Tamamlandi |
| 1 | Veri katmani | Bekliyor |
| 2 | Model katmani | Bekliyor |
| 3 | Egitim | Bekliyor |
| 4 | Degerlendirme | Bekliyor |
| 5 | Gradio arayuzu | Bekliyor |
| 6 | Dokumantasyon | Bekliyor |
