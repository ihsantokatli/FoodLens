# FoodLens

Food-101 veri seti üzerinde **EfficientNet-B0** transfer learning ile yemek fotoğrafı
sınıflandırma: eğitim + Grad-CAM açıklanabilirlik + Gradio arayüzü + FastAPI servisi.

| Metrik (Food-101 resmi test, 25.250 görsel) | Sonuç |
|---|---|
| **Top-1 doğruluk** | **%84,10** |
| **Top-5 doğruluk** | **%96,16** |
| En iyi doğrulama top-1 | %79,91 (stage2 epoch 11) |
| Eğitim | 5 + 12 epoch · Colab Free T4 · AMP · batch 64 |

- **Eğitim:** PyTorch + torchvision (Colab'da GPU, iki oturumda `--resume` ile tamamlandı)
- **Arayüz:** Gradio (top-5 tahmin + Grad-CAM ısı haritası)
- **Servis:** FastAPI (`/predict`, `/explain`)
- **Paket:** `foodlens` (src layout)

---

## Hızlı başlangıç

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Model ağırlığını yerleştir (repoya eklenmez):

```
runs/20261024_food101_effb0/best.pt
```

```bash
# FastAPI servisi -> http://127.0.0.1:8000/docs
uvicorn app.api:app --reload --port 8000

# Gradio arayüzü -> http://127.0.0.1:7860
python app/gradio_app.py
```

`best.pt` yoksa servis **mock** modda çalışır (`kaynak: "mock"`); Gradio bilgi mesajı gösterir.

---

## Proje yapısı

```
FoodLens/
├── src/foodlens/
│   ├── config.py      # YAML yükleme, "/" ayraçlı yollar -> Path
│   ├── data.py        # transforms, stratified val split, subset, preprocess_image
│   ├── model.py       # EfficientNet-B0 + 101 sınıf head, freeze/unfreeze
│   ├── train.py       # iki aşamalı fine-tune, runs/ logging, --resume
│   ├── explain.py     # Grad-CAM (features[-1] = features.8)
│   └── validate.py    # Food-101 ön doğrulama
├── app/
│   ├── api.py         # FastAPI: /predict, /explain
│   └── gradio_app.py  # Gradio: top-5 + heatmap
├── scripts/           # ince CLI sarmalayıcılar
├── notebooks/         # food101_colab.ipynb (12 hücre)
├── configs/           # default.yaml (tam), smoke.yaml (yerel)
├── tests/             # 57 test
└── runs/              # eğitim çıktıları (gitignore)
```

---

## Eğitim

Yerel smoke (CPU, indirmesiz):

```bash
python scripts/train.py --config configs/smoke.yaml --stage1-epochs 1 --stage2-epochs 1 --limit-batches 3
```

Tam eğitim **Colab'da** (`notebooks/food101_colab.ipynb`):

1. `FOODLENS_REPO_URL` → repo klonlanır, paketler kurulur
2. Food-101 indirilir (~5.4 GB — **yerelde asla indirilmez**)
3. `scripts/validate_food101.py` **zorunlu** (101 sınıf / 75.750 / 25.250 / 7.575 / 68.175)
4. **4a) Drive** hücresi → `RUNS_DIR = /content/drive/MyDrive/FoodLens/runs`
5. Eğitim: stage1 (5 epoch, donuk backbone) → stage2 (12 epoch, discriminative LR + cosine)

### Resume (oturum koparsa)

`last.pt` optimizer + scheduler + scaler + `stage`/`epoch` + `best` + `history` taşır:

```bash
python scripts/train.py --config configs/default.yaml --resume runs/<tarih>_<ad>/last.pt
```

Notebook'ta `RESUME_DIR` değişkenini doldurup eğitim hücresini tekrar çalıştırmak yeterli;
tamamlanmış stage atlanır, `history.csv` yeniden yazılır, `summary.csv` upsert edilir.

### Çıktılar

`runs/<tarih>_<deney_ad>/` (mevcut klasör **asla ezilmez**):

| Dosya | İçerik |
|---|---|
| `config.json` | O koşunun konfigürasyonu + `_effective` |
| `history.csv` | `stage,epoch,lr,train_loss,train_top1,...,val_top5,seconds` |
| `curves.png` | Loss + top-1 eğrileri (stage geçişi işaretli) |
| `best.pt` | En iyi val top-1 (API sözleşmesi) |
| `last.pt` | Son epoch + `train_state` (resume) |
| `runs/summary.csv` | Koşu başına tek satır özet |

### Checkpoint sözleşmesi

```python
{"model_state", "classes", "arch", "num_classes", "image_size", "config"}
```

`train_state` yalnızca `last.pt`'de bulunur; `best.pt` bu 6 anahtarı korur.

---

## API

```bash
uvicorn app.api:app --reload --port 8000   # http://127.0.0.1:8000/docs
```

| Endpoint | Açıklama |
|---|---|
| `GET /` | Karşılama mesajı |
| `POST /predict` | Top-5 sınıf + güven skoru |
| `POST /explain` | Grad-CAM ısı haritası (base64 PNG) + top-5 |

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@pizza.jpg"
```

```json
{
  "tahmin_edilen_yemek": "pizza",
  "dogruluk_orani": 61.95,
  "durum": "Başarılı",
  "kaynak": "model",
  "tahminler": [
    {"sinif": "pizza", "olasilik": 61.95},
    {"sinif": "paella", "olasilik": 17.83}
  ],
  "checkpoint": "runs/20261024_food101_effb0/best.pt",
  "dosya_adi": "pizza.jpg"
}
```

- Model yoksa `/predict` **mock** döner; `/explain` **503**.
- JPG/PNG dışı **415**, bozuk görsel **400**.
- Checkpoint arama: `FOODLENS_CHECKPOINT` → `runs/*/best.pt` (en yeni) → yok.

## Gradio

```bash
python app/gradio_app.py
```

Tek görsel → top-5 tablo + Grad-CAM overlay (batch yükleme yok). Hedef katman
`configs/default.yaml` → `gradio.gradcam_layer: features.8` (= `features[-1]`).

---

## Testler

```bash
python -m pytest tests/ -q     # 57 passed
```

| Dosya | Test |
|---|---|
| `test_data.py` | 19 |
| `test_model.py` | 11 |
| `test_api.py` | 8 |
| `test_train.py` | 9 |
| `test_explain.py` | 8 |
| `test_gradio_app.py` | 2 |

---

## Entegrasyon (teslim)

`myusufay/Food-Image-Classification` reposuna taşınan sürüm:
`yemek_API/yemek_API.py` (FastAPI) + vendor `foodlens/` + `configs/` + `requirements.txt`.
Arkadaşın mock alanları (`tahmin_edilen_yemek`, `dogruluk_orani`, `durum`) korunur.
Dal: `feature/foodlens-integration`.

---

## Aşamalar

| Aşama | İçerik | Durum |
|---|---|---|
| 0 | Proje iskeleti | ✅ |
| 1 | Veri katmanı | ✅ |
| 2 | Model katmanı | ✅ |
| 2-API | FastAPI servisi | ✅ |
| 3 | Eğitim + Colab defteri | ✅ |
| 4 | Grad-CAM + Gradio + `/explain` | ✅ |
| 4.5 | Resume + Drive desteği | ✅ |
| 5 | Colab tam eğitim (%84,10 / %96,16) | ✅ |
| 6 | Entegrasyon paketi + README | ✅ |

Detaylar: [PROGRESS.md](PROGRESS.md) · Kurallar: [RULES.md](RULES.md) · Yapılacaklar: [TODO.md](TODO.md)
