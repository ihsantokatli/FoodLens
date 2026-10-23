# RULES.md — FoodLens Teknolojiler & Kodlama Standartları

> Bu dosya AI Hafıza Sistemi'nin kurallar kaynağıdır.
> Değişiklik gerektiğinde bu dosyayı güncelle.

## 1. Proje Kimliği
- **Ad:** FoodLens — Food-101 yemek sınıflandırma.
- **Amaç:** EfficientNet-B0 transfer learning (PyTorch/torchvision) + Gradio UI + FastAPI mikroservis.
- **Ekip:** İhsan (veri/ön işleme), Emre (model), myusufay/Muhammet (FastAPI iskeleti).
- **Package:** `foodlens` (src layout: `src/foodlens/`).

## 2. Teknolojiler (sürümler yerel ortamda doğrulandı)
| Teknoloji | Sürüm | Not |
|---|---|---|
| Python | 3.13.2 | `requires-python >= 3.9` |
| torch | 2.14.1+cpu | Colab'da GPU sürümü farklıdır |
| torchvision | 0.29.1+cpu | EfficientNet `features` = 9 modül |
| gradio | 6.30.0 | UI (Aşama 4) |
| fastapi + uvicorn | — | Mikroservis (`app/api.py`) |
| pytest | >=7.4 | Testler |
| PyYAML, numpy, Pillow, matplotlib, scikit-learn, tqdm | — | Yardımcı |

**COLAB KURALI:** `torch`/`torchvision` Colab'da önceden kuruludur ve GPU ile eşleşir.
Colab'da `pip` ile AYRI AYRI kurma — `requirements.txt`'ten diğer paketleri kur, ikisini atla.

## 3. Kodlama Standartları
- **Yollar:** Her zaman `pathlib.Path`. Asla string birleştirme ile yol kurma.
- **Config yolları:** `"/"` ayracıyla yaz (`configs/*.yaml`). Windows + Colab uyumu için.
- **Docstring/yorum:** Türkçe, `"""..."""`. Modül başında ne yaptığı + hangi aşamada eklendiği.
- **Tip ipuçları:** `from __future__ import annotations`; fonksiyonlarda type hint.
- **Import:** Paket içi `from .modul import ...`; script'ler `sys.path.insert(0, .../src)` ile kendi kendine yükler (`pip install -e .` gerekmez).
- **Adlandırma:** fonksiyon/degisken `snake_case`, sınıf `CamelCase`, config anahtarları `snake_case`.
- **Testler:** `tests/` altında; `conftest.py` `src/` + `scripts/` yolunu ekler ve toy fixture verir.
- **Saflık:** İş mantığı `src/foodlens/` içinde, ince CLI sarmalayıcılar `scripts/` içinde.

## 4. Mimari Kararlar (Değişmezler)
- **Veri yolu:** Tam Food-101 (5.4 GB) **asla yerelde indirilmez**. Yerelde sadece toy/smoke subset. Tam indirme **Colab'da**.
- **Val split:** Train split içinden **stratified %10**, `seed=42`. İndeksler `data/val_indices.json`'a yazılır (tekrar üretilebilir). Resmi **test** split sadece son değerlendirmede.
- **Subset:** 80 train / 20 val per class → `val_ratio: 0.20`.
- **Meta formatı:** `data.py` `.txt` öncelikli okur (gerçek arşiv satır formatı `sinif/id`, uzantısız), `.json` fallback. İkisi **birebir aynı** örnek listesini vermeli (kanonik sıralama).
- **Model:** EfficientNet-B0 `IMAGENET1K_V1`, 101 sınıf head (`Dropout(0.2) + Linear(1280,101)`). Backbone = `model.features` (0=stem, 1..7=MBConv, **8=conv head=`features[-1]`**). Head = `model.classifier`.
- **Fine-tune (iki aşama):**
  - Aşama 1: `freeze_backbone` — sadece head, 5 epoch, head LR `1e-3`.
  - Aşama 2: `unfreeze_all`/`unfreeze_from_block(model,n)` — discriminative LR (backbone `1e-4`, head `1e-3`), cosine, 10-15 epoch. `train.py --unfreeze-from-block N` (n=-1 tümü).
- **Logging:** `runs/<tarih>_<deney_ad>/` → `config.json`, tek `history.csv` (**"stage" sütunu**), `curves.png`, `best.pt`, `last.pt` + `runs/summary.csv`. **Mevcut run klasörü asla üzerine yazma.** `reports/` klasörü YOK.
- **Checkpoint contract:** `best.pt`/`last.pt` = `{"model_state", "classes", "arch", "num_classes", "image_size", "config"}`. train.py ve api.py bu sözleşmeye uyar.
- **Grad-CAM:** `src/foodlens/explain.py`, hedef katman `features[-1]` (= `features.8`), Top-1 sınıf için. Gradio'da top-5 + heatmap. Batch yükleme YOK.

## 5. Git / İş Akışı Kuralları
- **Aşama başına tek commit.** Mesaj biçimi: `Stage N: <açıklama>` (Türkçe/İngilizce karışık serbest).
- **Her aşama:** çalıştır → çıktıyı göster → **onay bekle** → sıradakine geç.
- Commit kimliği: `-c user.name="FoodLens" -c user.email="foodlens@local"`.
- `checkpoints/` klasörü YOK (çıktılar `runs/` altına).
- Arkadaşın private repoya push: **kullanıcı (İhsan) yapar**, AI token istemez/ handles etmez.

## 6. Colab (Free T4, 16 GB RAM)
- Batch 64 (OOM ise 48/32), `num_workers=2`, AMP açık.
- `notebooks/` altında notebook (Aşama 3).
- Eğitim öncesi `scripts/validate_food101.py` **zorunlu** — tutmazsa `exit 1`, eğitim başlamaz.
  Beklenen: 101 sınıf, train 75.750, test 25.250, val 7.575, kalan train 68.175.
