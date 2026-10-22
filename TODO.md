# TODO.md — FoodLens Yapılacaklar

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince [PROGRESS.md](PROGRESS.md) + gerekirse bu dosyayı güncelle.
> Durum işaretleri: `[ ]` açık · `[~]` devam · `[x]` bitmiş · `[BLOCKED]` engelli.

## Sıradaki (Acil)
- [ ] **Aşama 2 commit:** `src/foodlens/model.py` + `tests/test_model.py` → `Stage 2: model (EfficientNet-B0 + 101 head, freeze/unfreeze, discriminative LR)`. (Commit onayı bekliyor.)
- [ ] **`yemek_api.txt` kararı:** arkadaşının FastAPI referansı commit edilsin mi, yoksa `.gitignore`'a mı eklensin? Karar ver.

## Aşama 2-API — FastAPI Entegrasyonu (checkpoint'e hazır)
- [ ] `src/foodlens/data.py`'ye `preprocess_image(pil_img, cfg) -> Tensor` (eval transform, augment'siz).
- [ ] `app/api.py` — FastAPI: `GET /`, `POST /predict` (arkadaşının iskeletini koru). `best.pt` varsa yükle → gerçek top-5; YOKSA mock fallback.
- [ ] Checkpoint contract'ı api.py'de uygula (`classes` checkpoint'ten).
- [ ] `requirements.txt`: `fastapi`, `uvicorn[standard]`, `python-multipart`.
- [ ] `tests/test_api.py` — mock mod + (checkpoint varsa) gerçek tahmin testi.

## Aşama 3 — Eğitim + Colab
- [ ] `src/foodlens/train.py`: AMP, label smoothing, grad clip, iki aşama (stage1 head-only → stage2 discriminative LR + cosine), `--unfreeze-from-block N`.
- [ ] Checkpoint contract'ı train.py'de uygula (`model_state`, `classes`, `arch`, `num_classes`, `image_size`, `config`).
- [ ] `runs/<tarih>_<deney_ad>/` logging: `config.json`, `history.csv` (**"stage" sütunu**), `curves.png`, `best.pt`, `last.pt` + `runs/summary.csv`. Mevcut klasör üzerine yazma.
- [ ] `notebooks/food101_colab.ipynb` — Full Food-101 indir (Colab) → `validate_food101.py` **zorunlu pre-flight** → subset/egitim → checkpoint'i indir.
- [ ] `runs/` için `.gitignore` ayarı (checkpoint'ler commit edilmesin).

## Aşama 4 — UI + Açıklanabilirlik
- [ ] `src/foodlens/explain.py` — Grad-CAM, hedef katman `features[-1]` (= `features.8`), Top-1 sınıf.
- [ ] `app/` Gradio UI — top-5 + Grad-CAM heatmap; tek görsel yükleme (batch YOK).
- [ ] `app/api.py`'ye `/explain` endpoint'i (Grad-CAM) ekle.

## Entegrasyon / Teslim
- [ ] Arkadaşın private repoya (`myusufay/Food-Image-Classification`) branch push:
  - `app/api.py` mantığını onun `main.py`'ine taşı + `src/foodlens/`'i vendor'layarak self-contained yap.
  - **Push'u kullanıcı yapar** (contributor). AI `git remote add` + `git push` komutlarını verir, token istemez.
- [ ] README.md'yi güncelle (kullanım, Colab, API, Gradio).

## Nice-to-have (sonra)
- [ ] confusion matrix + en çok karışan sınıflar raporu.
- [ ] test-time augmentation (TTA) opsiyonu.
- [ ] Docker (FastAPI servisi için).
- [ ] Model benchmark tablosu (top-1/top-5, EfficientNet-B0 vs baseline).
