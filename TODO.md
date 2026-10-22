# TODO.md — FoodLens Yapılacaklar

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince [PROGRESS.md](PROGRESS.md) + gerekirse bu dosyayı güncelle.
> Durum işaretleri: `[ ]` açık · `[~]` devam · `[x]` bitmiş · `[BLOCKED]` engelli.

## Sıradaki (Acil)
- [x] **Aşama 2 commit:** `Stage 2: model (...)` → `a495bf0`.
- [x] **`yemek_api.txt` kararı:** `.gitignore`'a eklendi (yerel referans, commit edilmedi).
- [x] **Aşama 2-API:** `app/api.py` + `preprocess_image` + requirements + `tests/test_api.py` → `da6f019`.

## Aşama 3 — Eğitim + Colab  ← SIRADAKİ
- [ ] `src/foodlens/train.py`: AMP, label smoothing, grad clip, iki aşama (stage1 head-only 5 epoch LR 1e-3 → stage2 discriminative LR (backbone 1e-4/head 1e-3) + cosine 10-15 epoch), `--unfreeze-from-block N`.
- [ ] Checkpoint contract'ı train.py'de uygula (`model_state`, `classes`, `arch`, `num_classes`, `image_size`, `config`) → `api.py` ile uyumlu.
- [ ] `runs/<tarih>_<deney_ad>/` logging: `config.json`, `history.csv` (**"stage" sütunu**), `curves.png`, `best.pt`, `last.pt` + `runs/summary.csv`. Mevcut klasör üzerine yazma.
- [ ] `runs/` için `.gitignore` ayarı (checkpoint'ler commit edilmesin) — `.gitignore`'da `runs/*` zaten var, `.gitkeep` kontrolü.
- [ ] `notebooks/food101_colab.ipynb` — Full Food-101 indir (Colab) → `validate_food101.py` **zorunlu pre-flight** → eğitim → checkpoint'i indir.
- [ ] `train.py` için `tests/test_train.py` (küçük epoch / toy veri ile ağ akışı).

## Aşama 4 — UI + Açıklanabilirlik
- [ ] `src/foodlens/explain.py` — Grad-CAM, hedef katman `features[-1]` (= `features.8`), Top-1 sınıf.
- [ ] `app/` Gradio UI — top-5 + Grad-CAM heatmap; tek görsel yükleme (batch YOK).
- [ ] `app/api.py`'ye `/explain` endpoint'i (Grad-CAM) ekle.
- [ ] `tests/test_explain.py`.

## Entegrasyon / Teslim
- [ ] Arkadaşın private repoya (`myusufay/Food-Image-Classification`) branch push:
  - `app/api.py` mantığını onun `main.py`'ine taşı + `src/foodlens/`'i vendor'layarak self-contained yap.
  - **Push'u kullanıcı yapar** (contributor). AI `git remote add` + `git push` komutlarını verir, token istemez.
- [ ] README.md'yi güncelle (kurulum, Colab, API `uvicorn app.api:app`, Gradio).

## Nice-to-have (sonra)
- [ ] confusion matrix + en çok karışan sınıflar raporu.
- [ ] test-time augmentation (TTA) opsiyonu.
- [ ] Docker (FastAPI servisi için).
- [ ] Model benchmark tablosu (top-1/top-5, EfficientNet-B0 vs baseline).
- [ ] API'de model yükleme için cihaz seçimi (GPU varsa `cuda`) — şu an CPU.
