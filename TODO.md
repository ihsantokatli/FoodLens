# TODO.md — FoodLens Yapılacaklar

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince [PROGRESS.md](PROGRESS.md) + gerekirse bu dosyayı güncelle.
> Durum işaretleri: `[ ]` açık · `[~]` devam · `[x]` bitmiş · `[BLOCKED]` engelli.

## Sıradaki (Acil)
- [x] **Aşama 2 commit:** `a495bf0`.
- [x] **`yemek_api.txt` kararı:** `.gitignore`'a eklendi.
- [x] **Aşama 2-API:** `da6f019`.

## Aşama 3 — Eğitim + Colab  ← TAMAMLANDI (`236bc8e`)
- [x] `src/foodlens/train.py`: AMP, label smoothing, grad clip, iki aşama, `--unfreeze-from-block N`.
- [x] Checkpoint contract train.py'de (`model_state`,`classes`,`arch`,`num_classes`,`image_size`,`config`) → api.py ile uyumlu **kanıtlandı**.
- [x] `runs/<tarih>_<ad>/` logging: `config.json`, `history.csv` (stage sütunu), `curves.png`, `best.pt`, `last.pt` + `runs/summary.csv`; mevcut klasör ezilmez.
- [x] `runs/` `.gitignore` kontrolü (`runs/*` zaten ignore).
- [x] `notebooks/food101_colab.ipynb` (10 hücre; indirme → **zorunlu validate** → eğitim → indirme).
- [x] `tests/test_train.py` (5 test: run klasoru, contract, epoch metrikleri, e2e).

## Aşama 4 — UI + Açıklanabilirlik  ← SIRADAKİ
- [ ] `src/foodlens/explain.py` — Grad-CAM, hedef katman `features[-1]` (= `features.8`), Top-1 sınıf.
- [ ] `app/` Gradio UI — top-5 + Grad-CAM heatmap; tek görsel yükleme (batch YOK).
- [ ] `app/api.py`'ye `/explain` endpoint'i (Grad-CAM) ekle.
- [ ] `tests/test_explain.py`.

## Colab / Gerçek Model
- [ ] `notebooks/food101_colab.ipynb`'i Colab'da çalıştır: tam Food-101 indir → validate → eğitim → `best.pt` indir.
- [ ] Elde edilen `best.pt`'yi Gradio/FastAPI için `runs/` altında sakla (API otomatik bulur).

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
