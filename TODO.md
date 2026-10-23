# TODO.md — FoodLens Yapılacaklar

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince [PROGRESS.md](PROGRESS.md) + gerekirse bu dosyayı güncelle.
> Durum işaretleri: `[ ]` açık · `[~]` devam · `[x]` bitmiş · `[BLOCKED]` engelli.

## Sıradaki (Acil)
- [x] **Aşama 3 commit:** `236bc8e`.
- [x] **Aşama 4 commit:** `e493b23`.
- [x] **`yemek_api.txt` kararı:** `.gitignore`'a eklendi.
- [ ] **Entegrasyon branch'i:** arkadaşın reposunda `feature/foodlens-integration` hazırla (aşağıdaki bölüm).

## Aşama 3 — Eğitim + Colab  ← TAMAMLANDI (`236bc8e`)
- [x] `src/foodlens/train.py`: AMP, label smoothing, grad clip, iki aşama, `--unfreeze-from-block N`.
- [x] Checkpoint contract train.py'de (`model_state`,`classes`,`arch`,`num_classes`,`image_size`,`config`) → api.py ile uyumlu **kanıtlandı**.
- [x] `runs/<tarih>_<ad>/` logging: `config.json`, `history.csv` (stage sütunu), `curves.png`, `best.pt`, `last.pt` + `runs/summary.csv`; mevcut klasör ezilmez.
- [x] `runs/` `.gitignore` kontrolü (`runs/*` zaten ignore).
- [x] `notebooks/food101_colab.ipynb` (10 hücre; indirme → **zorunlu validate** → eğitim → indirme).
- [x] `tests/test_train.py` (5 test: run klasoru, contract, epoch metrikleri, e2e).

## Aşama 4 — UI + Açıklanabilirlik  ← TAMAMLANDI (`e493b23`)
- [x] `src/foodlens/explain.py` — Grad-CAM, hedef katman `features[-1]` (= `features.8`), Top-1 sınıf; `features.8`/`features[-1]` yol desteği; parametre grad'ları kirletilmez.
- [x] `app/gradio_app.py` — top-5 + Grad-CAM heatmap; tek görsel yükleme (batch YOK); checkpoint yoksa bilgi mesajı.
- [x] `app/api.py` — `POST /explain` (base64 PNG + top-5; checkpoint yoksa 503).
- [x] `tests/test_explain.py` (8) + `/explain` testleri (2) + `test_gradio_app.py` (2) → **53 passed**.
- [x] E2E: smoke `best.pt` ile Gradio callback + `/explain` HTTP 200 doğrulandı.

## Entegrasyon / Teslim  ← SIRADAKİ
- [ ] **Hedef repo incele:** `myusufay/Food-Image-Classification` → mevcut `main.py`/`requirements.txt` yapısını oku, çakışmaları belirle.
- [ ] **Branch hazırla:** `feature/foodlens-integration` (veya benzeri) — `app/api.py` mantığını onun `main.py`'ine taşı; `src/foodlens/`'i vendor'la (self-contained, `pip install -e` gerektirmeden).
- [ ] **Uyumluluk:** arkadaşın mock cevap alanları (`tahmin_edilen_yemek`, `dogruluk_orani`, `durum`) korunur; checkpoint yoksa mock fallback aynen çalışır.
- [ ] **README güncelle:** kurulum, `uvicorn app.api:app`, `python app/gradio_app.py`, Colab akışı, checkpoint contract, `/predict` + `/explain` örnekleri.
- [ ] **Push'u kullanıcı yapar** (contributor). AI yalnızca `git remote add` + `git push -u origin <branch>` komutlarını verir; token istemez.
- [ ] **PR/teslim notu:** değişiklik özeti + test kanıtı (53 passed) + ekran görüntüsü/örnek JSON.

## Colab / Gerçek Model
- [ ] `notebooks/food101_colab.ipynb`'i Colab'da çalıştır: tam Food-101 indir → **zorunlu validate** → iki aşamalı eğitim → `best.pt` indir.
- [ ] `best.pt` + `history.csv` + `curves.png` + `summary.csv` dosyalarını indir; `best.pt`'yi `runs/<tarih>_food101_effb0/best.pt` altına koy (API otomatik bulur).
- [ ] **Gerçek modelle doğrula:** `/predict` top-5, `/explain` overlay, Gradio UI; beklenen test top-1'i `summary.csv` ile karşılaştır.
- [ ] (Opsiyonel) Colab süresi yetmezse `--unfreeze-from-block 5` veya epoch azalt; sonuçları `runs/summary.csv`'de karşılaştır.

## Nice-to-have (sonra)
- [ ] confusion matrix + en çok karışan sınıflar raporu.
- [ ] test-time augmentation (TTA) opsiyonu.
- [ ] Docker (FastAPI servisi için).
- [ ] Model benchmark tablosu (top-1/top-5, EfficientNet-B0 vs baseline).
- [ ] API'de model yükleme için cihaz seçimi (GPU varsa `cuda`) — şu an CPU.
