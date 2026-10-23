# TODO.md — FoodLens Yapılacaklar

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince [PROGRESS.md](PROGRESS.md) + gerekirse bu dosyayı güncelle.
> Durum işaretleri: `[ ]` açık · `[~]` devam · `[x]` bitmiş · `[BLOCKED]` engelli.

## Sıradaki (Acil)
- [x] **Aşama 3 commit:** `236bc8e`.
- [x] **Aşama 4 commit:** `e493b23`.
- [x] **Resume + Drive commit:** `0cba75f`.
- [x] **`yemek_api.txt` kararı:** `.gitignore`'a eklendi.
- [ ] **Colab tam eğitim:** Drive hücresi + 5+12 epoch (aşağıdaki bölüm).

## Aşama 3 — Eğitim + Colab  ← TAMAMLANDI (`236bc8e`)
- [x] `src/foodlens/train.py`: AMP, label smoothing, grad clip, iki aşama, `--unfreeze-from-block N`.
- [x] Checkpoint contract train.py'de (`model_state`,`classes`,`arch`,`num_classes`,`image_size`,`config`) → api.py ile uyumlu **kanıtlandı**.
- [x] `runs/<tarih>_<ad>/` logging: `config.json`, `history.csv` (stage sütunu), `curves.png`, `best.pt`, `last.pt` + `runs/summary.csv`; mevcut klasör ezilmez.
- [x] `runs/` `.gitignore` kontrolü (`runs/*` zaten ignore).
- [x] `notebooks/food101_colab.ipynb` (12 hücre; indirme → **zorunlu validate** → Drive yedeği → eğitim/resume → indirme).
- [x] `tests/test_train.py` (5 test: run klasoru, contract, epoch metrikleri, e2e).

## Aşama 4 — UI + Açıklanabilirlik  ← TAMAMLANDI (`e493b23`)
- [x] `src/foodlens/explain.py` — Grad-CAM, hedef katman `features[-1]` (= `features.8`), Top-1 sınıf; `features.8`/`features[-1]` yol desteği; parametre grad'ları kirletilmez.
- [x] `app/gradio_app.py` — top-5 + Grad-CAM heatmap; tek görsel yükleme (batch YOK); checkpoint yoksa bilgi mesajı.
- [x] `app/api.py` — `POST /explain` (base64 PNG + top-5; checkpoint yoksa 503).
- [x] `tests/test_explain.py` (8) + `/explain` testleri (2) + `test_gradio_app.py` (2) → **53 passed**.
- [x] E2E: smoke `best.pt` ile Gradio callback + `/explain` HTTP 200 doğrulandı.

## Aşama 4.5 — Resume + Drive  ← TAMAMLANDI (`0cba75f`)
- [x] `train.py --resume <last.pt|run_dir>`: model + optimizer + scheduler + scaler + stage/epoch + best + history.
- [x] `train.py --runs-dir`: runs dizinini ezer (Colab Drive yolu).
- [x] `last.pt`'de `train_state`; `best.pt` contract **değişmedi** (6 anahtar).
- [x] Resume'da tamamlanmış stage atlanır; `history.csv` yeniden yazılır, `summary.csv` upsert.
- [x] Notebook: Drive hücresi + `RESUME_DIR` değişkeni (12 hücre).
- [x] Testler: `test_train.py` 5 → 9 (mid-stage resume, stage geçişi, `--runs-dir`) → **57 passed**.
- [x] Gerçek smoke: stage1 → resume → stage2 doğrulandı.

## Colab / Gerçek Model  ← SIRADAKİ
- [ ] `notebooks/food101_colab.ipynb`'i Colab'da çalıştır: **4a) Drive** hücresi → tam Food-101 indir → **zorunlu validate** → iki aşamalı eğitim.
- [ ] Süre/kopma olursa: Drive'daki run klasörünü `RESUME_DIR`'e yaz → aynı hücreyi tekrar çalıştır (kaldığı stage/epoch+1).
- [ ] `best.pt` + `history.csv` + `curves.png` + `summary.csv` dosyalarını indir; `best.pt`'yi `runs/<tarih>_food101_effb0/best.pt` altına koy (API otomatik bulur).
- [ ] **Gerçek modelle doğrula:** `/predict` top-5, `/explain` overlay, Gradio UI; beklenen test top-1'i `summary.csv` ile karşılaştır.
- [ ] (Opsiyonel) Colab süresi yetmezse `--unfreeze-from-block 5` veya epoch azalt; sonuçları `runs/summary.csv`'de karşılaştır.

## Entegrasyon / Teslim
- [ ] **Hedef repo incele:** `myusufay/Food-Image-Classification` → mevcut `main.py`/`requirements.txt` yapısını oku, çakışmaları belirle.
- [ ] **Branch hazırla:** `feature/foodlens-integration` (veya benzeri) — `app/api.py` mantığını onun `main.py`'ine taşı; `src/foodlens/`'i vendor'la (self-contained, `pip install -e` gerektirmeden).
- [ ] **Uyumluluk:** arkadaşın mock cevap alanları (`tahmin_edilen_yemek`, `dogruluk_orani`, `durum`) korunur; checkpoint yoksa mock fallback aynen çalışır.
- [ ] **README güncelle:** kurulum, `uvicorn app.api:app`, `python app/gradio_app.py`, Colab akışı, checkpoint contract, `/predict` + `/explain` örnekleri.
- [ ] **Push'u kullanıcı yapar** (contributor). AI yalnızca `git remote add` + `git push -u origin <branch>` komutlarını verir; token istemez.
- [ ] **PR/teslim notu:** değişiklik özeti + test kanıtı (**57 passed**) + ekran görüntüsü/örnek JSON.

## Nice-to-have (sonra)
- [ ] confusion matrix + en çok karışan sınıflar raporu.
- [ ] test-time augmentation (TTA) opsiyonu.
- [ ] Docker (FastAPI servisi için).
- [ ] Model benchmark tablosu (top-1/top-5, EfficientNet-B0 vs baseline).
- [ ] API'de model yükleme için cihaz seçimi (GPU varsa `cuda`) — şu an CPU.
