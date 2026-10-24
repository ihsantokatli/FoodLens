# TODO.md — FoodLens Yapılacaklar

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince [PROGRESS.md](PROGRESS.md) + gerekirse bu dosyayı güncelle.
> Durum işaretleri: `[ ]` açık · `[~]` devam · `[x]` bitmiş · `[BLOCKED]` engelli.

## Sıradaki (Acil)
- [x] **Aşama 3 commit:** `236bc8e`.
- [x] **Aşama 4 commit:** `e493b23`.
- [x] **Resume + Drive commit:** `0cba75f`.
- [x] **`yemek_api.txt` kararı:** `.gitignore`'a eklendi.
- [x] **Colab tam eğitim:** 5+12 epoch, test **%84,10 / %96,16** (2 oturum, resume ile).
- [x] **Entegrasyon paketi:** `feature/foodlens-integration` hazır (`c5b3b9c`).
- [ ] **Push + PR (kullanıcı):** arkadaşın reposuna dalı gönder, PR aç, teslim notunu yaz.

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

## Colab / Gerçek Model  ← TAMAMLANDI
- [x] `notebooks/food101_colab.ipynb` Colab'da çalıştırıldı: **4a) Drive** → tam Food-101 → **validate** → iki aşamalı eğitim.
- [x] 1. oturum: stage1 5 epoch + stage2 epoch 1-2 → **Runtime kesildi**, `last.pt` Drive'da kaldı.
- [x] 2. oturum: `RESUME_DIR` ile devam → stage2 epoch 3-12 + test. **Resume sorunsuz çalıştı.**
- [x] `best.pt` + `history.csv` + `curves.png` indirildi → `runs/20261024_food101_effb0/` (16,05 MB, 101 sınıf, 224 px, `train_state` yok).
- [x] **Gerçek modelle doğrulandı:** `/predict` → `kaynak: model`, `/explain` → `features.8` + base64 PNG, Gradio UI → top-5 + heatmap. Hepsi başarılı.
- [x] Sonuçlar: **test top-1 %84,10 · top-5 %96,16** (best val top-1 %79,91, epoch 11).

## Entegrasyon / Teslim  ← PAKET HAZIR, PUSH KULLANICIDA
- [x] **Hedef repo incele:** `myusufay/Food-Image-Classification` → `main.py` değil `yemek_API/yemek_API.py` (mock `/predict`) + VS `.pyproj`/`.slnx`.
- [x] **Branch hazırla:** `C:\Projects\Food-Image-Classification` klonu → `feature/foodlens-integration` (`c5b3b9c`) — `app/api.py` mantığı `yemek_API.py`'ye taşındı; `foodlens/` vendor'landı.
- [x] **Uyumluluk:** mock alanları (`tahmin_edilen_yemek`, `dogruluk_orani`, `durum`, `dosya_adi`) korundu; checkpoint yoksa mock fallback aynen çalışır; `/explain` model yoksa 503.
- [x] **README güncelle (arkadaşın reposu):** kurulum, model yerleştirme, `python yemek_API/yemek_API.py` + `uvicorn yemek_API.yemek_API:app`, checkpoint contract, gerçek `/predict` + `/explain` JSON, mock mod, sorun giderme.
- [x] **Doğrulama:** gerçek pizza fotoğrafı → `pizza %61,95`; Grad-CAM pizzanın üzerinde; 3 başlatma yolu + 415/400/503 test edildi.
- [ ] **Push'u kullanıcı yapar:** `git push -u origin feature/foodlens-integration` → PR. AI token istemez.
- [ ] **PR/teslim notu:** değişiklik özeti + test kanıtı (**57 passed**) + pizza örnek JSON + Grad-CAM görseli.

## Nice-to-have (sonra)
- [ ] confusion matrix + en çok karışan sınıflar raporu.
- [ ] test-time augmentation (TTA) opsiyonu.
- [ ] Docker (FastAPI servisi için).
- [ ] Model benchmark tablosu (top-1/top-5, EfficientNet-B0 vs baseline).
- [ ] API'de model yükleme için cihaz seçimi (GPU varsa `cuda`) — şu an CPU.
