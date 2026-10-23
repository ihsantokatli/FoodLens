# PROGRESS.md — FoodLens Proje Durumu

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince bu dosyayı güncelle.
> Kural/karar kaynağı → [RULES.md](RULES.md). Yapılacaklar → [TODO.md](TODO.md). Çözülen hatalar → [MEMORY.md](MEMORY.md).

## Genel Durum
- **Aşamalar:** 0 ✓ · 1 ✓ · 2 (model) ✓ · 2-API ✓ · 3 (egitim + Colab) ✓ · 4 (Gradio + Grad-CAM + /explain) ✓ · **4.5 (resume + Drive) ✓** · **sıradaki: Colab tam eğitim → entegrasyon**.
- **Testler:** **57 passed** (`test_data.py` 19 + `test_model.py` 11 + `test_api.py` 8 + `test_train.py` 9 + `test_explain.py` 8 + `test_gradio_app.py` 2).
- **Son commit:** `0cba75f` (resume desteği). Stage 4: `e493b23`. Toplam **14 commit**.
- **Eğitim:** Yerel smoke dogrulandi (runs/ uretimi). Tam egitim Colab'da yapilacak (defter hazir; Drive + resume akisi eklendi).
- **Uzak repo:** `https://github.com/ihsantokatli/FoodLens` (**public**, `master`) — Colab token'sız klonlar.
- **Çalışma ağacı:** temiz.

## Commit'ler
| Hash | Mesaj | Durum |
|---|---|---|
| `9840a5d` | Stage 0: project skeleton | ✓ |
| `2915347` | Stage 1: data layer (transforms, stratified val split, subset, reports) | ✓ |
| `cb8af66` | Stage 2 pre-flight: archive format parity + Colab validator | ✓ |
| `a495bf0` | Stage 2: model (EfficientNet-B0 + 101 head, freeze/unfreeze, discriminative LR) | ✓ |
| `19210e6` | Add AI memory system (RULES/PROGRESS/TODO/MEMORY) + ignore temp notes | ✓ |
| `da6f019` | Stage 2-API: FastAPI inference service (checkpoint-ready, mock fallback) | ✓ |
| `9062a25` | Update memory: Stage 2 + Stage 2-API done, focus moves to Stage 3 | ✓ |
| `236bc8e` | Stage 3: training loop (two-stage fine-tune, runs logging, checkpoint contract) + Colab notebook | ✓ |
| `82f6df4` | Update memory: Stage 3 done (train + Colab), log top-k gather bug, focus moves to Stage 4 | ✓ |
| `c69b1b0` | chore: finish pending updates (RULES note + sync PROGRESS commit log) | ✓ |
| `e493b23` | Stage 4: Grad-CAM explainability + Gradio UI + /explain endpoint | ✓ |
| `fe62cd4` | docs: record GitHub remote (ihsantokatli/FoodLens, private, master) | ✓ |
| `0cba75f` | Add resume support (--resume, --runs-dir) + Colab Drive/resume flow | ✓ |

## Commit Edilmemiş
- (yok — çalışma ağacı temiz. `yemek_api.txt` `.gitignore`'da.)

## Aşama Detayı

### ✓ Aşama 0 — Skeleton
Paket yapısı (`src/foodlens/`, `app/`, `scripts/`, `tests/`, `configs/`, `notebooks/`), `pyproject.toml`, `requirements.txt`. `checkpoints/` kaldırıldı (çıktılar `runs/`).

### ✓ Aşama 1 — Veri Katmanı
`config.py` (`load_config`,`resolve_path`), `data.py` (`build_transforms`, `stratified_split`, `load_or_create_val_indices`, `ImageListDataset`, `build_dataloaders`, `build_from_config`, `collect_pool`, raporlar), `scripts/` (`make_subset`, `make_toy_food101`, `explore_data`).

### ✓ Aşama 2 Pre-flight — Format eşitliği + Colab validator
`read_food101_split` (.txt öncelikli), `read_food101_classes`, `samples_from_meta` (kanonik), toy üreteç gerçek arşiv formatı, `validate.py` + `scripts/validate_food101.py` (101/75750/25250/7575/68175, tutmazsa `exit 1`).

### ✓ Aşama 2 — Model (`a495bf0`)
`model.py`: `build_model`, `freeze_backbone`, `unfreeze_all`, `unfreeze_from_block`, `discriminative_param_groups`, `trainable_frozen_summary`. torchvision 0.29 `features`=9 modül; `features[8]`=`features[-1]`.

### ✓ Aşama 2-API — FastAPI (`da6f019`)
`data.py` + `preprocess_image`; `app/api.py` (`GET /`, `POST /predict`, checkpoint auto-find → mock fallback, contract uygulandı); `app/__init__.py`; requirements (fastapi/uvicorn/python-multipart/httpx).

### ✓ Aşama 3 — Eğitim + Colab (`236bc8e`)
- `src/foodlens/train.py`: iki aşamalı fine-tune (stage1 head-only sabit LR → stage2 discriminative LR + cosine), AMP (cuda), label smoothing, grad clip, `--unfreeze-from-block`, `--limit-batches` (yerel smoke).
- **`runs/<tarih>_<ad>/` (ASLA ezmez, `_1`,`_2` ekler):** `config.json`, `history.csv` (**stage sutunu**), `curves.png`, `best.pt`, `last.pt` + `runs/summary.csv`.
- **Checkpoint contract** train.py ↔ api.py uyumlu kanıtlandı (`best.pt` API'ye bağlanıp gerçek sınıf adlarıyla top-5 döndü).
- `scripts/train.py` (ince CLI), `notebooks/food101_colab.ipynb` (başlangıçta 10 hücre; Aşama 4.5'te Drive/resume ile 12'ye çıktı: kurulum → indir → **zorunlu validate** → Drive → eğitim → indirme).
- Verified: smoke run (stage1 0.01M → stage2 4.02M trainable), tüm çıktılar üretildi; 5 yeni test.

### ✓ Aşama 4 — UI + Açıklanabilirlik (`e493b23`)
- `src/foodlens/explain.py`: **Grad-CAM** — hedef `features[-1]` (= `features.8`); `features.8` ve `features[-1]` yolları; `torch.autograd.grad(score, act)` (parametre `.grad`'ları kirletmez); göreli normalizasyon (mutlak epsilon küçük değerlerde bozuyordu); overlay orijinal görsel boyutunu korur.
- `app/gradio_app.py`: tek görsel → top-5 + Grad-CAM overlay (batch YOK); `gradio.share`/`topk`/`gradcam_layer` config'ten; checkpoint yoksa bilgi mesajı (çökmez).
- `app/api.py`: **`POST /explain`** — top-5 + base64 PNG overlay; checkpoint yoksa **503**; dosya doğrulama `/predict` ile ortak `_read_upload` yardımcısında.
- Testler: `test_explain.py` (8), `test_api.py` (+2), `test_gradio_app.py` (2) → **53 passed**.
- E2E doğrulandı: smoke `best.pt` ile Gradio callback (RGB 320×240 overlay) + `/explain` HTTP 200 (`features.8`, base64 PNG açılabilir).

### ✓ Aşama 4.5 — Resume + Drive (`0cba75f`)
- `train.py`: **`--resume <last.pt|run_dir>`** — model + optimizer + scheduler + AMP scaler + `stage`/`epoch` + `best` + `history` `train_state`'ten yüklenir; tamamlanmış stage atlanır, aynı run klasörüne devam edilir; `history.csv` yeniden yazılır (satır tekrarı yok), `summary.csv` **upsert** edilir.
- `best.pt` sözleşmesi **değişmedi** (6 anahtar); `train_state` yalnızca `last.pt`'de.
- `--runs-dir`: runs dizinini ezer (Colab'da Drive: `/content/drive/MyDrive/FoodLens/runs`) → oturum kapansa da `last.pt` kalıcı.
- Notebook: **4a) Drive** hücresi + `RESUME_DIR` değişkeni; indirme yolu Drive/yerel otomatik.
- Testler: `test_train.py` 5 → **9** (+train_state sözleşmesi, mid-stage resume, stage1→stage2 geçişi, `--runs-dir`) → toplam **57 passed**.
- Gerçek smoke doğrulaması: stage1 (1 epoch) → `--resume` → stage2 epoch 1; `history.csv` 2 satır, `last.pt`'de optimizer+scheduler, `best` korundu.

## Sırada Ne Var (özet)
1. **Colab tam eğitim:** Drive hücresi → Food-101 indir → validate → 5+12 epoch; süre/kopma olursa `RESUME_DIR` ile devam → `best.pt` indir.
2. **Entegrasyon:** arkadaşın reposuna branch (kod taşıma + vendor) — **push'u kullanıcı yapar**; README güncellemesi bu adımda.
3. **Teslim doğrulaması:** gerçek 101-sınıf `best.pt` ile `/predict`, `/explain` ve Gradio'yu duman testinden geçir.
