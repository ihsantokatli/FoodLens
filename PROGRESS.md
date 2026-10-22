# PROGRESS.md — FoodLens Proje Durumu

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince bu dosyayı güncelle.
> Kural/karar kaynağı → [RULES.md](RULES.md). Yapılacaklar → [TODO.md](TODO.md). Çözülen hatalar → [MEMORY.md](MEMORY.md).

## Genel Durum
- **Aşamalar:** 0 ✓ · 1 ✓ · 2 (model) ✓ · 2-API ✓ · **3 (egitim + Colab) ✓** · **sıradaki: 4 (Gradio + Grad-CAM)** · entegrasyon bekliyor.
- **Testler:** **41 passed** (`test_data.py` 19 + `test_model.py` 11 + `test_api.py` 6 + `test_train.py` 5).
- **Son commit:** `236bc8e` (Stage 3). Toplam 8 commit.
- **Eğitim:** Yerel smoke dogrulandi (runs/ uretimi). Tam egitim Colab'da yapilacak (defter hazir).
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
- `scripts/train.py` (ince CLI), `notebooks/food101_colab.ipynb` (10 hücre: kurulum → indir → **zorunlu validate** → eğitim → indirme).
- Verified: smoke run (stage1 0.01M → stage2 4.02M trainable), tüm çıktılar üretildi; 5 yeni test.

## Sırada Ne Var (özet)
1. **Aşama 4:** `src/foodlens/explain.py` (Grad-CAM, `features[-1]`) + `app/` Gradio (top-5 + heatmap, batch yok) → `api.py`'ye `/explain`.
2. **Entegrasyon:** arkadaşın private repoya branch push (kullanıcı yapar) + README güncelle.
3. **Colab:** defteri çalıştırıp tam modeli eğit, `best.pt`'yi indir (Gradio/API gerçekten kullanır).
