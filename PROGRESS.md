# PROGRESS.md — FoodLens Proje Durumu

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince bu dosyayı güncelle.
> Kural/karar kaynağı → [RULES.md](RULES.md). Yapılacaklar → [TODO.md](TODO.md). Çözülen hatalar → [MEMORY.md](MEMORY.md).

## Genel Durum
- **Aşamalar:** 0 ✓ · 1 ✓ · 2 (model) ✓ · 2-API ✓ · **sıradaki: 3 (eğitim + Colab)** · 4 bekliyor.
- **Testler:** **36 passed** (`test_data.py` 19 + `test_model.py` 11 + `test_api.py` 6).
- **Son commit:** `da6f019` (Stage 2-API). Toplam 6 commit.
- **Eğitim:** Henüz başlamadı (Colab, Aşama 3). `best.pt` yok → API şu an mock modda.
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

## Commit Edilmemiş
- (yok — çalışma ağacı temiz. `yemek_api.txt` `.gitignore`'da, referans olarak yerelde duruyor.)

## Aşama Detayı

### ✓ Aşama 0 — Skeleton
Paket yapısı (`src/foodlens/`, `app/`, `scripts/`, `tests/`, `configs/`, `notebooks/`), `pyproject.toml`, `requirements.txt`. `checkpoints/` kaldırıldı (çıktılar `runs/`).

### ✓ Aşama 1 — Veri Katmanı
- `config.py`: `load_config`, `resolve_path`.
- `data.py`: `build_transforms`, `stratified_split`, `load_or_create_val_indices`, `Food101Subset`, `build_from_config`, `collect_pool`, `per_class_distribution`, `format_distribution_report`.
- `scripts/`: `make_subset.py`, `make_toy_food101.py`, `explore_data.py`.
- Verified: smoke subset 10 sınıf, 800/200, `test_split_touched: false`; grid → `data/_preview/smoke_subset_grid.png`.

### ✓ Aşama 2 Pre-flight — Format eşitliği + Colab validator
- `data.py`: `read_food101_split` (.txt öncelikli, .json fallback), `read_food101_classes`, `samples_from_meta` (kanonik sıralı).
- `make_toy_food101.py`: gerçek arşivle birebir `classes.txt`/`labels.txt`/`train.txt`/`test.txt` + `.json`.
- `src/foodlens/validate.py` + `scripts/validate_food101.py`: Colab pre-flight (101/75750/25250/7575/68175); tutmazsa `exit 1` → eğitim başlamaz.

### ✓ Aşama 2 — Model (`a495bf0`)
- `src/foodlens/model.py`: `build_model(101 head)`, `freeze_backbone`, `unfreeze_all`, `unfreeze_from_block`, `discriminative_param_groups`, `trainable_frozen_summary`, `format_summary`.
- Verified: head `Dropout(0.2)+Linear(1280,101)`; (4,3,224,224)→(4,101); freeze→0.13M trainable; unfreeze_all→4.14M; gruplar backbone 211 param @1e-4 + head 2 param @1e-3.
- torchvision 0.29 `features` = 9 modül (0..8); `features[8]`=conv head=`features[-1]`.

### ✓ Aşama 2-API — FastAPI (`da6f019`)
- `data.py`'ye `preprocess_image(pil_img, image_size=224)` — eval transform (augment'siz), (C,H,W) döner.
- `app/api.py`: `GET /`, `POST /predict` (arkadaşının iskelet alanları korunur). Checkpoint `env FOODLENS_CHECKPOINT` → `runs/*/best.pt` (en yeni) ile otomatik bulunur; **yoksa mock mod** (`Pizza/94.5`).
- Checkpoint contract uygulandı: `{"model_state", "classes", "arch", "num_classes", "image_size", "config"}`.
- `app/__init__.py` eklendi (`uvicorn app.api:app` + test import güvenilirliği).
- `requirements.txt`: `fastapi`, `uvicorn[standard]`, `python-multipart`, `httpx` (TestClient).
- Verified: mock & gerçek (top-5) & `415` yanıtları canlı gösterildi; `tests/test_api.py` 6 test.

## Sırada Ne Var (özet)
1. **Aşama 3:** `src/foodlens/train.py` (AMP, label smoothing, grad clip, iki aşama, `--unfreeze-from-block`, `runs/` logging "stage" sütunu, checkpoint contract) + `notebooks/food101_colab.ipynb` (tam indirme + zorunlu `validate_food101.py` pre-flight + eğitim + checkpoint indirme). `runs/` için `.gitignore`.
2. **Aşama 4:** `src/foodlens/explain.py` (Grad-CAM, `features[-1]`) + `app/` Gradio (top-5 + heatmap) → `api.py`'ye `/explain`.
3. **Entegrasyon:** arkadaşın private repoya branch push (kullanıcı yapar; AI komutları verir) + README güncelle.
