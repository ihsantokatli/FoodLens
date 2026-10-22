# PROGRESS.md — FoodLens Proje Durumu

> AI Hafıza Sistemi. Her yeni görevde ÖNCE oku; iş bitince bu dosyayı güncelle.
> Kural/karar kaynağı → [RULES.md](RULES.md). Yapılacaklar → [TODO.md](TODO.md). Çözülen hatalar → [MEMORY.md](MEMORY.md).

## Genel Durum
- **Aşamalar:** 0 ✓ · 1 ✓ · 2 (model) **kod/test hazır, commit BEKLİYOR** · 2-API/3/4 bekliyor.
- **Testler:** **30 passed** (`tests/test_data.py` 19 + `tests/test_model.py` 11).
- **Son commit:** `cb8af66` (Stage 2 pre-flight). Toplam 3 commit.
- **Eğitim:** Henüz başlamadı (Colab, Aşama 3).

## Commit'ler
| Hash | Mesaj | Durum |
|---|---|---|
| `9840a5d` | Stage 0: project skeleton | ✓ |
| `2915347` | Stage 1: data layer (transforms, stratified val split, subset, reports) | ✓ |
| `cb8af66` | Stage 2 pre-flight: archive format parity + Colab validator | ✓ |

## Commit Edilmemiş (Çalışma ağacı)
- `src/foodlens/model.py` — Aşama 2 model katmanı (test edildi, commit onayı bekliyor).
- `tests/test_model.py` — 11 model testi.
- `yemek_api.txt` — arkadaşının FastAPI iskeleti (referans; commit edilmedi).

## Aşama Detayı

### ✓ Aşama 0 — Skeleton
Paket yapısı (`src/foodlens/`, `app/`, `scripts/`, `tests/`, `configs/`, `notebooks/`), `pyproject.toml`, `requirements.txt`. `checkpoints/` kaldırıldı (çıktılar `runs/`).

### ✓ Aşama 1 — Veri Katmanı
- `src/foodlens/config.py`: `load_config`, `resolve_path`.
- `src/foodlens/data.py`: `build_transforms`, `stratified_split`, `load_or_create_val_indices`, `Food101Subset`, `build_from_config`, `collect_pool`, `per_class_distribution`, `format_distribution_report`.
- `scripts/`: `make_subset.py`, `make_toy_food101.py`, `explore_data.py`.
- Verified: smoke subset 10 sınıf, 800/200, `test_split_touched: false`; grid → `data/_preview/smoke_subset_grid.png`.

### ✓ Aşama 2 Pre-flight — Format eşitliği + Colab validator
- `data.py`: `read_food101_split` (.txt öncelikli, .json fallback), `read_food101_classes`, `samples_from_meta` (kanonik sıralı).
- `make_toy_food101.py`: gerçek arşivle birebir `classes.txt`/`labels.txt`/`train.txt`/`test.txt` + `.json`.
- `src/foodlens/validate.py` + `scripts/validate_food101.py`: Colab pre-flight (101/75750/25250/7575/68175); tutmazsa `exit 1` → eğitim başlamaz.

### ⏳ Aşama 2 — Model (KOD HAZIR, COMMIT BEKLİYOR)
- `src/foodlens/model.py`: `build_model(101 head)`, `freeze_backbone`, `unfreeze_all`, `unfreeze_from_block`, `discriminative_param_groups`, `trainable_frozen_summary`, `format_summary`.
- Verified çıktı: head `Dropout(0.2)+Linear(1280,101)`; (4,3,224,224)→(4,101); freeze→0.13M trainable; unfreeze_all→4.14M; discriminative gruplar backbone 211 param @1e-4 + head 2 param @1e-3.
- torchvision 0.29 `features` = 9 modül (0..8); `features[8]`=conv head=`features[-1]`.

## Sırada Ne Var (özet)
1. Aşama 2 `model.py` commit (onay bekliyor).
2. Aşama 2-API: `app/api.py` FastAPI (checkpoint'e hazır, mock fallback) + `foodlens.data.preprocess_image` + requirements (fastapi/uvicorn/python-multipart).
3. Aşama 3: `train.py` (optimizer/scheduler/AMP, iki aşama, `--unfreeze-from-block`, `runs/` logging, checkpoint contract) + Colab notebook.
4. Aşama 4: `app/` Gradio (top-5 + Grad-CAM) + `src/foodlens/explain.py` → `/explain`.
5. Arkadaşın private repoya (`myusufay/Food-Image-Classification`) branch push (kullanıcı yapar).
