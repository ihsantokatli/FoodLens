# MEMORY.md — Çözülen Zorlu Hatalar Arşivi

> AI Hafıza Sistemi. Bir hata yeniden çıkarsa ÖNCE buraya bak. Yori bir çözüm üretince buraya ekle.
> Format: **Belirti → Kök neden → Çözüm → Ders**.

---

### 1) `random.Random` tuple seed kabul etmiyor
- **Belirti:** `random.Random((42, "apple_pie"))` → `TypeError`. Per-class deterministik split isteniyordu.
- **Kök neden:** `random.Random` seed'i hash'lenebilir tek bir değer ister; tuple verilirse patlar.
- **Çözüm:** String birleştirme `random.Random(f"{seed}:{cls}")`. (`data.py:stratified_split`, `make_subset.py`.)
- **Ders:** Per-class determinizm için seed'i string'e kanalize et. Bu, val split'inin tekrar üretilebilirliğini garantiler.

---

### 2) Food-101 meta `.txt` vs `.json` SIRA farkı → val_indices tutarsızlığı riski
- **Belirti:** Gerçek arşivde `train.txt` sınıfa göre **sıralı** (`apple_pie` ilk), `train.json` anahtarları **insertion order** (`churros` ilk). Farklı kaynaktan okununca örnek sırası değişebilir → `val_indices.json` indeksleri başka görsellere işaret edebilir.
- **Kök neden:** İki meta biçimi aynı bilgiyi farklı sırayla tutuyor.
- **Çözüm:** `samples_from_meta` **kanonik sıralama** uygular: sınıflar sıralı + her sınıf içinde rel-path sıralı. `.txt` ve `.json` artık **birebir aynı** örnek listesini verir. `data.py` `.txt` öncelikli okur.
- **Ders:** Dış veri kaynaklarını normalize et; indeks/tabanlı kalıcılık formatından bağımsız olmalı. Testle kanıtla (`test_txt_and_json_meta_produce_identical_samples`).

---

### 3) `torchvision.ops.Conv2dNormActivation` — `torch.nn`'de değil
- **Belirti:** `AttributeError: module 'torch.nn' has no attribute 'Conv2dNormActivation'`.
- **Kök neden:** Bu sınıf `torchvision.ops`'tadır, `nn`'de değil.
- **Çözüm:** `from torchvision.ops import Conv2dNormActivation`.
- **Ders:** torchvision özel katmanları için doğru namespace'i kullan; `nn` varsayma.

---

### 4) EfficientNet `features` sandığından AZ modül içeriyor (9, 15 değil)
- **Belirti:** `gradcam_layer: features.8` "geçersiz" sanıldı; `len(model.features)` = **9** çıktı (0..8).
- **Kök neden:** torchvision 0.29 EfficientNet-B0 `features` = stem(0) + 7 MBConv(1..7) + conv head(8). `features[8]` = `features[-1]` = son konvolüsyon → config'teki değer **doğru**.
- **Çözüm:** Yapı `print`'lenerek doğrulandı; `unfreeze_from_block(model,n)` bu indekslere göre tasarlandı (n=-1 tümü, n=5 → features[5..8]+head).
- **Ders:** Sürüm mimariyi değiştirebilir — model katman indekslerini **her zaman runtime'da doğrula**, docstring/ config yorumundan varsayma.

---

### 5) Python `round()` banker yuvarlaması → per-class val sayısı şaşırıyor
- **Belirti:** 105 train × 0.10 = 10.5 ama `round(10.5)` = **10** (çift'e yuvarlar), 11 değil.
- **Kök neden:** Python 3 `round()` kara (banker's) yuvarlama kullanır.
- **Çözüm:** Beklentiyi `data.py` ile **aynı formülle** hesapla: `max(1, round(n * val_ratio))`. Testte bu formulü kullan, elle 11 yazma.
- **Ders:** Yuvarlama içeren sayısal beklentileri asla elle hesaplama; üretim kodundaki formülü referans al.

---

### 6) Colab'da `pip install torch torchvision` GPU'yu bozuyor
- **Belirti:** Colab'da ayrı pip kurulumu, GPU ile eşleşmeyen CPU sürümü çekiyor → CUDA hatası.
- **Kök neden:** Colab torch/torchvision'u önceden GPU eşleşecek şekilde kurar.
- **Çözüm:** `requirements.txt` notu: Colab'da bu ikisini **atla**, sadece diğer paketleri kur. `notebooks/` bunu uygulayacak.
- **Ders:** Colab'da mevcut ortamı ezme; GPU paketlerini ortama bırak.

---

### 7) `make_subset.py --force` diye bir argüman yok
- **Belirti:** `error: unrecognized arguments: --force`.
- **Kök neden:** Script `--force` tanımlamıyor (çıkış klasörüne yazıyor).
- **Çözüm:** `--force` yok; mevcut klasörü `Remove-Item -Recurse` ile temizleyip yeniden üret, ya da `--help`'e bak.
- **Ders:** Script argümanlarını varsayma, `--help` ile doğrula.

---

### 8) Top-k doğruluğunda `gather` yanlış kullanımı → "index out of bounds"
- **Belirti:** `RuntimeError: index 5 is out of bounds for dimension 1 with size 5` (10 sınıflı veride top-5 hesaplarken). Kod: `topk.gather(1, yb.view(-1,1)).any(1)`.
- **Kök neden:** `logits.topk(k).indices` → `(bs, k)` **sınıf indeksleri** tutar (sıra değil). `gather(1, yb)` ifadesi bunları **sutun numarası** sanıp gerçek etiketi (`yb`, 0..9) sutun indeksi olarak kullanıyor; ama `topk`'un yalnızca `k=5` sutunu var → 5+ index patlıyor.
- **Çözüm:** "Gerçek etiket top-k içinde mi?" `topk.eq(yb.view(-1,1)).any(dim=1)` ile sorulur. Doğru kod:
  ```python
  topk = logits.topk(k, dim=1).indices      # (bs, k) SINIF indeksleri
  c1 += (topk[:, 0] == yb).sum().item()
  c5 += topk.eq(yb.view(-1, 1)).any(dim=1).sum().item()   # Top-k dogruluk
  ```
- **Ders:** `gather` yalnızca **aynı boyuttaki indeks-tensorunu** seçmek içindir; top-k accuracy'de indeksler sınıf numarası olduğu için `eq + any` kullan. `topk` sütunları = rank değil, sınıf indeksi.
