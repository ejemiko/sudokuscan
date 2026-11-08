# SudokuScan

COMP 482 Bilgisayarlı Görü, Proje 1 · Ecem Yaşar (332501003) · MEF Üniversitesi

SudokuScan, telefonla çekilmiş bir fotoğraftaki basılı sudokuyu bulur, dört dış köşesini Harris
dedektörüyle tespit eder, ızgarayı üç geometrik modelle (A3, A4, H4) düzeltir, 81 hücreyi
Tesseract ile okur ve sonucu sudoku kurallarına göre kontrol eder.

## Aşamalar

| Aşama | Dosya | Ne yapar |
|---|---|---|
| 1 | `stage1_preprocess.py` | Küçültme (uzun kenar 1000 px), gri tonlama, Gauss ile gürültü azaltma, aydınlatma düzeltme (tahmin edilen arka plana bölme) |
| 2 | `stage2_corners.py` | Harris yanıt haritası, aday noktalar (göreli eşik + 7×7 maksimum olmayanları bastırma), bulmaca bölgesinin bulunması, dört dış köşenin seçimi |
| 3 | `stage3_rectify.py` | A3: üç köşeden (TL, TR, BL) afin; A4: dört köşeden en küçük kareler ile afin; H4: dört köşeden homografi. Hedef 450×450 px. A3 dördüncü köşe hatasını ve A4 RMS artığını raporlar |
| 4 | `stage4_residual.py`, `stage4_grid.py`, `stage4_ocr.py` | Afin ve homografi karşılaştırması (aşağıda) |
| 5 | `stage5_recognize.py` | 4 px iç kenar paylı 81 hücre, ızgara çizgisi kalıntılarının silinmesi, OCR'dan önce boş/dolu kararı, hücre hazırlama, Tesseract (`--psm 8`, izinli karakterler 1–9) |
| 6 | `stage6_validate.py` | Satır, sütun ve 3×3 kutularda tekrar eden rakamlar → `valid` / `invalid`; çakışan hücreler debug görüntüsünde işaretlenir |

Son dört köşe her zaman `cv2.cornerHarris` ile bulunan noktalardır. Konturlar yalnızca
bulmacayı içeren bölgeyi bulmak için kullanılır; köşeleri hiçbir zaman köşe koordinatı olarak
kullanılmaz.

## Kurulum

Python 3.10 veya üstü (macOS'ta Python 3.14 ile geliştirildi).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**OCR motoru.** `pytesseract` yalnızca bir arayüzdür; Tesseract programının ayrıca kurulması
gerekir (Tesseract 5.5 ile geliştirildi):

| Sistem | Komut |
|---|---|
| macOS (Homebrew) | `brew install tesseract` |
| Ubuntu / Debian | `sudo apt install tesseract-ocr` |
| Windows | https://github.com/UB-Mannheim/tesseract/wiki adresindeki kurulum dosyası, ardından `PATH`'e ekleyin |

Kontrol için: `tesseract --version`.

## Kullanım

```bash
python sudoku_scan.py --input photo.jpg --outdir out/          # tek görüntü
python sudoku_scan.py --batch images_dir/ --outdir results/    # değerlendirme modu
```

Her girdi görüntüsü `<ad>` için çıktı klasörüne şunlar yazılır:

| Dosya | İçerik |
|---|---|
| `<ad>_grid.txt` | 9 satır × 9 rakam, boş hücreler 0 (H4 düzeltmesinden) |
| `<ad>_result.json` | köşeler (orijinal piksellerde TL, TR, BR, BL), A3, A4, H4 matrisleri, `a3_corner_err_px`, `a4_rms_px`, ızgara, sonuç, çakışmalar, süre |
| `<ad>_harris.png` | bütün adaylar (yeşil) ve seçilen dört köşe (kırmızı) ile Harris yanıt haritası |
| `<ad>_rect_A3.png`, `_rect_A4.png`, `_rect_H4.png` | üç adet 450×450 düzeltilmiş görüntü |
| `<ad>_debug.png` | hücre kutuları, okunan rakamlar ve çakışan hücreler (kırmızı) ile H4 görüntüsü |

Program hatalı bir görüntüde çökmez: bulmaca bulunamazsa `"verdict": "not_found"` içeren bir
sonuç dosyası yazar.

## Değerlendirme

```bash
python sudoku_scan.py --batch data/ --outdir results/
python evaluate.py --results results/ --data data/
```

`evaluate.py`, ödevin 7. bölümündeki metrikleri (köşe hatası, hücre doğruluğu, boş hücre F1,
tam ızgara eşleşmesi, sağlamlık) toplamda ve C1–C5 koşullarına göre hesaplar.

Kişisel veri setindeki sonuçlar (15 fotoğraf):

| Köşe hatası | Hücre doğruluğu | Boş hücre F1 | Tam eşleşme | Sağlamlık | Fotoğraf başına süre |
|---|---|---|---|---|---|
| %0,93 | %97,4 | 0,983 | %86,7 | %100 | 1,3–2,8 s |

## Stage 4: afin ve homografi

Yukarıdaki batch komutundan sonra çalıştırın (betikler `results/` klasörünü okur):

```bash
python stage4_residual.py   # eğikliğe göre A3 dördüncü köşe hatası ve A4 RMS -> out/stage4/residual_vs_obliqueness.png
python stage4_grid.py       # ızgara çizgilerinin ideal konumlardan sapması (izdüşüm profilleri)
python stage4_ocr.py        # A3, A4 ve H4 düzeltmeleri üzerinde OCR hücre doğruluğu
```

Eğiklik, tespit edilen dörtgenin iki karşılıklı kenar oranının (uzun/kısa) büyüğüdür; dik
çekimde 1'dir.

## Testler ve deneyler

```bash
python tests/test_stage1.py   # ön işleme
python tests/test_stage2.py   # etiketlere göre köşe hatası
python tests/test_stage3.py   # A3 / A4 / H4 ve artıkları
python tests/test_stage5.py   # hücre doğruluğu, boş hücre F1, hazırlıklı/hazırlıksız, psm modları, karışıklıklar
python tests/test_stage6.py   # geçerlilik kontrolü
```

Parametre seçiminde kullanılan deneyler (`experiments/`):

| Betik | Amaç |
|---|---|
| `harris_test.py` | Harris eşiğine ve blockSize'a ilk bakış |
| `param_study.py` | bütün fotoğraflarda Harris parametre çalışması (blockSize, ksize, k) |
| `stage5_margin.py`, `stage5_margin_ocr.py` | hücre kenar payının seçimi |
| `stage5_empty.py` | boş hücre eşiğinin seçimi |

Debug görüntüleri `out/` klasörüne yazılır.

## Repo yapısı

```
sudoku_scan.py            komut satırı programı, bütün aşamaları çalıştırır
stage1_preprocess.py ... stage6_validate.py
stage4_residual.py, stage4_grid.py, stage4_ocr.py
evaluate.py               7. bölüm metrikleri
tests/                    her aşama için bir test betiği
experiments/              parametre çalışmaları
tools/annotate.py         köşe etiketleme aracı
tools/fill_metadata.py    ızgaraları ve koşulları etiket dosyalarına yazar
data/                     kişisel veri seti: fotoğraflar + etiketler (Ek A formatı)
```

## Yapay zekâ kullanımı

Claude (Anthropic), `AI_LOG.md` dosyasında ve raporun Ek C'sinde anlatıldığı şekilde kullanıldı.
Köşe seçim mantığını (`stage2_corners.py` içindeki `find_candidates`, `find_puzzle_region`,
`select_corners`) ve Stage 4 karşılaştırma analizini (`stage4_residual.py`, `stage4_grid.py`,
`stage4_ocr.py`) yapay zekâ kodu kullanmadan ben yazdım.