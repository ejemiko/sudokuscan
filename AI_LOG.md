# Yapay zekâ kullanım kaydı (AI_LOG)

Araç: **Claude** (Anthropic), claude.ai üzerinden öğretici ve kodlama yardımcısı olarak
kullanıldı. Ödev (Section 11), köşe seçim mantığının (Stage 2) ve karşılaştırma analizinin
(Stage 4) yapay zekâya yazdırılmasına izin vermiyor. Bu iki kısımda Claude yalnızca
kavramları açıkladı, soru sordu ve hataları gösterdi; kod ve sonuçlar bana ait.
Raporda gerekçelendirilen her parametreyi deney sonuçlarına bakarak ben seçtim.

## Aşama aşama

### Veri ve araçlar
- **İstediğim:** her fotoğrafın dört köşesini tıklayarak işaretlemek için bir araç ve
  ızgaraları ile koşulları etiket dosyalarına (Ek A formatı) yazan bir betik.
- **Yapay zekânın yazdığı:** `tools/annotate.py`, `tools/fill_metadata.py`.
- **Benim yaptığım:** 15 fotoğrafı çektim, bütün köşeleri işaretledim, ızgaraları girdim.
  `fill_metadata.py`'nin ilk sürümünün 6 dosyaya yanlış bulmaca ve koşul yazdığını fark ettim
  (dosya numaraları çekim saatine göre, fotoğrafları gönderdiğim sıraya göre değil); düzeltildi.

### Stage 1 – ön işleme
- **İstediğim:** açıklamalarıyla birlikte küçültme, gri tonlama, gürültü azaltma ve gölge
  düzeltme.
- **Yapay zekânın yazdığı:** `stage1_preprocess.py`, `tests/test_stage1.py`.
- **Benim yaptığım:** sonuçları gölgeli fotoğrafta (15) ve ahşap masa fotoğrafında (11)
  kontrol ettim; yan etkileri (gölge kenarı izleri, ahşap damarları) hata analizi için not ettim.

### Stage 2 – Harris köşeleri ve köşe seçimi
- **Yapay zekânın yazdığı:** `harris_response` (görüntü boyutuna göre blockSize ile
  `cv2.cornerHarris` çağrısı), `tests/test_stage2.py`, `experiments/param_study.py`
  (yalnızca döngü).
- **Benim yazdığım (yapay zekâ kodu yok):** `find_candidates` (göreli eşik + 7×7 maksimum
  olmayanları bastırma), `find_puzzle_region` (Otsu, RETR_CCOMP dış konturları, dışbükey zarf,
  4 köşe testi, alan ve mürekkep filtreleri, içinde en çok Harris adayı olan bölge) ve
  `select_corners` (bölgenin içindeki ve %3 tolerans içindeki adaylar, uç x+y / x−y değerleri).
  Claude; Harris'i, NMS'i, Otsu'yu, kontur hiyerarşilerini, convexHull ve approxPolyDP'yi
  açıkladı ve sorular sordu; kodumda hata olduğunda ipucu verdi, hataları ben düzelttim.
- **Verilere bakarak benim seçtiğim parametreler:** THRES_FRAC 0.002 (eşik çalışması);
  MAX_REGION_FRAC 0.48 ve MAX_INK_FRAC 0.27 (fotoğraflarımda bulmaca olan ve olmayan
  bölgelerin arasındaki boşluğun ortası); blockSize/ksize/k çalışmasından sonra BLOCK_FRAC
  0.005'te kaldım (0.003 kendi setimde 15/15 verdi, ama komşusu 0.002 fotoğraf 06'da zaten
  başarısız oluyor; bu yüzden iki komşusu da iyi olan değeri tuttum).
- **Hata:** bir yorum düzenlemesi `MIN_REGION_FRAC` tanımını yorum satırının içine
  yapıştırmıştı; Claude yerini bulmama yardım etti, ben düzelttim.

### Stage 3 – düzeltme
- **İstediğim:** A3 dördüncü köşe hatası ve A4 RMS ile birlikte A3, A4 (en küçük kareler) ve H4.
- **Yapay zekânın yazdığı:** `stage3_rectify.py`, `tests/test_stage3.py` (A4'ü
  `np.linalg.lstsq` ile karşılaştırarak doğruluyor).
- **Benim yaptığım:** serbestlik derecelerini ve A3'ün neden perspektifi modelleyemediğini
  çalıştım.

### Stage 4 – afin ve homografi
- **Benim yazdığım (yapay zekâ kodu yok):** `stage4_residual.py`, `stage4_grid.py`,
  `stage4_ocr.py` ve bütün sonuçlar.
- **Claude:** eğikliği tanımlamak için seçenekleri tartıştı (oranların büyüğü, oranların
  çarpımı, yalnızca üst/alt); karşılıklı kenar oranlarının büyüğünü ben seçtim. Adımları ve
  kullanılacak fonksiyonları listeledi (glob, json, np.linalg.norm, matplotlib, `sum(axis)`
  ile izdüşüm profilleri), yönlendirici sorular sordu (09 neden eğik görünüyor, 09'da A3 neden
  H4'ten iyi) ve hata ayıklarken bir görüntünün her ızgara çizgisini yazdıran tek seferlik bir
  terminal komutu verdi (teslim edilen kodun parçası değil). Bu çıktıya bakarak, görüntü
  sınırında kaldığı için 450 px'teki çizgiyi ölçmemeye karar verdim.

### Stage 5 – hücreler ve OCR
- **Yapay zekânın yazdığı:** `stage5_recognize.py`, `tests/test_stage5.py`,
  `experiments/stage5_margin.py`, `experiments/stage5_margin_ocr.py`,
  `experiments/stage5_empty.py`.
- **Benim yaptığım:** kenar payı tablolarına ve görsellerine bakarak 4 px'i seçtim (daha büyük
  paylar KrazyDad'in büyük rakamlarını ve 06'daki kaymış rakamları kesiyor); EMPTY_FRAC = 0.02'yi
  seçtim (boş ve dolu hücreler arasındaki boşluğun ortası; boşluğun ortasını Claude önerdi);
  psm karşılaştırmasından (10 / 8 / 13) sonra `--psm 8`'i seçtim.

### Stage 6 – geçerlilik kontrolü ve komut satırı programı
- **Yapay zekânın yazdığı:** `stage6_validate.py`, `tests/test_stage6.py`, `sudoku_scan.py`
  ve debug görüntüsünü geçerlilik kontrolünden sonra çizen değişiklik.

### Değerlendirme ve belgeler
- **Yapay zekânın yazdığı:** `evaluate.py` (Section 7 metrikleri); `README.md`,
  `requirements.txt` ve bu kaydın ilk taslakları.
- **Benim yaptığım:** kontrol edip düzenledim.

### Rapor
- Claude, sonuçlarımdan ve notlarımdan Bölüm IV dışındaki bölümlerin taslağını yazıp Türkçeye
  çevirdi; Bölüm IV ve sonuç paragrafı benim metnim, taslağı kontrol edip düzenledim.

## Reddedilen ya da değiştirilen öneriler
- Aşama dosyalarını bir `pipeline/` klasörüne taşımak (benim fikrimdi) ertelendi: teslimden
  birkaç gün önce bütün import'ları bozacaktı ve ödev bunu istemiyor.
- blockSize 0.003 reddedildi, 0.005'te kalındı (bkz. Stage 2).
- 450 px'teki çizgi, "zayıf tepe" kuralı yerine ızgara sapması ölçüsünden çıkarıldı; çünkü bu
  kural tam da ölçmek istediğimiz A3 kaymasını gizlerdi.