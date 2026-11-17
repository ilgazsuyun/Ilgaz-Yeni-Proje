# Parazit

Parazitli bir televizyon yayınını düşünün: gördüğünüz tek karede aslında
birkaç matris iç içe durur. Asıl yayın düzenli ve düşük ranklı bir yapıdır;
ekran karı seyrek ama şiddetli bir bozulmadır; altta da zemin gürültüsü
vardır. **Parazit**, bu iç içe geçmiş matrisleri birbirinden ayıran bir
araç takımıdır.

İki adımda çalışır:

1. **Dönüşüm** — ses dosyası STFT ile (frekans × zaman) spektrogram
   matrisine, görüntü piksel matrisine çevrilir. Ayrıştırılan katmanlar
   aynı yoldan geriye, sese (WAV) ve görüntüye (PNG) döndürülür.
2. **Ayrıştırma** — matris anlamlı parçalara bölünür:

   | Yöntem | Ne yapar | Ne zaman |
   |---|---|---|
   | **Robust PCA** | `M ≈ yayın + parazit`: düşük ranklı yapı ile seyrek bozulmayı ayırır | Parazit temizleme |
   | **SVD** | Matrisi önem sırasına dizilmiş rank-1 katmanlara böler | Yapı çözümleme, sıkıştırma |
   | **NMF** | Spektrogramı toplamsal bileşenlere (kaynaklara) ayırır | Ses kaynağı ayrıştırma |

## Kurulum

```bash
pip install numpy scipy Pillow
pip install -e .   # istenirse, `parazit` komutu için
```

## Kullanım

```bash
# Örnek parazitli ses ve görüntü üret, ikisini de ayrıştır
python -m parazit demo

# Bir ses kaydını ayrıştır: yayın.wav, parazit.wav ve NMF bileşenleri yazılır
python -m parazit ses kayit.wav -c cikti/ses -b 4

# Bir görüntüyü ayrıştır: yayın, parazit ve SVD katmanları yazılır
python -m parazit resim foto.png -c cikti/resim -r 8
```

Python içinden:

```python
from parazit import ses_matrise, matris_sese, rpca_ayristir
import numpy as np

matris = ses_matrise("kayit.wav")          # ses -> spektrogram matrisi
sonuc = rpca_ayristir(matris.genlik)        # matris -> yayın + parazit
matris_sese(matris, "temiz.wav",            # yayın katmanı -> ses
            genlik=np.maximum(sonuc.yayin, 0))
```

## Testler

```bash
python -m pytest tests/
```

## Sınırlar

- Robust PCA, tam satır veya sütun boyunca uzanan paraziti (ör. yatay
  çizgiler) yayın katmanında bırakabilir; böyle bir çizgi kendisi rank-1
  olduğu için matematiksel olarak "düzenli yapı" sayılır.
- Ses geri dönüşümünde özgün karışımın fazı kullanılır; katman sesleri bu
  yüzden hafif artefakt taşıyabilir.
