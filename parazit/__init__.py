"""Parazit: iç içe geçmiş matrisleri ayrıştıran araç takımı.

Parazitli bir televizyon yayını gibi, elimizdeki veri çoğu zaman tek bir
matris değildir: asıl yayın (düzenli, düşük ranklı yapı), parazit (seyrek,
ani bozulmalar) ve zemin gürültüsü aynı matrisin içinde iç içe durur.

Bu paket iki iş yapar:

1. **Dönüşüm** (`donusum`): ses dosyasını spektrogram matrisine, görüntüyü
   piksel matrisine çevirir; ayrıştırılmış matrisleri tekrar sese ve
   görüntüye döndürür.
2. **Ayrıştırma** (`ayristirma`): bir matrisi anlamlı parçalara böler:
   SVD ile rank-1 katmanlara, Robust PCA ile yayın + parazit ikilisine,
   NMF ile toplamsal bileşenlere.
"""

from parazit.donusum import (
    ses_matrise,
    matris_sese,
    resim_matrise,
    matris_resme,
)
from parazit.ayristirma import (
    svd_ayristir,
    rpca_ayristir,
    nmf_ayristir,
)

__version__ = "0.1.0"

__all__ = [
    "ses_matrise",
    "matris_sese",
    "resim_matrise",
    "matris_resme",
    "svd_ayristir",
    "rpca_ayristir",
    "nmf_ayristir",
]
