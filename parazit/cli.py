"""Komut satırı arayüzü.

Kullanım örnekleri::

    python -m parazit demo                        # örnek parazitli ses + görüntü üret ve ayrıştır
    python -m parazit ses kayit.wav -c cikti/     # sesi yayın/parazit/bileşenlere ayır
    python -m parazit resim foto.png -c cikti/    # görüntüyü yayın/parazit/katmanlara ayır
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from parazit import ayristirma, donusum


# ---------------------------------------------------------------------------
# Ortak yardımcılar
# ---------------------------------------------------------------------------

def _klasor_hazirla(yol: str) -> str:
    os.makedirs(yol, exist_ok=True)
    return yol


def _goruntu_olarak_normalize(M: np.ndarray) -> np.ndarray:
    """Herhangi bir matrisi görselleştirme için [0, 1] aralığına çeker."""
    en_kucuk, en_buyuk = float(M.min()), float(M.max())
    if en_buyuk - en_kucuk < 1e-12:
        return np.zeros_like(M)
    return (M - en_kucuk) / (en_buyuk - en_kucuk)


def _spektrogram_gorseli(G: np.ndarray) -> np.ndarray:
    """Genlik spektrogramını log ölçekte görselleştirilebilir matrise çevirir."""
    return _goruntu_olarak_normalize(np.log1p(G * 100.0))


# ---------------------------------------------------------------------------
# Ses akışı
# ---------------------------------------------------------------------------

def ses_ayristir(yol: str, cikti: str, bilesen_sayisi: int) -> list[str]:
    """Bir WAV dosyasını okur, spektrogramını ayrıştırır, sonuçları yazar."""
    _klasor_hazirla(cikti)
    yazilanlar: list[str] = []
    matris = donusum.ses_matrise(yol)
    G = matris.genlik
    print(f"[1/3] {yol} okundu: {G.shape[0]} frekans x {G.shape[1]} zaman matrisi")

    # Spektrogramın kendisi, bakılabilsin diye
    hedef = os.path.join(cikti, "spektrogram.png")
    donusum.matris_resme(_spektrogram_gorseli(G), hedef)
    yazilanlar.append(hedef)

    # Robust PCA: yayın + parazit
    rpca = ayristirma.rpca_ayristir(G)
    print(f"[2/3] Robust PCA {rpca.yineleme} adımda ayrıştı: yayın + parazit")
    for ad, katman in (("yayin", rpca.yayin), ("parazit", rpca.parazit)):
        temiz = np.maximum(katman, 0.0)
        hedef = os.path.join(cikti, f"{ad}.wav")
        donusum.matris_sese(matris, hedef, genlik=temiz)
        yazilanlar.append(hedef)
        hedef = os.path.join(cikti, f"{ad}.png")
        donusum.matris_resme(_spektrogram_gorseli(temiz), hedef)
        yazilanlar.append(hedef)

    # NMF: anlamlı bileşenler (kaynaklar)
    nmf = ayristirma.nmf_ayristir(G, bilesen_sayisi=bilesen_sayisi)
    print(f"[3/3] NMF {bilesen_sayisi} bileşene ayırdı (hata: {nmf.hata:.4f})")
    for i, B in enumerate(nmf.bilesenler, start=1):
        hedef = os.path.join(cikti, f"bilesen_{i}.wav")
        donusum.matris_sese(matris, hedef, genlik=B)
        yazilanlar.append(hedef)

    return yazilanlar


# ---------------------------------------------------------------------------
# Görüntü akışı
# ---------------------------------------------------------------------------

def resim_ayristir(yol: str, cikti: str, rank: int) -> list[str]:
    """Bir görüntüyü okur, piksel matrisini ayrıştırır, sonuçları yazar."""
    _klasor_hazirla(cikti)
    yazilanlar: list[str] = []
    M = donusum.resim_matrise(yol)
    print(f"[1/3] {yol} okundu: {M.shape[0]} x {M.shape[1]} piksel matrisi")

    # Robust PCA: yayın + parazit
    rpca = ayristirma.rpca_ayristir(M)
    print(f"[2/3] Robust PCA {rpca.yineleme} adımda ayrıştı: yayın + parazit")
    for ad, katman in (("yayin", rpca.yayin), ("parazit", rpca.parazit)):
        hedef = os.path.join(cikti, f"{ad}.png")
        gorsel = np.clip(katman, 0, 1) if ad == "yayin" else _goruntu_olarak_normalize(np.abs(katman))
        donusum.matris_resme(gorsel, hedef)
        yazilanlar.append(hedef)

    # SVD: önem sırasına dizilmiş katmanlar
    svd = ayristirma.svd_ayristir(M, rank=rank)
    enerji = float(np.sum(svd.tekil[:rank] ** 2) / np.sum(svd.tekil**2) * 100)
    print(f"[3/3] SVD ilk {rank} katmanı çıkardı (enerjinin %{enerji:.1f}'i)")
    hedef = os.path.join(cikti, "svd_yaklasik.png")
    donusum.matris_resme(np.clip(svd.yaklasik, 0, 1), hedef)
    yazilanlar.append(hedef)
    for i, K in enumerate(svd.katmanlar[: min(rank, 6)], start=1):
        hedef = os.path.join(cikti, f"katman_{i}.png")
        donusum.matris_resme(_goruntu_olarak_normalize(K), hedef)
        yazilanlar.append(hedef)

    return yazilanlar


# ---------------------------------------------------------------------------
# Demo: parazitli yayın üretimi
# ---------------------------------------------------------------------------

def _parazitli_ses_uret(yol: str, sr: int = 16000, sure: float = 3.0) -> None:
    """İki notalı bir 'yayın' ile çıtırtı parazitini üst üste bindirir."""
    from scipy.io import wavfile

    rastgele = np.random.default_rng(42)
    t = np.linspace(0, sure, int(sr * sure), endpoint=False)
    yayin = 0.5 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    yayin += 0.5 * np.sin(2 * np.pi * 660 * t) * (np.sin(2 * np.pi * 0.5 * t) <= 0)
    parazit = np.zeros_like(t)
    citirti = rastgele.choice(len(t), size=len(t) // 200, replace=False)
    parazit[citirti] = rastgele.uniform(-1, 1, size=len(citirti))
    zemin = rastgele.normal(0, 0.02, size=len(t))
    karisim = yayin + parazit + zemin
    karisim /= np.max(np.abs(karisim))
    wavfile.write(yol, sr, (karisim * 32767).astype(np.int16))


def _parazitli_resim_uret(yol: str, boyut: int = 256) -> None:
    """Düzenli bir desen ('yayın') üstüne ekran karı ve çizgi paraziti bindirir."""
    rastgele = np.random.default_rng(42)
    y, x = np.mgrid[0:boyut, 0:boyut] / boyut
    yayin = 0.5 + 0.25 * np.sin(2 * np.pi * 3 * x) * np.cos(2 * np.pi * 2 * y)
    parazit = np.zeros((boyut, boyut))
    kar = rastgele.random((boyut, boyut)) < 0.05
    parazit[kar] = rastgele.uniform(-0.8, 0.8, size=int(kar.sum()))
    for satir in rastgele.choice(boyut, size=5, replace=False):
        parazit[satir, :] += rastgele.uniform(0.4, 0.8)
    donusum.matris_resme(yayin + parazit, yol)


def demo(cikti: str) -> list[str]:
    """Parazitli örnek ses ve görüntü üretir, sonra ikisini de ayrıştırır."""
    _klasor_hazirla(cikti)
    print("— Parazitli yayın örnekleri üretiliyor —")
    ses_yolu = os.path.join(cikti, "parazitli_ses.wav")
    resim_yolu = os.path.join(cikti, "parazitli_resim.png")
    _parazitli_ses_uret(ses_yolu)
    _parazitli_resim_uret(resim_yolu)

    print("\n— Ses ayrıştırılıyor —")
    yazilanlar = [ses_yolu, resim_yolu]
    yazilanlar += ses_ayristir(ses_yolu, os.path.join(cikti, "ses"), bilesen_sayisi=3)
    print("\n— Görüntü ayrıştırılıyor —")
    yazilanlar += resim_ayristir(resim_yolu, os.path.join(cikti, "resim"), rank=4)
    return yazilanlar


# ---------------------------------------------------------------------------
# Argüman çözümleme
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    cozumleyici = argparse.ArgumentParser(
        prog="parazit",
        description="İç içe geçmiş matrisleri ayrıştırır: ses/görüntü -> matris -> anlamlı parçalar.",
    )
    altlar = cozumleyici.add_subparsers(dest="komut", required=True)

    p_demo = altlar.add_parser("demo", help="örnek parazitli ses ve görüntü üret, ayrıştır")
    p_demo.add_argument("-c", "--cikti", default="cikti", help="çıktı klasörü")

    p_ses = altlar.add_parser("ses", help="WAV dosyasını ayrıştır")
    p_ses.add_argument("dosya", help="girdi WAV dosyası")
    p_ses.add_argument("-c", "--cikti", default="cikti/ses", help="çıktı klasörü")
    p_ses.add_argument("-b", "--bilesen", type=int, default=4, help="NMF bileşen sayısı")

    p_resim = altlar.add_parser("resim", help="görüntü dosyasını ayrıştır")
    p_resim.add_argument("dosya", help="girdi görüntü dosyası (PNG/JPG)")
    p_resim.add_argument("-c", "--cikti", default="cikti/resim", help="çıktı klasörü")
    p_resim.add_argument("-r", "--rank", type=int, default=8, help="SVD katman sayısı")

    args = cozumleyici.parse_args(argv)

    if args.komut == "demo":
        yazilanlar = demo(args.cikti)
    elif args.komut == "ses":
        yazilanlar = ses_ayristir(args.dosya, args.cikti, args.bilesen)
    else:
        yazilanlar = resim_ayristir(args.dosya, args.cikti, args.rank)

    print("\nYazılan dosyalar:")
    for yol in yazilanlar:
        print(f"  {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
