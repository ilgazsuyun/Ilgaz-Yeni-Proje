"""Parazit paketinin temel doğruluk testleri."""

import os
import tempfile

import numpy as np

from parazit import ayristirma, donusum


def test_svd_dusuk_rankli_matrisi_geri_kurar():
    rastgele = np.random.default_rng(0)
    A = rastgele.random((40, 3)) @ rastgele.random((3, 50))  # rank 3
    sonuc = ayristirma.svd_ayristir(A, rank=3)
    assert np.allclose(sonuc.yaklasik, A, atol=1e-8)
    assert np.linalg.norm(sonuc.artik) < 1e-8
    assert len(sonuc.katmanlar) == 3


def test_svd_katmanlarin_toplami_yaklasigi_verir():
    rastgele = np.random.default_rng(1)
    A = rastgele.random((20, 30))
    sonuc = ayristirma.svd_ayristir(A, rank=5)
    assert np.allclose(np.sum(sonuc.katmanlar, axis=0), sonuc.yaklasik)


def test_rpca_seyrek_paraziti_yayindan_ayirir():
    rastgele = np.random.default_rng(2)
    yayin = rastgele.random((60, 2)) @ rastgele.random((2, 80))  # düşük rank
    parazit = np.zeros_like(yayin)
    noktalar = rastgele.choice(yayin.size, size=yayin.size // 20, replace=False)
    parazit.flat[noktalar] = rastgele.uniform(2, 4, size=len(noktalar))
    sonuc = ayristirma.rpca_ayristir(yayin + parazit)
    # Yayın katmanı özgün yayına, parazit katmanı özgün parazite yakın olmalı
    assert np.linalg.norm(sonuc.yayin - yayin) / np.linalg.norm(yayin) < 0.05
    assert np.linalg.norm(sonuc.parazit - parazit) / np.linalg.norm(parazit) < 0.05


def test_nmf_bilesenleri_matrisi_yaklasik_kurar():
    rastgele = np.random.default_rng(3)
    W = rastgele.random((30, 3))
    H = rastgele.random((3, 40))
    M = W @ H
    sonuc = ayristirma.nmf_ayristir(M, bilesen_sayisi=3, azami_yineleme=500)
    assert sonuc.hata / np.linalg.norm(M) < 0.05
    assert np.allclose(np.sum(sonuc.bilesenler, axis=0), sonuc.desenler @ sonuc.etkinlikler)


def test_nmf_negatif_matrisi_reddeder():
    try:
        ayristirma.nmf_ayristir(np.array([[-1.0, 2.0]]))
    except ValueError:
        return
    raise AssertionError("negatif matris ValueError vermeliydi")


def test_ses_gidis_donusu_dalgayi_korur():
    from scipy.io import wavfile

    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    dalga = 0.5 * np.sin(2 * np.pi * 440 * t)
    with tempfile.TemporaryDirectory() as klasor:
        girdi = os.path.join(klasor, "girdi.wav")
        cikti = os.path.join(klasor, "cikti.wav")
        wavfile.write(girdi, sr, (dalga * 32767).astype(np.int16))
        matris = donusum.ses_matrise(girdi)
        assert matris.genlik.shape[0] > 0 and matris.genlik.shape[1] > 0
        donusum.matris_sese(matris, cikti)
        sr2, geri = wavfile.read(cikti)
        geri = geri.astype(np.float64) / 32767
        assert sr2 == sr
        # Kenar etkilerini dışarıda bırakıp ortayı karşılaştır
        orta = slice(sr // 10, -sr // 10)
        assert np.corrcoef(dalga[orta], geri[orta])[0, 1] > 0.99


def test_resim_gidis_donusu_pikselleri_korur():
    rastgele = np.random.default_rng(4)
    M = rastgele.random((32, 32))
    with tempfile.TemporaryDirectory() as klasor:
        yol = os.path.join(klasor, "resim.png")
        donusum.matris_resme(M, yol)
        geri = donusum.resim_matrise(yol)
        assert geri.shape == M.shape
        assert np.max(np.abs(geri - M)) < 1 / 255 + 1e-9
