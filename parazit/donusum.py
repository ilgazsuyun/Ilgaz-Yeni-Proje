"""Dönüşüm katmanı: ses ve görüntü ile matris arasında gidiş geliş.

Ses tarafında kısa zamanlı Fourier dönüşümü (STFT) kullanılır: ses dalgası,
satırları frekans ve sütunları zaman olan bir karmaşık matrise açılır.
Genlik matrisi ayrıştırmaya girer; faz matrisi kenarda tutulur ve geri
dönüşte yeniden kullanılır.

Görüntü tarafında iş daha doğrudan: gri tonlamalı görüntü zaten bir
matristir, renkli görüntü ise kanal başına bir matristir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann


# ---------------------------------------------------------------------------
# Ses <-> matris
# ---------------------------------------------------------------------------

@dataclass
class SesMatrisi:
    """Bir ses kaydının matris hâli.

    genlik : (frekans x zaman) gerçek değerli matris; ayrıştırmaya giren bu.
    faz    : aynı boyutta faz matrisi; geri dönüşüm için saklanır.
    sr     : örnekleme hızı (Hz).
    n_ornek: özgün kaydın örnek sayısı (geri dönüşte kırpmak için).
    pencere: STFT pencere uzunluğu.
    atlama : STFT atlama (hop) uzunluğu.
    """

    genlik: np.ndarray
    faz: np.ndarray
    sr: int
    n_ornek: int
    pencere: int = 1024
    atlama: int = 256

    def _stft(self) -> ShortTimeFFT:
        return ShortTimeFFT(
            hann(self.pencere, sym=False), hop=self.atlama, fs=self.sr
        )


def _dalgayi_normalize_et(veri: np.ndarray) -> np.ndarray:
    """WAV verisini hangi türde gelirse gelsin [-1, 1] float'a çevirir."""
    if veri.ndim == 2:  # stereo -> mono
        veri = veri.mean(axis=1)
    if np.issubdtype(veri.dtype, np.integer):
        bilgi = np.iinfo(veri.dtype)
        veri = veri.astype(np.float64) / max(abs(bilgi.min), bilgi.max)
    else:
        veri = veri.astype(np.float64)
    return veri


def ses_matrise(yol: str, pencere: int = 1024, atlama: int = 256) -> SesMatrisi:
    """WAV dosyasını okur ve spektrogram matrisine açar."""
    sr, veri = wavfile.read(yol)
    dalga = _dalgayi_normalize_et(veri)
    donusum = ShortTimeFFT(hann(pencere, sym=False), hop=atlama, fs=sr)
    Z = donusum.stft(dalga)
    return SesMatrisi(
        genlik=np.abs(Z),
        faz=np.angle(Z),
        sr=int(sr),
        n_ornek=len(dalga),
        pencere=pencere,
        atlama=atlama,
    )


def matris_sese(matris: SesMatrisi, yol: str, genlik: np.ndarray | None = None) -> None:
    """Genlik matrisini (saklanan fazla birleştirip) WAV olarak yazar.

    `genlik` verilirse özgün genlik yerine o kullanılır; böylece
    ayrıştırmadan çıkan tek bir katman sese döndürülebilir.
    """
    G = matris.genlik if genlik is None else genlik
    Z = G * np.exp(1j * matris.faz)
    dalga = matris._stft().istft(Z, k1=matris.n_ornek)
    tepe = np.max(np.abs(dalga))
    if tepe > 1.0:
        dalga = dalga / tepe
    wavfile.write(yol, matris.sr, (dalga * 32767).astype(np.int16))


# ---------------------------------------------------------------------------
# Görüntü <-> matris
# ---------------------------------------------------------------------------

def resim_matrise(yol: str) -> np.ndarray:
    """Görüntüyü gri tonlamalı, [0, 1] aralığında bir matrise çevirir."""
    goruntu = Image.open(yol).convert("L")
    return np.asarray(goruntu, dtype=np.float64) / 255.0


def matris_resme(M: np.ndarray, yol: str) -> None:
    """Matrisi [0, 1] aralığına kırpıp PNG olarak yazar."""
    M = np.clip(M, 0.0, 1.0)
    Image.fromarray((M * 255).astype(np.uint8), mode="L").save(yol)
