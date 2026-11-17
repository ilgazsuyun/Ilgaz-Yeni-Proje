"""Ayrıştırma katmanı: bir matrisi iç içe geçmiş parçalarına bölme.

Üç yöntem var; üçü de aynı soruya başka açıdan bakar:
"Bu matrisin içinde hangi matrisler üst üste binmiş?"

- **SVD**: matrisi önem sırasına dizilmiş rank-1 katmanların toplamı olarak
  yazar. İlk katmanlar kaba yapıyı, son katmanlar ince ayrıntıyı ve
  gürültüyü taşır.
- **Robust PCA**: matrisi `yayin + parazit` olarak ikiye böler. `yayin`
  düşük ranklı düzenli yapıdır, `parazit` seyrek ama şiddetli bozulmadır;
  parazitli ekran görüntüsündeki çizgiler ve benekler gibi.
- **NMF**: negatif olmayan matrisi (örneğin spektrogramı) toplamsal
  bileşenlere ayırır. Her bileşen bir frekans deseni ile onun zaman
  içindeki etkinliğinin çarpımıdır; sesteki ayrı kaynaklara denk düşer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# SVD: rank-1 katmanlar
# ---------------------------------------------------------------------------

@dataclass
class SvdSonucu:
    """SVD ayrıştırmasının sonucu.

    yaklasik  : ilk `rank` katmanın toplamı (temiz yapı).
    artik     : geriye kalan her şey (ince ayrıntı + gürültü).
    katmanlar : her biri rank-1 bir matris olan katman listesi.
    tekil     : tekil değerler (katmanların ağırlıkları).
    """

    yaklasik: np.ndarray
    artik: np.ndarray
    katmanlar: list[np.ndarray] = field(default_factory=list)
    tekil: np.ndarray = field(default_factory=lambda: np.empty(0))


def svd_ayristir(M: np.ndarray, rank: int = 8) -> SvdSonucu:
    """Matrisi tekil değer ayrışımıyla rank-1 katmanlara böler."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    rank = max(1, min(rank, len(s)))
    katmanlar = [s[i] * np.outer(U[:, i], Vt[i, :]) for i in range(rank)]
    yaklasik = np.sum(katmanlar, axis=0)
    return SvdSonucu(
        yaklasik=yaklasik,
        artik=M - yaklasik,
        katmanlar=katmanlar,
        tekil=s,
    )


# ---------------------------------------------------------------------------
# Robust PCA: yayın + parazit
# ---------------------------------------------------------------------------

@dataclass
class RpcaSonucu:
    """Robust PCA sonucu: M ≈ yayin + parazit."""

    yayin: np.ndarray    # düşük ranklı düzenli yapı
    parazit: np.ndarray  # seyrek bozulma
    yineleme: int        # kaç adımda yakınsadığı


def _yumusak_esik(X: np.ndarray, esik: float) -> np.ndarray:
    """Yumuşak eşikleme: küçük değerleri sıfırlar, büyükleri eşik kadar çeker."""
    return np.sign(X) * np.maximum(np.abs(X) - esik, 0.0)


def _tekil_deger_esikle(X: np.ndarray, esik: float) -> np.ndarray:
    """Tekil değerlere yumuşak eşik uygular (düşük rank'a itme)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - esik, 0.0)
    return (U * s) @ Vt


def rpca_ayristir(
    M: np.ndarray,
    lam: float | None = None,
    azami_yineleme: int = 200,
    tolerans: float = 1e-7,
) -> RpcaSonucu:
    """Matrisi düşük ranklı 'yayın' ile seyrek 'parazit'e ayırır.

    Candès ve arkadaşlarının Principal Component Pursuit problemi,
    inexact ALM (artırılmış Lagrange) yöntemiyle çözülür:

        min |L|_* + lam * |S|_1   öyle ki   M = L + S
    """
    M = np.asarray(M, dtype=np.float64)
    if lam is None:
        lam = 1.0 / np.sqrt(max(M.shape))

    norm = np.linalg.norm(M, ord="fro")
    if norm == 0:
        return RpcaSonucu(yayin=M.copy(), parazit=np.zeros_like(M), yineleme=0)

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = M / max(np.linalg.norm(M, ord=2), np.max(np.abs(M)) / lam)
    mu = 1.25 / np.linalg.norm(M, ord=2)
    mu_carpan = 1.5

    yineleme = 0
    for yineleme in range(1, azami_yineleme + 1):
        L = _tekil_deger_esikle(M - S + Y / mu, 1.0 / mu)
        S = _yumusak_esik(M - L + Y / mu, lam / mu)
        fark = M - L - S
        Y = Y + mu * fark
        mu = min(mu * mu_carpan, mu * 1e7)
        if np.linalg.norm(fark, ord="fro") / norm < tolerans:
            break

    return RpcaSonucu(yayin=L, parazit=S, yineleme=yineleme)


# ---------------------------------------------------------------------------
# NMF: toplamsal bileşenler
# ---------------------------------------------------------------------------

@dataclass
class NmfSonucu:
    """NMF sonucu: M ≈ W @ H, bileşen başına bir matris.

    desenler   : W, (satır x bileşen); her sütun bir desen (örn. frekans profili).
    etkinlikler: H, (bileşen x sütun); her satır o desenin zaman çizgisi.
    bilesenler : her biri W[:, i] ⊗ H[i, :] olan matris listesi.
    hata       : son yeniden kurma hatası (Frobenius).
    """

    desenler: np.ndarray
    etkinlikler: np.ndarray
    bilesenler: list[np.ndarray]
    hata: float


def nmf_ayristir(
    M: np.ndarray,
    bilesen_sayisi: int = 4,
    azami_yineleme: int = 300,
    tohum: int | None = 0,
) -> NmfSonucu:
    """Negatif olmayan matrisi çarpımsal güncellemelerle W·H'ye ayırır."""
    M = np.asarray(M, dtype=np.float64)
    if np.any(M < 0):
        raise ValueError("NMF yalnızca negatif olmayan matrislerle çalışır.")

    eps = 1e-10
    rastgele = np.random.default_rng(tohum)
    olcek = np.sqrt(M.mean() / max(bilesen_sayisi, 1)) + eps
    W = rastgele.uniform(eps, 1.0, size=(M.shape[0], bilesen_sayisi)) * olcek
    H = rastgele.uniform(eps, 1.0, size=(bilesen_sayisi, M.shape[1])) * olcek

    for _ in range(azami_yineleme):
        H *= (W.T @ M) / (W.T @ W @ H + eps)
        W *= (M @ H.T) / (W @ H @ H.T + eps)

    bilesenler = [np.outer(W[:, i], H[i, :]) for i in range(bilesen_sayisi)]
    hata = float(np.linalg.norm(M - W @ H, ord="fro"))
    return NmfSonucu(desenler=W, etkinlikler=H, bilesenler=bilesenler, hata=hata)
