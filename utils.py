"""
utils.py — Ortak yardımcı fonksiyonlar
Tüm modüller buradan import eder; hiçbir yerde tekrar tanımlanmaz.
"""
import cv2
import numpy as np


# ── Görsel yardımcılar ────────────────────────────────────────────────────────

# HSV renk aralıkları  (alt_hsv, ust_hsv)
HSV_ARALIK: dict[str, tuple] = {
    "red":    ((0,  70, 40), (10, 255, 255)),   # kırmızı — ikinci aralık ayrıca
    "red2":   ((160, 70, 40), (180, 255, 255)),
    "blue":   ((90,  50, 50), (130, 255, 255)),
    "green":  ((35,  50, 40), (85,  255, 255)),
    "yellow": ((20,  60, 60), (35,  255, 255)),
    "cyan":   ((80,  50, 50), (100, 255, 255)),
    "purple": ((130, 50, 50), (160, 255, 255)),
    "orange": ((10,  80, 80), (20,  255, 255)),
    "white":  ((0,    0,170), (180, 40, 255)),
    "black":  ((0,    0,  0), (180,255,  60)),
    "pink":   ((140, 30, 150),(180,255, 255)),
}

# Hedef renkler: hue değeri  (siyah/beyaz için özel mod)
HEDEF_HUE: dict[str, int | None] = {
    "red": 0, "blue": 110, "green": 60, "yellow": 28,
    "cyan": 90, "purple": 145, "orange": 15, "pink": 155,
    "white": -1,   # özel: doygunluğu sıfırla + parlaklık max
    "black": -2,   # özel: parlaklığı sıfırla
}

_RENKLER = [
    (0, 255, 0),    (0, 200, 255),  (255, 100, 0),
    (255, 0, 200),  (0, 100, 255),  (180, 255, 0),
]

def sinif_rengi(sinif_id: int) -> tuple:
    """Her sınıf ID'sine tutarlı bir BGR rengi döndürür."""
    return _RENKLER[sinif_id % len(_RENKLER)]


def frame_yeniden_boyutlandir(frame: np.ndarray, maks_yukseklik: int = 800) -> np.ndarray:
    h, w = frame.shape[:2]
    if h > maks_yukseklik:
        oran = maks_yukseklik / h
        frame = cv2.resize(frame, (int(w * oran), int(h * oran)))
    return frame


# ── Görüntü iyileştirme ───────────────────────────────────────────────────────

def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """LAB uzayında L kanalına CLAHE uygular; karanlık/sisli görüntüleri iyileştirir."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    updated_lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)


def rengi_degistir(roi: np.ndarray,
                   kaynak: str,
                   hedef: str) -> np.ndarray:
    """
    roi içindeki 'kaynak' rengini 'hedef' renge dönüştürür.
    Siyah ve beyaz için özel HSV manipülasyonu kullanılır.
    """
    import numpy as np
    aralik = HSV_ARALIK.get(kaynak)
    if aralik is None:
        return roi

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Kırmızı iki aralıklı
    if kaynak == "red":
        aralik2 = HSV_ARALIK.get("red2")
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(aralik[0]),  np.array(aralik[1])),
            cv2.inRange(hsv, np.array(aralik2[0]), np.array(aralik2[1])),
        )
    else:
        mask = cv2.inRange(hsv, np.array(aralik[0]), np.array(aralik[1]))

    if cv2.countNonZero(mask) == 0:
        return roi

    h, s, v = cv2.split(hsv)
    hue = HEDEF_HUE.get(hedef, 60)

    if hue == -1:       # → beyaz
        h[mask > 0] = 0
        s[mask > 0] = 0
        v[mask > 0] = 255
    elif hue == -2:     # → siyah
        v[mask > 0] = 0
    else:
        h[mask > 0] = hue
        # Doygunluğu canlı tut
        s[mask > 0] = np.clip(s[mask > 0].astype(int) + 30, 80, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


# ── Etiket çizimi ─────────────────────────────────────────────────────────────

def kutu_ve_etiket_ciz(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                        etiket: str, renk: tuple, kalinlik: int = 2) -> None:
    """Bounding box + arka planlı etiket çizer. frame üzerinde in-place çalışır."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, kalinlik)
    (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), renk, -1)
    cv2.putText(frame, etiket, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)