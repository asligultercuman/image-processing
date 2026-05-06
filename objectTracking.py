"""
tracker.py — Nesne takibi modülü
agent.py bu sınıfı import ederek kullanır.
Doğrudan çalıştırılabilir: python tracker.py video.mp4
"""
import sys
import cv2
import numpy as np
import utils


def _tracker_olustur():
    """OpenCV kurulumuna göre en iyi tracker'ı döndürür."""
    for nesne, kaynak in [
        ("TrackerCSRT_create",  cv2),
        ("TrackerKCF_create",   cv2),
        ("TrackerMOSSE_create", cv2),
        ("TrackerCSRT_create",  getattr(cv2, "legacy", None)),
        ("TrackerKCF_create",   getattr(cv2, "legacy", None)),
    ]:
        if kaynak is None:
            continue
        factory = getattr(kaynak, nesne, None)
        if factory:
            return factory()
    return None


class ObjectTracker:
    """
    Tek nesne CSRT/KCF takibi.

    Kullanım:
        tracker = ObjectTracker()
        tracker.init(frame, bbox)          # bbox = (x, y, w, h)
        ok, new_bbox = tracker.update(frame)
        tracker.reset()
    """

    def __init__(self):
        self._tracker = None
        self.aktif = False
        self.son_bbox: tuple | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def init(self, frame: np.ndarray, bbox: tuple) -> bool:
        """Tracker'ı başlatır. Başarılıysa True döner."""
        t = _tracker_olustur()
        if t is None:
            print("[TRACKER] Tracker bulunamadı. `opencv-contrib-python` kurun.")
            return False
        t.init(frame, bbox)
        self._tracker = t
        self.son_bbox = bbox
        self.aktif = True
        print(f"[TRACKER] Başlatıldı → bbox: {bbox}")
        return True

    def update(self, frame: np.ndarray) -> tuple[bool, tuple | None]:
        """Bir kare ilerletir. (ok, bbox) döner; bbox = (x, y, w, h)."""
        if not self.aktif or self._tracker is None:
            return False, None
        ok, bbox = self._tracker.update(frame)
        if ok:
            self.son_bbox = tuple(map(int, bbox))
        else:
            self.aktif = False
            self._tracker = None
        return ok, self.son_bbox

    def reset(self):
        self._tracker = None
        self.aktif = False
        self.son_bbox = None
        print("[TRACKER] Sıfırlandı.")

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Aktif takip kutusunu frame üzerine çizer."""
        if not self.aktif or self.son_bbox is None:
            return frame
        x, y, w, h = self.son_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 3)
        cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 140, 255), -1)
        cv2.putText(frame, "Takip ediliyor", (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
        return frame

    # ── Yardımcı ──────────────────────────────────────────────────────────────

    @staticmethod
    def nokta_icindeki_bbox(nokta: tuple, boxes: list[dict]) -> tuple | None:
        """
        Tıklanan noktayı kapsayan en küçük bbox'ı döndürür.
        boxes: detector.detect() çıktısı formatında olmalı.
        """
        px, py = nokta
        adaylar = []
        for b in boxes:
            if b["x1"] <= px <= b["x2"] and b["y1"] <= py <= b["y2"]:
                alan = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                adaylar.append((alan, (b["x1"], b["y1"],
                                       b["x2"] - b["x1"], b["y2"] - b["y1"])))
        if not adaylar:
            return None
        adaylar.sort(key=lambda a: a[0])
        return adaylar[0][1]


# ── Bağımsız çalıştırma (demo) ────────────────────────────────────────────────

def _demo(video_path: str):
    from objectDetection import ObjectDetector
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    detector = ObjectDetector()
    tracker  = ObjectTracker()
    son_boxes: list[dict] = []
    son_frame = None

    def fare_tikla(event, x, y, flags, param):
        nonlocal son_frame
        if event != cv2.EVENT_LBUTTONDOWN or son_frame is None:
            return
        bbox = ObjectTracker.nokta_icindeki_bbox((x, y), son_boxes)
        if bbox:
            tracker.init(son_frame, bbox)
        else:
            print(f"[TAKİP] ({x},{y}) noktasında nesne yok.")

    cv2.namedWindow("Nesne Takibi")
    cv2.setMouseCallback("Nesne Takibi", fare_tikla)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = utils.frame_yeniden_boyutlandir(frame)
        son_frame = frame.copy()
        son_boxes = detector.detect(frame)
        detector.draw(frame, son_boxes)
        ok, _ = tracker.update(frame)
        if not ok and tracker.son_bbox is not None:
            cv2.putText(frame, "Takip kayboldu — nesneye tiklayin",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
        tracker.draw(frame)

        h = frame.shape[0]
        cv2.putText(frame, "Tiklayarak takip et  |  R: sifirla  |  ESC: cikis",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow("Nesne Takibi", frame)

        tus = cv2.waitKey(int(1000 / fps)) & 0xFF
        if tus == 27:
            break
        elif tus == ord("r"):
            tracker.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo(sys.argv[1] if len(sys.argv) > 1 else "car-video.mp4")