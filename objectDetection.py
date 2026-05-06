"""
detector.py — Nesne tespiti modülü
agent.py bu sınıfı import ederek kullanır.
Doğrudan çalıştırılabilir: python detector.py video.mp4
"""
import sys
import numpy as np
import cv2
from ultralytics import YOLO
import utils


class ObjectDetector:
    """
    YOLOv8 tabanlı nesne tespiti ve renk değiştirme.

    Kullanım:
        detector = ObjectDetector("yolov8n-seg.pt")
        boxes = detector.detect(frame)          # → list[dict]
        frame = detector.draw(frame, boxes)     # kutular çizilir
    """

    # Mavi gömlek → yeşile dönüştürme varsayılan değerleri
    _ALT_MAVI = np.array([90,  50,  50])
    _UST_MAVI = np.array([130, 255, 255])
    _HEDEF_HUE = 60   # HSV'de yeşil

    def __init__(self, model_path="yolov8n-seg.pt", conf=0.4,
                renk_komutu: dict | None = None, renk_degistir=False):
        """
        renk_komutu = {
            "hedef_sinif": "car",      # hangi YOLO sınıfına uygulanacak (None = hepsi)
            "kaynak_renk": "blue",     # değiştirilecek renk
            "hedef_renk":  "green",    # yeni renk
        }
        """
        self.model = YOLO(model_path)
        self.conf  = conf
        self.renk_komutu: dict | None = renk_komutu
        self.renk_degistir = renk_degistir
        # ── Ana metodlar ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        frame üzerinde YOLO çıkarımı yapar.
        Döndürür: [{"x1","y1","x2","y2","sinif_id","etiket","guven"}, ...]
        """
        results = self.model(frame, verbose=False, conf=self.conf)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            sinif_id = int(box.cls[0])
            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "sinif_id": sinif_id,
                "etiket": results.names[sinif_id],
                "guven": float(box.conf[0]),
            })
        return boxes

    def draw(self, frame, boxes, hedef_sinif=None):
        for b in boxes:
            if hedef_sinif and b["etiket"] != hedef_sinif:
                continue

            # Dinamik renk değiştirme
            if self.renk_komutu:
                rk = self.renk_komutu
                sinif_esle = (rk.get("hedef_sinif") is None or
                            rk["hedef_sinif"] == b["etiket"])
                if sinif_esle:
                    roi = frame[b["y1"]:b["y2"], b["x1"]:b["x2"]]
                    if roi.size > 0:
                        frame[b["y1"]:b["y2"], b["x1"]:b["x2"]] = utils.rengi_degistir(
                            roi,
                            rk["kaynak_renk"],
                            rk["hedef_renk"],
                        )

            utils.kutu_ve_etiket_ciz(
                frame, b["x1"], b["y1"], b["x2"], b["y2"],
                f"{b['etiket']}  {b['guven']:.0%}",
                utils.sinif_rengi(b["sinif_id"]),
            )
        return frame

"""
# ── Bağımsız çalıştırma (demo) ────────────────────────────────────────────────

def _demo(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    detector = ObjectDetector(renk_degistir=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = utils.frame_yeniden_boyutlandir(frame)
        frame = utils.apply_clahe(frame)
        boxes = detector.detect(frame)
        frame = detector.draw(frame, boxes)

        h = frame.shape[0]
        cv2.putText(frame, f"Nesne: {len(boxes)}  |  ESC cikis",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow("Nesne Tespiti", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo(sys.argv[1] if len(sys.argv) > 1 else "blue-tshirt.mp4")
"""