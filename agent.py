"""
agent.py — Video analiz ajanı
Yazılı veya sesli komutlarla nesne tespiti ve takibi yönetir.

Kullanım:
    python agent.py                      # varsayılan video
    python agent.py video.mp4            # belirli video
"""
from curses import raw
import queue
import re
import sys
import threading
from collections import deque
from datetime import datetime

import cv2

# Kendi modüllerimiz — artık global değişken ve tekrar eden kod yok
from objectDetection import ObjectDetector
from objectTracking import ObjectTracker
import utils

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# ── Sabitler ──────────────────────────────────────────────────────────────────

CLASS_KEYWORDS = {
    "person":     ["insan", "kisi", "adam", "kadin", "person"],
    "car":        ["araba", "otomobil", "car"],
    "truck":      ["kamyon", "tir", "truck"],
    "bus":        ["otobus", "bus"],
    "motorcycle": ["motor", "motosiklet", "motorcycle"],
    "bicycle":    ["bisiklet", "bicycle"],
}

COLOR_KEYWORDS = {
    "red":    ["kirmizi", "kizil", "red"],
    "blue":   ["mavi", "lacivert", "blue"],
    "green":  ["yesil", "green"],
    "yellow": ["sari", "yellow"],
    "white":  ["beyaz", "white"],
    "black":  ["siyah", "black"],
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── Komut ayrıştırma (saf fonksiyonlar — sınıftan bağımsız) ──────────────────

def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    tr = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocigusoc")
    text = text.translate(tr)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_class(text: str) -> str | None:
    for cls, keywords in CLASS_KEYWORDS.items():
        if any(k in text for k in keywords):
            return cls
    return None


def extract_color(text: str) -> str | None:
    for color, keywords in COLOR_KEYWORDS.items():
        if any(k in text for k in keywords):
            return color
    return None


def parse_command(raw: str) -> dict:
    t = normalize(raw)
    renkler = []
    for color, keywords in COLOR_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                renkler.append((t.index(kw), color))
                break
    renkler.sort()
    renk_listesi = [c for _, c in renkler]
    base = {
        "target_class": extract_class(t),
        "target_color":  renk_listesi[0] if renk_listesi else None,
        "hedef_renk":    renk_listesi[1] if len(renk_listesi) >= 2 else None,
        "shape": "circle" if any(x in t for x in ["daire", "circle"])
                 else ("rectangle" if any(x in t for x in ["kare", "dikdortgen", "rectangle", "square"])
                       else None),
    }

    if not t:
        return {"action": "noop", **base}
    if t in {"q", "quit", "exit", "cikis"}:
        return {"action": "quit", **base}
    if "yardim" in t or "help" in t:
        return {"action": "help", **base}
    if any(x in t for x in ["hedefi sifirla", "filtreyi temizle", "hedefi temizle"]):
        return {"action": "reset_target", **base}
    if any(x in t for x in ["takibi durdur", "takip durdur", "stop tracking", "takibi bitir"]):
        return {"action": "stop_tracking", **base}
    if any(x in t for x in ["takibi baslat", "takip et", "takibe al", "start tracking"]):
        return {"action": "start_tracking", **base}
    if any(x in t for x in ["kaza tespitini ac", "kaza modu ac"]):
        return {"action": "accident_on", **base}
    if any(x in t for x in ["kaza tespitini kapat", "kaza modu kapat"]):
        return {"action": "accident_off", **base}
    if "kaza esigini arttir" in t:
        return {"action": "accident_threshold_up", **base}
    if "kaza esigini azalt" in t:
        return {"action": "accident_threshold_down", **base}
    if any(x in t for x in ["kaza var mi", "kaza durumu"]):
        return {"action": "accident_status", **base}
    if any(x in t for x in ["cizimi temizle", "temizle", "sekilleri temizle"]):
        return {"action": "clear_drawings", **base}
    if any(x in t for x in ["daire ciz", "daire", "circle"]):
        return {"action": "draw_circle", **base}
    if any(x in t for x in ["kare ciz", "kare", "dikdortgen", "rectangle", "square"]):
        return {"action": "draw_rectangle", **base}
    if any(x in t for x in ["isaretle", "mark"]):
        return {"action": "mark_object", **base}
    if any(x in t for x in ["rengini", "renge cevir", "renge boya", "rengi degistir",
                            "rengini degistir", "boyasini degistir"]):
        return {"action": "renk_degistir", **base}
    if any(x in t for x in ["renk degistirmeyi kapat", "renk degistirme kapat",
                            "rengi geri al", "renk iptal"]):
        return {"action": "renk_iptal", **base}

    return {"action": "unknown", **base}


# ── VideoAssistant ────────────────────────────────────────────────────────────

class VideoAssistant:
    def __init__(self, video_path: str = "blue-tshirt.mp4",
                 det_model: str = "yolov8n-seg.pt"):
        self.video_path = video_path

        self.detector = ObjectDetector(model_path=det_model, renk_degistir=True)
        self.tracker  = ObjectTracker()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video açılamadı: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25

        # Durum
        self.running        = True
        self.status_message = "Hazır."
        self.frame_index    = 0
        self.last_boxes: list[dict] = []
        self.last_frame     = None
        self.last_bbox      = None
        self.target_class   = None
        self.target_color   = None
        self.pending_draw   = None
        self.mark_enabled   = False

        # Performans
        self.fast_mode    = True
        self.infer_stride = 2
        self.infer_imgsz  = 640

        # Kaza tespiti
        self.accident_enabled        = True
        self.accident_threshold      = 2.4
        self.accident_score          = 0.0
        self.accident_alert          = False
        self.accident_cooldown       = 0
        self.vehicle_tracks: dict    = {}
        self.next_track_id           = 1

        # Ses
        self.command_queue  = queue.Queue()
        self.voice_continuous   = False
        self.voice_stop_event   = threading.Event()
        self.voice_lock         = threading.Lock()
        self.voice_thread       = None
        self.preferred_mic      = self._detect_mic()

    # ── Ses ──────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        print(f"[{datetime.now():%H:%M:%S}] {msg}")

    def _detect_mic(self):
        if sr is None:
            return None
        try:
            mics = sr.Microphone.list_microphone_names()
        except Exception:
            return None
        tokens = ["mikrofon dizisi", "microphone array", "realtek"]
        for i, name in enumerate(mics):
            if any(t in name.lower() for t in tokens):
                return i
        return 0 if mics else None

    def _build_recognizer(self):
        r = sr.Recognizer()
        r.dynamic_energy_threshold = False
        r.energy_threshold = 220
        r.pause_threshold = 0.55
        return r

    def start_input_listener(self):
        def loop():
            while self.running:
                try:
                    cmd = input("Komut: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                self.command_queue.put(cmd)
                if cmd.lower() in {"q", "quit", "exit", "cikis"}:
                    break
        threading.Thread(target=loop, daemon=True).start()

    def maybe_voice_once(self):
        if sr is None:
            self.status_message = "speech_recognition kurulu değil."
            return
        r = self._build_recognizer()
        try:
            with sr.Microphone(device_index=self.preferred_mic) as src:
                self.status_message = "Dinleniyor..."
                r.adjust_for_ambient_noise(src, duration=0.8)
                audio = r.listen(src, timeout=3, phrase_time_limit=4)
            text = r.recognize_google(audio, language="tr-TR")
            self._log(f"Ses: {text}")
            self.command_queue.put(text)
        except sr.WaitTimeoutError:
            self.status_message = "Ses algılanamadı."
        except sr.UnknownValueError:
            self.status_message = "Ses anlaşılamadı."
        except Exception as e:
            self.status_message = f"Ses hatası: {e}"

    def start_continuous_voice(self):
        if sr is None or (self.voice_thread and self.voice_thread.is_alive()):
            return
        self.voice_stop_event.clear()
        self.voice_continuous = True

        def loop():
            r = self._build_recognizer()
            while self.running and not self.voice_stop_event.is_set():
                try:
                    with sr.Microphone(device_index=self.preferred_mic) as src:
                        r.adjust_for_ambient_noise(src, duration=0.2)
                        audio = r.listen(src, timeout=2.5, phrase_time_limit=3.5)
                    text = r.recognize_google(audio, language="tr-TR")
                    self._log(f"Ses: {text}")
                    self.command_queue.put(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    self._log(f"Ses döngü hatası: {e}")
                    break
            self.voice_continuous = False

        with self.voice_lock:
            self.voice_thread = threading.Thread(target=loop, daemon=True)
            self.voice_thread.start()

    def stop_continuous_voice(self):
        self.voice_stop_event.set()
        self.voice_continuous = False

    # ── Komut yürütme ────────────────────────────────────────────────────────

    def handle_command(self, raw: str):
        p = parse_command(raw)
        action = p["action"]
        if p.get("target_class"):
            self.target_class = p["target_class"]
        if p.get("target_color"):
            self.target_color = p["target_color"]
        if p.get("shape"):
            self.pending_draw = p["shape"]

        if action == "quit":
            self.running = False
        elif action == "help":
            self._print_help()
        elif action == "reset_target":
            self.target_class = self.target_color = self.pending_draw = None
            self.mark_enabled = False
            self.status_message = "Hedef sıfırlandı."
        elif action == "start_tracking":
            self._start_tracking()
        elif action == "stop_tracking":
            self.tracker.reset()
            self.status_message = "Takip durduruldu."
        elif action == "draw_circle":
            self.pending_draw = "circle"
            self.status_message = "Daire çizim modu aktif."
        elif action == "draw_rectangle":
            self.pending_draw = "rectangle"
            self.status_message = "Kare çizim modu aktif."
        elif action == "clear_drawings":
            self.pending_draw = None
            self.mark_enabled = False
            self.status_message = "Çizimler temizlendi."
        elif action == "mark_object":
            self.mark_enabled = True
            self.status_message = "Nesne işaretleme aktif."
        elif action == "accident_on":
            self.accident_enabled = True
            self.status_message = "Kaza tespiti açıldı."
        elif action == "accident_off":
            self.accident_enabled = False
            self.accident_alert = False
            self.status_message = "Kaza tespiti kapatıldı."
        elif action == "accident_threshold_up":
            self.accident_threshold = min(5.0, self.accident_threshold + 0.2)
            self.status_message = f"Kaza eşiği: {self.accident_threshold:.1f}"
        elif action == "accident_threshold_down":
            self.accident_threshold = max(1.2, self.accident_threshold - 0.2)
            self.status_message = f"Kaza eşiği: {self.accident_threshold:.1f}"
        elif action == "accident_status":
            durum = "VAR" if self.accident_alert else "YOK"
            self.status_message = f"Kaza: {durum} | skor={self.accident_score:.2f}"
        elif action == "renk_degistir":
            kaynak = p.get("target_color")   # "mavi gömleği" → kaynak=blue
            hedef  = p.get("hedef_renk")     # "yeşile çevir" → hedef=green
            sinif  = p.get("target_class")   # "arabanın" → car

            # Hedef rengi de metinden çıkar (ikinci renk kelimesi)
            if hedef is None:
                # parse_command sadece ilk rengi alıyor; ikincisini burada çıkaralım
                hedef = self._ikinci_rengi_bul(raw, kaynak)

            if kaynak and hedef:
                self.detector.renk_komutu = {
                    "hedef_sinif": sinif,
                    "kaynak_renk": kaynak,
                    "hedef_renk":  hedef,
                }
                self.status_message = (
                    f"Renk değiştirme: {kaynak} → {hedef}"
                    + (f" ({sinif})" if sinif else "")
                )
            else:
                self.status_message = "Kaynak veya hedef renk anlaşılamadı."
        elif action == "renk_iptal":
            self.detector.renk_komutu = None
            self.status_message = "Renk değiştirme kapatıldı."           
        elif action == "unknown":
            self.status_message = f"Komut anlaşılamadı: '{raw}'"

    def _start_tracking(self):
        """Son tıklanan ya da en iyi eşleşen nesneyi takibe alır."""
        if self.last_bbox is None:
            best = self._select_best_bbox()
            if best:
                self.last_bbox = best
            else:
                self.status_message = "Takip için nesneye tıklayın."
                return
        if self.last_frame is not None:
            ok = self.tracker.init(self.last_frame, self.last_bbox)
            self.status_message = "Takip başlatıldı." if ok else "Tracker başlatılamadı."

    def _select_best_bbox(self) -> tuple | None:
        """Hedef sınıf/renk varsa en iyi kutuyu otomatik seçer."""
        if not self.last_boxes or self.last_frame is None:
            return None
        best, best_score = None, -1.0
        for b in self.last_boxes:
            cls_score = 2.0 if (self.target_class and self.target_class in b["etiket"]) else 1.0
            color_score = 1.0
            if self.target_color:
                roi = self.last_frame[b["y1"]:b["y2"], b["x1"]:b["x2"]]
                color_score = self._color_score(roi, self.target_color)
            score = cls_score * 2.0 + color_score * 1.3 + b["guven"] * 0.8
            if score > best_score:
                best_score = score
                best = (b["x1"], b["y1"], b["x2"] - b["x1"], b["y2"] - b["y1"])
        return best

    def _color_score(self, roi, color: str) -> float:
        import numpy as np
        if roi is None or roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ranges = {
            "red":    [(0, 70, 40), (10, 255, 255), (160, 70, 40), (180, 255, 255)],
            "blue":   [(90, 60, 40), (130, 255, 255)],
            "green":  [(35, 50, 40), (85, 255, 255)],
            "yellow": [(20, 60, 60), (35, 255, 255)],
            "white":  [(0, 0, 170), (180, 70, 255)],
            "black":  [(0, 0, 0), (180, 255, 50)],
        }
        r = ranges.get(color, [])
        if not r:
            return 1.0
        if len(r) == 4:  # red — iki aralık
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array(r[0]), np.array(r[1])),
                cv2.inRange(hsv, np.array(r[2]), np.array(r[3])),
            )
        else:
            mask = cv2.inRange(hsv, np.array(r[0]), np.array(r[1]))
        ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1] + 1e-6)
        return min(2.0, ratio * 6.0)
    
    """Cümlede geçen ikinci renk kelimesini döndürür."""
    def _ikinci_rengi_bul(self, raw: str, ilk_renk: str | None) -> str | None:
        t = normalize(raw)
        bulunanlar = []
        for color, keywords in COLOR_KEYWORDS.items():
            for kw in keywords:
                if kw in t:
                    bulunanlar.append((t.index(kw), color))
                    break
        bulunanlar.sort()
        renkler = [c for _, c in bulunanlar]
        if len(renkler) >= 2:
            return renkler[1]
        if len(renkler) == 1 and renkler[0] != ilk_renk:
            return renkler[0]
        return None

    # ── Fare ─────────────────────────────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bbox = ObjectTracker.nokta_icindeki_bbox((x, y), self.last_boxes)
        if bbox:
            self.last_bbox = bbox
            self.status_message = f"Nesne seçildi: {bbox}"
        else:
            self.status_message = f"({x},{y}) noktasında nesne yok."

    # ── Kaza tespiti ─────────────────────────────────────────────────────────

    def _bbox_iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
        iy = max(0, min(ay+ah, by+bh) - max(ay, by))
        inter = ix * iy
        return inter / (aw*ah + bw*bh - inter + 1e-6)

    def update_accident(self):
        if not self.accident_enabled:
            self.accident_score = 0.0
            self.accident_alert = False
            return

        # Araç tespitlerini topla
        dets = [b for b in self.last_boxes if b["etiket"] in VEHICLE_CLASSES]
        assigned = set()
        for tid, tr in list(self.vehicle_tracks.items()):
            best_i, best_d = None, 1e9
            for i, d in enumerate(dets):
                if i in assigned or d["etiket"] != tr["cls"]:
                    continue
                cx, cy = (d["x1"]+d["x2"])//2, (d["y1"]+d["y2"])//2
                dist = ((cx-tr["cx"])**2 + (cy-tr["cy"])**2)**0.5
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i is not None and best_d < 120:
                d = dets[best_i]
                assigned.add(best_i)
                cx, cy = (d["x1"]+d["x2"])//2, (d["y1"]+d["y2"])//2
                spd = ((cx-tr["cx"])**2 + (cy-tr["cy"])**2)**0.5
                tr["speed"].append(spd)
                tr["cx"], tr["cy"] = cx, cy
                tr["bbox"] = (d["x1"], d["y1"], d["x2"]-d["x1"], d["y2"]-d["y1"])
                tr["miss"] = 0
            else:
                tr["miss"] += 1
            if tr["miss"] > 12:
                self.vehicle_tracks.pop(tid, None)

        for i, d in enumerate(dets):
            if i in assigned:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            cx, cy = (d["x1"]+d["x2"])//2, (d["y1"]+d["y2"])//2
            self.vehicle_tracks[tid] = {
                "cls": d["etiket"], "cx": cx, "cy": cy,
                "bbox": (d["x1"], d["y1"], d["x2"]-d["x1"], d["y2"]-d["y1"]),
                "speed": deque([0.0], maxlen=12), "miss": 0,
            }

        # Skor
        tracks = list(self.vehicle_tracks.values())
        score = 0.0
        max_iou = max(
            (self._bbox_iou(tracks[i]["bbox"], tracks[j]["bbox"])
             for i in range(len(tracks)) for j in range(i+1, len(tracks))),
            default=0.0
        )
        if max_iou > 0.12:
            score += 1.4
        for tr in tracks:
            h = list(tr["speed"])
            if len(h) >= 6:
                prev = sum(h[:-3]) / max(1, len(h[:-3]))
                now  = sum(h[-3:]) / 3.0
                if prev > 5.0 and now < prev * 0.45:
                    score += 0.8
                if now < 1.2:
                    score += 0.3

        self.accident_score = score
        if self.accident_cooldown > 0:
            self.accident_cooldown -= 1
            self.accident_alert = True
        elif score >= self.accident_threshold:
            self.accident_alert = True
            self.accident_cooldown = int(max(15, self.fps * 1.5))
            self.status_message = "Muhtemel kaza tespit edildi."
        else:
            self.accident_alert = False

    # ── Overlay çizimi ───────────────────────────────────────────────────────

    def draw_overlays(self, frame):
        h, w = frame.shape[:2]

        if self.mark_enabled and self.last_bbox:
            x, y, bw, bh = self.last_bbox
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (255, 255, 0), 2)
            cv2.putText(frame, "Isaretli", (x, max(20, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

        if self.pending_draw and self.last_bbox:
            x, y, bw, bh = self.last_bbox
            cx, cy = x + bw//2, y + bh//2
            if self.pending_draw == "circle":
                cv2.circle(frame, (cx, cy), max(20, min(bw, bh)//2), (0, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 255), 2)

        if self.accident_alert:
            cv2.putText(frame, "MUHTEMEL KAZA!", (w//5, h//6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

        # Durum satırları
        lines = [
            (self.status_message, (0, 255, 0), 0.6, 2, 24),
            (f"Fast:{'ON' if self.fast_mode else 'OFF'} | stride:{self.infer_stride} | "
             f"sinif:{self.target_class or '-'} renk:{self.target_color or '-'}",
             (200, 255, 200), 0.45, 1, 46),
            (f"{'Takip: AKTIF' if self.tracker.aktif else 'Takip: PASIF'} | "
             f"V:ses C:surekli-ses F:hiz ESC:cikis",
             (230, 230, 230), 0.45, 1, h - 10),
        ]
        for metin, renk, olcek, kalinlik, y_pos in lines:
            cv2.putText(frame, metin, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, olcek, renk, kalinlik, cv2.LINE_AA)

    # ── Ana döngü ─────────────────────────────────────────────────────────────

    def run(self):
        self._print_help()
        self.start_input_listener()
        cv2.namedWindow("Video Analizi")
        cv2.setMouseCallback("Video Analizi", self.on_mouse)

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_message = "Video bitti."
                    break

                self.frame_index += 1
                frame = utils.frame_yeniden_boyutlandir(frame)

                # YOLO — stride'a göre
                if self.last_boxes == [] or self.frame_index % self.infer_stride == 0:
                    enhanced = utils.apply_clahe(frame)
                    self.last_boxes = self.detector.detect(enhanced)

                self.last_frame = frame.copy()

                # Tespit kutularını çiz
                self.detector.draw(frame, self.last_boxes, self.target_class)

                # Tracker güncelle
                ok, _ = self.tracker.update(frame)
                if not ok and self.tracker.son_bbox is not None:
                    self.status_message = "Takip kayboldu."
                self.tracker.draw(frame)

                self.update_accident()

            except Exception as e:
                self.status_message = f"Hata: {e}"
                self._log(f"İşleme hatası: {e}")
                continue

            # Kuyruktan komutları işle
            while not self.command_queue.empty():
                self.handle_command(self.command_queue.get())

            self.draw_overlays(frame)
            cv2.imshow("Video Analizi", frame)

            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
            if key == 27:
                break
            elif key in (ord("v"), ord("V")):
                self.maybe_voice_once()
            elif key in (ord("c"), ord("C")):
                if self.voice_continuous:
                    self.stop_continuous_voice()
                else:
                    self.start_continuous_voice()
            elif key in (ord("f"), ord("F")):
                self.fast_mode = not self.fast_mode
                self.infer_stride = 2 if self.fast_mode else 1
                self.status_message = f"Hız modu: {'AÇIK' if self.fast_mode else 'KAPALI'}"

        self.running = False
        self.voice_stop_event.set()
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _print_help():
        print("""
Komutlar: takip baslat | takip durdur | daire ciz | kare ciz |
          nesneyi isaretle | hedefi sifirla | cizimi temizle |
          kaza tespitini ac/kapat | kaza var mi | q
Klavye  : V=tek ses | C=surekli ses | F=hiz modu | ESC=cikis
""")


# ── Giriş noktası ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "blue-tshirt.mp4"
    model = sys.argv[2] if len(sys.argv) > 2 else "yolov8n-seg.pt"
    try:
        VideoAssistant(video_path=video, det_model=model).run()
    except FileNotFoundError as e:
        print(f"[HATA] {e}")
        print("Kullanım: python agent.py <video.mp4> [model.pt]")