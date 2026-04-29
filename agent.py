import queue
import re
import sys
import threading
from datetime import datetime
from collections import deque

import cv2
from ultralytics import YOLO

try:
    import speech_recognition as sr
except Exception:
    sr = None

CLASS_KEYWORDS = {
    "person": ["insan", "kisi", "adam", "kadin", "person"],
    "car": ["araba", "otomobil", "car"],
    "truck": ["kamyon", "tir", "truck"],
    "bus": ["otobus", "bus"],
    "motorcycle": ["motor", "motosiklet", "motorcycle"],
    "bicycle": ["bisiklet", "bicycle"],
}

COLOR_KEYWORDS = {
    "red": ["kirmizi", "kizil", "red"],
    "blue": ["mavi", "lacivert", "blue"],
    "green": ["yesil", "green"],
    "yellow": ["sari", "yellow"],
    "white": ["beyaz", "white"],
    "black": ["siyah", "black"],
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# PowerShell'de Türkçe karakter bozulmasını azaltmak için.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_text_command(user_input: str) -> dict:
    """
    Türkçe/İngilizce yazılı komutları basit kurallarla eyleme çevirir.
    """
    text = normalize_command_text(user_input)
    target_class = extract_target_class(text)
    target_color = extract_target_color(text)
    wants_circle = ("daire" in text) or ("dair" in text) or ("circle" in text)
    wants_rectangle = (
        "kare" in text
        or "kara" in text
        or "dikdortgen" in text
        or "rectangle" in text
        or "square" in text
    )
    command_info = {
        "target_class": target_class,
        "target_color": target_color,
        "shape": "circle" if wants_circle else ("rectangle" if wants_rectangle else None),
    }

    if not text:
        return {"action": "noop", "message": "Boş komut."}

    if text in {"q", "quit", "exit", "cikis"}:
        return {"action": "quit", "message": "Çıkış komutu alındı."}

    if "yardim" in text or "help" in text:
        return {"action": "help", "message": "Yardım komutu alındı."}

    if (
        "hedefi sifirla" in text
        or "hedef sifirla" in text
        or "hedefi temizle" in text
        or "filtreyi temizle" in text
    ):
        return {"action": "reset_target", "message": "Hedef ve filtreler sıfırlandı.", **command_info}

    if (
        "sinirla" in text
        or "sadece" in text
        or "yalniz" in text
        or "only" in text
    ) and target_class:
        return {"action": "set_target_filter", "message": "Hedef sınıf filtresi ayarlandı.", **command_info}

    if "kaza tespitini ac" in text or "kaza modu ac" in text:
        return {"action": "accident_on", "message": "Kaza tespiti açıldı.", **command_info}

    if "kaza tespitini kapat" in text or "kaza modu kapat" in text:
        return {"action": "accident_off", "message": "Kaza tespiti kapatıldı.", **command_info}

    if "kaza var mi" in text or "kaza durumu" in text:
        return {"action": "accident_status", "message": "Kaza durumu sorgulandı.", **command_info}

    if "kaza esigini arttir" in text:
        return {"action": "accident_threshold_up", "message": "Kaza eşiği artırıldı.", **command_info}

    if "kaza esigini azalt" in text:
        return {"action": "accident_threshold_down", "message": "Kaza eşiği azaltıldı.", **command_info}

    if (
        "isaretlemeyi kapat" in text
        or "isareti kapat" in text
        or "isareti kaldir" in text
        or "isaret kapat" in text
        or "mark off" in text
    ):
        return {"action": "disable_mark", "message": "Nesne işaretleme kapatıldı.", **command_info}

    if (
        "daireyi kaldir" in text
        or "daireyi sil" in text
        or "daire kapat" in text
        or text == "daire kapat"
    ):
        return {"action": "clear_circle", "message": "Daire çizimi kaldırıldı.", **command_info}

    if (
        "kareyi kaldir" in text
        or "kareyi sil" in text
        or "dikdortgeni kaldir" in text
        or "dikdortgeni sil" in text
        or "kare kapat" in text
        or text == "kare kapat"
    ):
        return {"action": "clear_rectangle", "message": "Kare/dikdörtgen çizimi kaldırıldı.", **command_info}

    if (
        "cizimi temizle" in text
        or "cizimleri temizle" in text
        or "sekilleri temizle" in text
        or text == "temizle"
    ):
        return {"action": "clear_drawings", "message": "Tüm çizimler temizlendi.", **command_info}

    if (
        "takibi baslat" in text
        or "takip baslat" in text
        or "takibi basla" in text
        or "takibe basla" in text
        or "takip basl" in text
        or "takip et" in text
        or "takibe al" in text
        or "start tracking" in text
    ):
        return {"action": "start_tracking", "message": "Takip başlatıldı.", **command_info}

    if (
        "takibi durdur" in text
        or "takip durdur" in text
        or "takibi bitir" in text
        or "takibi kapat" in text
        or "takibi birak" in text
        or "takibi dur dur" in text
        or "stop tracking" in text
    ):
        return {"action": "stop_tracking", "message": "Takip durduruldu.", **command_info}

    if "daire ciz" in text or "daire" in text or "dair" in text or "circle" in text:
        return {"action": "draw_circle", "message": "Daire çizim komutu alındı.", **command_info}

    if (
        "kare ciz" in text
        or "kare" in text
        or "kara ciz" in text
        or "kara" in text
        or "dikdortgen ciz" in text
        or "dikdortgen" in text
        or "rectangle" in text
        or "square" in text
    ):
        return {"action": "draw_rectangle", "message": "Kare/dikdörtgen çizim komutu alındı.", **command_info}

    if "isaretle" in text or "nesneyi isaretle" in text or "mark" in text:
        return {"action": "mark_object", "message": "Nesneyi işaretleme komutu alındı.", **command_info}

    return {"action": "unknown", "message": "Komut anlaşılmadı.", **command_info}


def normalize_command_text(text: str) -> str:
    text = (text or "").strip().lower()
    tr_map = str.maketrans(
        {
            "ı": "i",
            "ğ": "g",
            "ü": "u",
            "ş": "s",
            "ö": "o",
            "ç": "c",
            "İ": "i",
            "Ğ": "g",
            "Ü": "u",
            "Ş": "s",
            "Ö": "o",
            "Ç": "c",
        }
    )
    text = text.translate(tr_map)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_target_class(text: str):
    for cls_name, keywords in CLASS_KEYWORDS.items():
        if any(k in text for k in keywords):
            return cls_name
    return None


def extract_target_color(text: str):
    for color_name, keywords in COLOR_KEYWORDS.items():
        if any(k in text for k in keywords):
            return color_name
    return None


def pick_object_at_point(point, results):
    px, py = point
    candidates = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x1 <= px <= x2 and y1 <= py <= y2:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, (x1, y1, x2 - x1, y2 - y1)))
    if not candidates:
        return None
    candidates.sort(key=lambda i: i[0])
    return candidates[0][1]


def create_tracker():
    # OpenCV kurulumuna göre farklı tracker fabrikaları mevcut olabilir.
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"):
        factory = getattr(cv2, name, None)
        if factory is not None:
            return factory()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"):
            factory = getattr(legacy, name, None)
            if factory is not None:
                return factory()
    return None


class VideoAssistant:
    def __init__(self, video_path="car-video.mp4", model_path="yolov8n.pt"):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video dosyası açılamadı: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.command_queue = queue.Queue()
        self.running = True
        self.status_message = "Hazır."

        self.last_results = None
        self.last_frame = None
        self.last_bbox = None
        self.clicked_point = None
        self.target_class = None
        self.target_color = None

        self.tracker = None
        self.tracking_active = False

        self.pending_draw = None
        self.mark_enabled = False
        self.voice_continuous = False
        self.voice_thread = None
        self.voice_stop_event = threading.Event()
        self.voice_lock = threading.Lock()
        self.preferred_mic_index = self.detect_preferred_microphone()
        self.frame_index = 0

        # Performans ayarları (CPU için varsayılan hızlı mod)
        self.fast_mode = True
        self.infer_stride = 2     # Her 2 karede 1 kez YOLO
        self.infer_imgsz = 640    # YOLO giriş boyutu
        self.accident_detection_enabled = True
        self.accident_threshold = 2.4
        self.accident_score = 0.0
        self.accident_alert = False
        self.accident_cooldown_frames = 0
        self.vehicle_tracks = {}
        self.next_track_id = 1

    def set_fast_mode(self, enabled: bool):
        self.fast_mode = enabled
        if enabled:
            self.infer_stride = 2
            self.infer_imgsz = 640
            self.status_message = "Hızlı mod: AÇIK (stride=2, imgsz=640)"
        else:
            self.infer_stride = 1
            self.infer_imgsz = 960
            self.status_message = "Hızlı mod: KAPALI (stride=1, imgsz=960)"

    def log_voice(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[SES {ts}] {message}")

    def detect_preferred_microphone(self):
        if sr is None:
            return None
        try:
            mics = sr.Microphone.list_microphone_names()
        except Exception:
            return None

        # Dizi/array mikrofonlar genellikle laptoplarda daha stabil.
        preferred_tokens = [
            "mikrofon dizisi",
            "microphone array",
            "realtek",
        ]
        for idx, name in enumerate(mics):
            low = str(name).lower()
            if any(token in low for token in preferred_tokens):
                self.log_voice(f"Mikrofon seçildi: [{idx}] {name}")
                return idx
        if mics:
            self.log_voice(f"Mikrofon seçildi (varsayılan): [0] {mics[0]}")
            return 0
        return None

    def _build_recognizer(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = False
        recognizer.energy_threshold = 220
        recognizer.pause_threshold = 0.55
        recognizer.non_speaking_duration = 0.25
        recognizer.phrase_threshold = 0.25
        return recognizer

    def start_input_listener(self):
        def input_loop():
            while self.running:
                try:
                    cmd = input("Komut (yardim için yaz): ").strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    cmd = "q"
                self.command_queue.put(cmd)
                if cmd.lower() in {"q", "quit", "exit", "cikis"}:
                    break

        th = threading.Thread(target=input_loop, daemon=True)
        th.start()

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.clicked_point = (x, y)
        if self.last_results is None or self.last_frame is None:
            self.status_message = "Henüz tespit yok, birkaç kare bekleyin."
            return
        bbox = pick_object_at_point((x, y), self.last_results)
        if bbox is None:
            self.status_message = f"({x},{y}) noktasında nesne bulunamadı."
            return
        self.last_bbox = bbox
        self.status_message = f"Nesne seçildi: {bbox}"

    def handle_command(self, raw_cmd: str):
        parsed = parse_text_command(raw_cmd)
        action = parsed["action"]
        self.status_message = parsed["message"]
        if parsed.get("target_class"):
            self.target_class = parsed["target_class"]
        if parsed.get("target_color"):
            self.target_color = parsed["target_color"]
        if parsed.get("shape"):
            self.pending_draw = parsed["shape"]

        if action == "quit":
            self.running = False
            return

        if action == "help":
            print(
                "\nKomutlar: takip baslat | takip durdur | daire ciz | kare ciz | "
                "bu nesneyi isaretle | hedefi sifirla | sadece araba ile sinirla |\n"
                "kaza tespitini ac/kapat | kaza var mi | kaza esigini arttir/azalt | q\n"
            )
            print("Klavye: C(sürekli ses) | V(tek ses) | F(hız modu) | ESC")
            return

        if action == "accident_on":
            self.accident_detection_enabled = True
            self.status_message = "Kaza tespiti açıldı."
            return

        if action == "accident_off":
            self.accident_detection_enabled = False
            self.accident_alert = False
            self.status_message = "Kaza tespiti kapatıldı."
            return

        if action == "accident_status":
            state = "VAR" if self.accident_alert else "YOK"
            self.status_message = f"Kaza alarmı: {state} | skor={self.accident_score:.2f}"
            return

        if action == "accident_threshold_up":
            self.accident_threshold = min(5.0, self.accident_threshold + 0.2)
            self.status_message = f"Kaza eşiği: {self.accident_threshold:.1f}"
            return

        if action == "accident_threshold_down":
            self.accident_threshold = max(1.2, self.accident_threshold - 0.2)
            self.status_message = f"Kaza eşiği: {self.accident_threshold:.1f}"
            return

        if action == "reset_target":
            self.target_class = None
            self.target_color = None
            self.pending_draw = None
            self.mark_enabled = False
            return

        if action == "set_target_filter":
            if self.target_class is None:
                self.status_message = "Filtre için sınıf bulunamadı. Örn: sadece araba ile sinirla."
                return
            self.status_message = f"Hedef sınıf filtresi aktif: {self.target_class}"
            return

        if action == "disable_mark":
            self.mark_enabled = False
            return

        if action == "clear_circle":
            if self.pending_draw == "circle":
                self.pending_draw = None
            return

        if action == "clear_rectangle":
            if self.pending_draw == "rectangle":
                self.pending_draw = None
            return

        if action == "clear_drawings":
            self.pending_draw = None
            self.mark_enabled = False
            return

        if action == "start_tracking":
            if self.last_bbox is None:
                # Kullanıcı tıklamadıysa komuttaki hedefe göre otomatik seçim.
                best_box = self.select_best_bbox(
                    self.last_results, self.last_frame, self.target_class, self.target_color
                )
                if best_box is not None:
                    self.last_bbox = best_box
                    self.status_message = "Hedef nesne otomatik seçildi, takip başlatılıyor."
                else:
                    self.status_message = "Önce videodan bir nesneye tıklayın."
                    return
            tracker = create_tracker()
            if tracker is None:
                self.status_message = "Tracker yok. `opencv-contrib-python` kurun."
                return
            tracker.init(self.last_frame, self.last_bbox)
            self.tracker = tracker
            self.tracking_active = True
            return

        if action == "stop_tracking":
            self.tracker = None
            self.tracking_active = False
            return

        if action == "draw_circle":
            self.pending_draw = "circle"
            return

        if action == "draw_rectangle":
            self.pending_draw = "rectangle"
            return

        if action == "mark_object":
            self.mark_enabled = True
            return

        if "sesli modu baslat" in (raw_cmd or "").lower():
            self.start_continuous_voice_mode()
            return

        if "sesli modu durdur" in (raw_cmd or "").lower():
            self.stop_continuous_voice_mode()
            return

    def maybe_read_voice_command(self):
        # V tuşu ile tek seferlik sesli komut.
        if sr is None:
            self.status_message = "speech_recognition kurulu değil."
            return
        recognizer = self._build_recognizer()
        try:
            with sr.Microphone(device_index=self.preferred_mic_index) as source:
                self.status_message = "Dinleniyor..."
                self.log_voice("Tek seferlik dinleme başladı.")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
            text = recognizer.recognize_google(audio, language="tr-TR")
            self.log_voice(f"Algılanan komut: {text}")
            self.command_queue.put(text)
            self.status_message = f"Sesli komut: {text}"
        except sr.WaitTimeoutError:
            self.status_message = "Ses algılanamadı (zaman aşımı)."
            self.log_voice("Ses algılanamadı (timeout).")
        except sr.UnknownValueError:
            self.status_message = "Ses anlaşılamadı."
            self.log_voice("Ses anlaşılamadı.")
        except sr.RequestError as e:
            self.status_message = "Konuşma servisi hatası."
            self.log_voice(f"Servis hatası: {e}")
        except Exception as e:
            self.status_message = f"Sesli komut alınamadı: {e}"
            self.log_voice(f"Beklenmeyen hata: {e}")

    def start_continuous_voice_mode(self):
        if sr is None:
            self.status_message = "speech_recognition kurulu değil."
            return

        with self.voice_lock:
            if self.voice_thread is not None and self.voice_thread.is_alive():
                self.status_message = "Sürekli ses modu zaten aktif."
                return
            self.voice_stop_event.clear()
            self.voice_continuous = True
            self.status_message = "Sürekli ses modu aktif."
            self.log_voice("Sürekli ses modu başlatıldı.")

        def loop():
            recognizer = self._build_recognizer()
            while self.running and not self.voice_stop_event.is_set():
                try:
                    with sr.Microphone(device_index=self.preferred_mic_index) as source:
                        # Her döngüde kalibrasyon yapmak yerine bir kez kısa kalibrasyon.
                        recognizer.adjust_for_ambient_noise(source, duration=0.2)
                        self.log_voice("Dinleniyor...")
                        audio = recognizer.listen(source, timeout=2.5, phrase_time_limit=3.5)
                    text = recognizer.recognize_google(audio, language="tr-TR")
                    self.log_voice(f"Algılanan komut: {text}")
                    self.command_queue.put(text)
                except sr.WaitTimeoutError:
                    # Sessizlikte spam log olmaması için geçiyoruz.
                    continue
                except sr.UnknownValueError:
                    self.log_voice("Ses anlaşılamadı.")
                except sr.RequestError as e:
                    self.log_voice(f"Servis hatası: {e}")
                    self.status_message = "İnternet/servis hatası (ses)."
                    break
                except Exception as e:
                    self.log_voice(f"Sürekli mod hatası: {e}")
                    break

            with self.voice_lock:
                self.voice_continuous = False
                self.voice_thread = None
            self.log_voice("Sürekli ses modu durdu.")

        with self.voice_lock:
            self.voice_thread = threading.Thread(target=loop, daemon=True)
            self.voice_thread.start()

    def stop_continuous_voice_mode(self):
        with self.voice_lock:
            if not self.voice_continuous and (self.voice_thread is None or not self.voice_thread.is_alive()):
                self.status_message = "Sürekli ses modu zaten kapalı."
                return
        self.voice_stop_event.set()
        with self.voice_lock:
            self.voice_continuous = False
        self.status_message = "Sürekli ses modu kapatıldı."
        self.log_voice("Sürekli ses modu kapatılıyor...")
        th = self.voice_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.2)

    def draw_overlays(self, frame):
        h, w = frame.shape[:2]

        if self.last_bbox is not None and self.mark_enabled:
            x, y, bw, bh = map(int, self.last_bbox)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
            cv2.putText(
                frame,
                "Isaretli Nesne",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.pending_draw and self.last_bbox is not None:
            x, y, bw, bh = map(int, self.last_bbox)
            cx, cy = x + bw // 2, y + bh // 2
            if self.pending_draw == "circle":
                radius = max(20, min(bw, bh) // 2)
                cv2.circle(frame, (cx, cy), radius, (0, 255, 255), 2)
            elif self.pending_draw == "rectangle":
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

        status = "Takip: AKTIF" if self.tracking_active else "Takip: PASIF"
        cv2.putText(
            frame,
            f"{status} | F hiz modu | V tek | C surekli-ses | ESC cikis",
            (10, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        perf_text = f"Fast:{'ON' if self.fast_mode else 'OFF'} | stride:{self.infer_stride} | imgsz:{self.infer_imgsz}"
        cv2.putText(
            frame,
            perf_text,
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 255, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self.status_message,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        target_text = f"Hedef: sinif={self.target_class or '-'} renk={self.target_color or '-'}"
        cv2.putText(
            frame,
            target_text,
            (10, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
        # Kaza durum bilgisini sürekli göstermiyoruz; ekranı sade tutuyoruz.
        if self.accident_alert:
            cv2.putText(
                frame,
                "MUHTEMEL KAZA TESPIT EDILDI!",
                (max(20, w // 5), max(60, h // 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

    def select_best_bbox(self, results, frame, target_class=None, target_color=None):
        if results is None or frame is None or len(results.boxes) == 0:
            return None
        best = None
        best_score = -1.0
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue
            cls_id = int(box.cls[0])
            cls_name = str(results.names.get(cls_id, cls_id)).lower()
            conf = float(box.conf[0])
            area = float((x2 - x1) * (y2 - y1))

            # Class puanı
            class_score = 1.0
            if target_class:
                class_score = 2.0 if target_class in cls_name else 0.1

            # Renk puanı
            color_score = 1.0
            if target_color:
                roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                color_score = self.compute_color_score(roi, target_color)

            score = (class_score * 2.0) + (color_score * 1.3) + (conf * 0.8) + (area / 100000.0)
            if score > best_score:
                best_score = score
                best = (x1, y1, x2 - x1, y2 - y1)
        return best

    def compute_color_score(self, roi, color_name: str) -> float:
        if roi is None or roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if color_name == "red":
            mask1 = cv2.inRange(hsv, (0, 70, 40), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (160, 70, 40), (180, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name == "blue":
            mask = cv2.inRange(hsv, (90, 60, 40), (130, 255, 255))
        elif color_name == "green":
            mask = cv2.inRange(hsv, (35, 50, 40), (85, 255, 255))
        elif color_name == "yellow":
            mask = cv2.inRange(hsv, (20, 60, 60), (35, 255, 255))
        elif color_name == "white":
            mask = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))
        elif color_name == "black":
            mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        else:
            return 1.0
        ratio = float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1] + 1e-6)
        return min(2.0, max(0.0, ratio * 6.0))

    def bbox_iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = (aw * ah) + (bw * bh) - inter + 1e-6
        return inter / union

    def _collect_vehicle_detections(self, results):
        detections = []
        if results is None:
            return detections
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = str(results.names.get(cls_id, cls_id)).lower()
            if cls_name not in VEHICLE_CLASSES:
                continue
            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            detections.append({"bbox": bbox, "center": (cx, cy), "cls": cls_name})
        return detections

    def _update_vehicle_tracks(self, detections):
        assigned = set()
        for tid, tr in list(self.vehicle_tracks.items()):
            prev_cx, prev_cy = tr["center"]
            best_idx = None
            best_dist = 1e9
            for i, det in enumerate(detections):
                if i in assigned or det["cls"] != tr["cls"]:
                    continue
                cx, cy = det["center"]
                dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None and best_dist < 120:
                det = detections[best_idx]
                assigned.add(best_idx)
                cx, cy = det["center"]
                speed = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                tr["speed_hist"].append(speed)
                tr["center"] = det["center"]
                tr["bbox"] = det["bbox"]
                tr["miss"] = 0
            else:
                tr["miss"] += 1
            if tr["miss"] > 12:
                self.vehicle_tracks.pop(tid, None)

        for i, det in enumerate(detections):
            if i in assigned:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            self.vehicle_tracks[tid] = {
                "cls": det["cls"],
                "center": det["center"],
                "bbox": det["bbox"],
                "speed_hist": deque([0.0], maxlen=12),
                "miss": 0,
            }

    def _compute_accident_score(self):
        score = 0.0
        tracks = list(self.vehicle_tracks.values())
        max_iou = 0.0
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                iou = self.bbox_iou(tracks[i]["bbox"], tracks[j]["bbox"])
                max_iou = max(max_iou, iou)
        if max_iou > 0.12:
            score += 1.4

        for tr in tracks:
            hist = list(tr["speed_hist"])
            if len(hist) < 6:
                continue
            prev_avg = sum(hist[:-3]) / max(1, len(hist[:-3]))
            now_avg = sum(hist[-3:]) / 3.0
            if prev_avg > 5.0 and now_avg < (prev_avg * 0.45):
                score += 0.8
            if now_avg < 1.2:
                score += 0.3
        return score

    def update_accident_detection(self, results):
        if not self.accident_detection_enabled:
            self.accident_score = 0.0
            self.accident_alert = False
            return

        detections = self._collect_vehicle_detections(results)
        self._update_vehicle_tracks(detections)
        self.accident_score = self._compute_accident_score()

        if self.accident_cooldown_frames > 0:
            self.accident_cooldown_frames -= 1
            self.accident_alert = True
            return

        if self.accident_score >= self.accident_threshold:
            self.accident_alert = True
            self.accident_cooldown_frames = int(max(15, self.fps * 1.5))
            self.status_message = "Muhtemel kaza tespit edildi."
        else:
            self.accident_alert = False

    def run(self):
        print("Sistem başlatıldı. Nesneye fare ile tıklayın, sonra komut verin.")
        print("Yazılı komut örnekleri: takip baslat, takip durdur, daire ciz, kare ciz, bu nesneyi isaretle")
        print("V: tek sefer sesli komut | C: sürekli ses modu aç/kapat | F: hız modu | ESC: çıkış")

        self.start_input_listener()
        cv2.namedWindow("Komutlu Video Analizi")
        cv2.setMouseCallback("Komutlu Video Analizi", self.on_mouse)

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_message = "Video bitti."
                    break

                self.frame_index += 1
                run_detection = (self.last_results is None) or (self.frame_index % self.infer_stride == 0)

                if run_detection:
                    results = self.model(frame, verbose=False, conf=0.4, imgsz=self.infer_imgsz)[0]
                    self.last_results = results
                else:
                    results = self.last_results

                self.last_frame = frame.copy()
                self.update_accident_detection(results)

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = results.names[cls_id]
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.0%}",
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 200, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if self.tracking_active and self.tracker is not None:
                    ok, bbox = self.tracker.update(frame)
                    if ok:
                        tx, ty, tw, th = map(int, bbox)
                        self.last_bbox = (tx, ty, tw, th)
                        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 140, 255), 3)
                        cv2.putText(
                            frame,
                            "Takip ediliyor",
                            (tx, max(22, ty - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 140, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        self.tracking_active = False
                        self.tracker = None
                        self.status_message = "Takip kayboldu. Yeniden nesne seçin."
            except Exception as e:
                self.status_message = f"Kare işleme hatası: {e}"
                self.log_voice(f"Video işleme hatası: {e}")
                continue

            while not self.command_queue.empty():
                self.handle_command(self.command_queue.get())

            self.draw_overlays(frame)
            cv2.imshow("Komutlu Video Analizi", frame)

            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
            if key == 27:  # ESC
                break
            if key in (ord("v"), ord("V")):
                self.maybe_read_voice_command()
            if key in (ord("c"), ord("C")):
                if self.voice_continuous:
                    self.stop_continuous_voice_mode()
                else:
                    self.start_continuous_voice_mode()
            if key in (ord("f"), ord("F")):
                self.set_fast_mode(not self.fast_mode)

        self.running = False
        self.voice_stop_event.set()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_arg = sys.argv[1] if len(sys.argv) > 1 else "car-video.mp4"
    try:
        app = VideoAssistant(video_path=video_arg, model_path="yolov8n.pt")
        app.run()
    except FileNotFoundError as e:
        print(f"[HATA] {e}")
        print("Kullanım: python agent.py <video_dosyasi.mp4>")