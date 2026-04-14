from ultralytics import YOLO
import cv2

# ── 1. Model ve video ─────────────────────────────────────────────────────────
model = YOLO('yolov8n.pt')         
cap   = cv2.VideoCapture('car-video-2.mp4')

if not cap.isOpened():
    raise FileNotFoundError("Video dosyası açılamadı: car-video.mp4")

fps    = cap.get(cv2.CAP_PROP_FPS) or 25
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height} @ {fps:.1f} fps")

# ── 2. Renk paleti (sınıf ID'sine göre otomatik renk) ────────────────────────
def sinif_rengi(sinif_id: int) -> tuple:
    """Her sınıfa tutarlı bir BGR rengi atar."""
    renkler = [
        (0, 255, 0),    (0, 200, 255),  (255, 100, 0),
        (255, 0, 200),  (0, 100, 255),  (180, 255, 0),
    ]
    return renkler[sinif_id % len(renkler)]

# ── 3. Ana döngü ──────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti.")
        break

    # --- Dikey Videolar İçin Otomatik Boyutlandırma ---
    ekran_max_yukseklik = 800  # Ekranın max yüksekliği
    h, w = frame.shape[:2]

    if h > ekran_max_yukseklik:
        oran = ekran_max_yukseklik / h
        yeni_w = int(w * oran)
        yeni_h = int(h * oran)
        frame = cv2.resize(frame, (yeni_w, yeni_h))
        # Koordinatların kaymaması için güncel yükseklik bilgisini alalım
        height, width = yeni_h, yeni_w 
    # -------------------------------------------------------

    # 3a. Tespit — sadece güven skoru >= 0.4 olan kutular
    results = model(frame, verbose=False, conf=0.4)[0]

    # 3b. Tespit edilen her nesneyi işaretle
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        sinif_id        = int(box.cls[0])
        etiket          = results.names[sinif_id]
        guven           = float(box.conf[0])
        renk            = sinif_rengi(sinif_id)

        # Kutu
        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)

        # Etiket arka planı + metin
        metin     = f"{etiket}  {guven:.0%}"
        (tw, th), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), renk, -1)
        cv2.putText(frame, metin, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # 3c. Kare bilgisi
    nesne_sayisi = len(results.boxes)
    cv2.putText(frame, f"Nesne: {nesne_sayisi}  |  ESC: cikis  S: kaydet",
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.imshow("Nesne Tespiti", frame)

    # 3d. Klavye
    tus = cv2.waitKey(int(1000 / fps)) & 0xFF
    if tus == 27:               # ESC → çıkış
        break
    elif tus == ord('s'):       # S → ekran görüntüsü
        dosya = "tespit_goruntü.png"
        cv2.imwrite(dosya, frame)
        print(f"Kaydedildi: {dosya}")

# ── 4. Temizlik ───────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()