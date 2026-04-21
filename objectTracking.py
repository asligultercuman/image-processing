from ultralytics import YOLO
import cv2

# ── 1. Model ve video ─────────────────────────────────────────────────────────
model = YOLO('yolov8n.pt')
cap   = cv2.VideoCapture('car-video.mp4')

if not cap.isOpened():
    raise FileNotFoundError("Video dosyası açılamadı: car-video-2.mp4")

fps    = cap.get(cv2.CAP_PROP_FPS) or 25
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height} @ {fps:.1f} fps")
print("Kullanım: Takip etmek istediğiniz nesneye FARE ile tıklayın.")

# ── 2. Yardımcı fonksiyonlar ──────────────────────────────────────────────────
def sinif_rengi(sinif_id: int) -> tuple:
    renkler = [
        (0, 255, 0),   (0, 200, 255), (255, 100, 0),
        (255, 0, 200), (0, 100, 255), (180, 255, 0),
    ]
    return renkler[sinif_id % len(renkler)]


def frame_yeniden_boyutlandir(frame, maks_yukseklik=800):
    h, w = frame.shape[:2]
    if h > maks_yukseklik:
        oran = maks_yukseklik / h
        frame = cv2.resize(frame, (int(w * oran), int(h * oran)))
    return frame


def yolo_ile_nesne_bul(frame, tiklanan_nokta, results):
    """
    Kullanıcının tıkladığı noktanın içinde kalan YOLO kutusunu döndürür.
    Birden fazla kutu çakışıyorsa en küçük alanı olanı seçer (en spesifik nesne).
    """
    px, py = tiklanan_nokta
    adaylar = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x1 <= px <= x2 and y1 <= py <= y2:
            alan = (x2 - x1) * (y2 - y1)
            adaylar.append((alan, (x1, y1, x2 - x1, y2 - y1)))

    if not adaylar:
        return None
    adaylar.sort(key=lambda a: a[0])   # en küçük alan → en spesifik
    return adaylar[0][1]               # (x, y, w, h)


# ── 3. Durum değişkenleri ─────────────────────────────────────────────────────
tracker        = None     # aktif CSRT tracker nesnesi
takip_bbox     = None     # son bilinen konum (x, y, w, h)
takip_aktif    = False
son_results    = None     # fare tıklamasında kullanmak için son YOLO sonucu
son_frame      = None     # fare callback'inde erişmek için


# ── 4. Fare callback ──────────────────────────────────────────────────────────
def fare_tikla(event, x, y, flags, param):
    global tracker, takip_bbox, takip_aktif

    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if son_results is None or son_frame is None:
        return

    bbox = yolo_ile_nesne_bul(son_frame, (x, y), son_results)

    if bbox is None:
        print(f"[TAKİP] ({x},{y}) noktasında tespit edilen nesne yok.")
        return

    # Yeni tracker oluştur ve başlat
    tracker = cv2.TrackerCSRT_create()
    tracker.init(son_frame, bbox)
    takip_bbox  = bbox
    takip_aktif = True
    print(f"[TAKİP] Başlatıldı → bbox: {bbox}")


cv2.namedWindow("Nesne Takibi")
cv2.setMouseCallback("Nesne Takibi", fare_tikla)


# ── 5. Ana döngü ──────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti.")
        break

    frame = frame_yeniden_boyutlandir(frame)
    h, w  = frame.shape[:2]

    # 5a. YOLO tespiti — her karede çalışır (takipte de arka planda aktif)
    results    = model(frame, verbose=False, conf=0.4)[0]
    son_results = results
    son_frame   = frame.copy()

    # 5b. Tüm tespit kutularını soluk çiz (arka plan bilgisi)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        sinif_id = int(box.cls[0])
        renk     = sinif_rengi(sinif_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 1)

    # 5c. Takip aktifse tracker'ı güncelle
    if takip_aktif and tracker is not None:
        ok, bbox = tracker.update(frame)

        if ok:
            tx, ty, tw, th_box = map(int, bbox)
            takip_bbox = (tx, ty, tw, th_box)

            # Kalın turuncu kutu — takip edilen nesne
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th_box), (0, 140, 255), 3)

            # Merkez nokta
            cx, cy = tx + tw // 2, ty + th_box // 2
            cv2.circle(frame, (cx, cy), 5, (0, 140, 255), -1)

            # Etiket
            cv2.putText(frame, "Takip ediliyor", (tx, ty - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
        else:
            takip_aktif = False
            tracker     = None
            cv2.putText(frame, "Takip kayboldu — nesneye tekrar tiklayin",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 220), 2, cv2.LINE_AA)

    # 5d. Durum bilgisi
    durum = "Takip: AKTIF" if takip_aktif else "Takip: PASIF — nesneye tiklayin"
    cv2.putText(frame, f"{durum}  |  Nesne: {len(results.boxes)}  |  ESC: cikis",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.imshow("Nesne Takibi", frame)

    tus = cv2.waitKey(int(1000 / fps)) & 0xFF
    if tus == 27:              # ESC
        break
    elif tus == ord('r'):      # R → takibi sıfırla
        tracker     = None
        takip_aktif = False
        print("[TAKİP] Sıfırlandı.")
    elif tus == ord('s'):      # S → ekran görüntüsü
        cv2.imwrite("takip_goruntu.png", frame)
        print("Kaydedildi: takip_goruntu.png")

# ── 6. Temizlik ───────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()