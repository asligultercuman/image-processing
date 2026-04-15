import cv2
import os
import time
import csv
from ultralytics import YOLO
from utils import apply_clahe  # Ön işleme fonksiyonunu içe aktar

# ── 1. KURULUM VE PARAMETRELER ──────────────────────────────────────────
model = YOLO('yolov8n-seg.pt')  # Test edilecek model
dataset_path = 'ExDark/ExDark/People'   # ExDark içindeki hedef klasörün yolu
output_folder = 'test_results'   # İşlenmiş resimlerin kaydedileceği yer
os.makedirs(output_folder, exist_ok=True)

# Test raporu için dosya hazırlığı
csv_file = open('performans_raporu.csv', mode='w', newline='')
fieldnames = ['dosya_adi', 'tespit_sayisi', 'islem_suresi_ms', 'ortalama_guven']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

# ── 3. TEST DÖNGÜSÜ ─────────────────────────────────────────────────────
print(f"Test basliyor: {dataset_path} klasoru taraniyor...")

resimler = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

for resim_adi in resimler:
    img_path = os.path.join(dataset_path, resim_adi)
    frame = cv2.imread(img_path)
    
    if frame is None: continue

    # Zamanlama başlat
    start_time = time.time()

    # Ön İşleme: CLAHE (Bunu kapatıp/açarak test yapabilirsin)
    enhanced_frame = apply_clahe(frame)

    # YOLO Tespiti
    results = model(enhanced_frame, verbose=False, conf=0.4)[0]
    
    # Süre hesapla (ms cinsinden)
    process_time = (time.time() - start_time) * 1000

    # Verileri Topla
    tespit_sayisi = len(results.boxes)
    guven_skorlari = results.boxes.conf.tolist() if tespit_sayisi > 0 else [0]
    avg_conf = sum(guven_skorlari) / len(guven_skorlari)

    # Raporu Yaz
    writer.writerow({
        'dosya_adi': resim_adi,
        'tespit_sayisi': tespit_sayisi,
        'islem_suresi_ms': round(process_time, 2),
        'ortalama_guven': round(avg_conf, 2)
    })

    # Görsel Kaydet (Kutuları çizilmiş haliyle)
    annotated_frame = results.plot()
    cv2.imwrite(os.path.join(output_folder, resim_adi), annotated_frame)

print(f"Test tamamlandi. Sonuclar 'performans_raporu.csv' dosyasina kaydedildi.")
csv_file.close()