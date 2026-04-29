# Proje Tamamlama Listesi (Sıralı)

## 1) Ortam ve Bağımlılıklar
- [x] `opencv-python` kurulu
- [x] `ultralytics` kurulu
- [x] `SpeechRecognition` kurulu
- [ ] Video dosyası klasörde mevcut (`car-video.mp4` veya kendi videon)

## 2) Zorunlu Görüntü Analizi
- [x] Nesne tespiti (YOLO)
- [x] Nesne takibi (tracker)

## 3) Zorunlu Komut Sistemi (Sesli veya Yazılı)
- [x] Yazılı komut desteği
- [x] Komutla eylem tetikleme:
  - [x] `takip baslat`
  - [x] `takip durdur`
  - [x] `daire ciz`
  - [x] `kare ciz`
  - [x] `bu nesneyi isaretle`
- [x] Sesli komut denemesi (V tuşu ile tek seferlik dinleme)

## 4) Demo Akışı (Sunum için)
- [ ] Senaryo 1: Nesne seç → `takip baslat`
- [ ] Senaryo 2: `bu nesneyi isaretle` + `daire ciz`/`kare ciz`
- [ ] Senaryo 3: `takip durdur`

## 5) Teslim Dokümanı
- [ ] Kullanılan teknolojiler
- [ ] Mimari akış diyagramı (girdi -> analiz -> komut -> çıktı)
- [ ] Demo ekran görüntüleri
- [ ] Karşılaşılan sorunlar ve çözümler

## Çalıştırma
```bash
python agent.py
```

Varsayılan video: `car-video.mp4`
