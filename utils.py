import cv2

def apply_clahe(frame): 
    # 1. Görüntüyü BGR'den LAB renk uzayına çeviriyoruz
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. CLAHE objesini oluşturuyoruz, clipLimit: Kontrast sınırı, tileGridSize: Görüntünün kaç parçaya bölüneceğidir
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # 3. Sadece 'L' (parlaklık) kanalına CLAHE uyguluyoruz
    l_updated = clahe.apply(l)

    # 4. Kanalları tekrar birleştirip BGR formatına geri dönüyoruz
    updated_lab = cv2.merge((l_updated, a, b))
    result = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
    
    return result

def rengi_degistir(roi, alt_hsv, ust_hsv, yeni_renk_kodu):
    """
    Belirli bir renk aralığını hedef renge dönüştürür.
    yeni_renk_kodu: HSV uzayındaki yeni 'Hue' (Renk Tonu) değeridir.
    """
    # 1. BGR'den HSV'ye geçiş
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 2. Renk maskesi oluştur (Gömleğin o anki rengini bul)
    mask = cv2.inRange(hsv_roi, alt_hsv, ust_hsv)
    
    # 3. Maskelenen yerlerin Hue kanalını değiştir
    # H, S, V kanallarını ayır
    h, s, v = cv2.split(hsv_roi)
    
    # Sadece maskenin 255 (beyaz) olduğu yerlerde H değerini güncelle
    h[mask > 0] = yeni_renk_kodu
    
    # 4. Kanalları birleştir ve BGR'ye dön
    merged_hsv = cv2.merge([h, s, v])
    yeni_roi = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
    
    return yeni_roi