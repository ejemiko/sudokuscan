import cv2
import numpy as np

# 1. Fotoğrafı oku ve küçült (12 MP çok büyük, 1/4'e indiriyoruz)
img = cv2.imread("data/s332501003_01.jpg")
img = cv2.resize(img, None, fx=0.25, fy=0.25)

# 2. Gri tona çevir (Harris renkle değil parlaklıkla çalışır)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 3. Harris: her piksel için bir "köşelik skoru" (R) hesaplar
R = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.04)

# 4. Skoru en yüksek skorun %1'inden büyük olan pikselleri kırmızıya boya
img[R > 0.01 * R.max()] = [0, 0, 255]

cv2.imwrite("harris_sonuc.png", img)
print("Görüntü boyutu:", gray.shape)
print("Kırmızı piksel sayısı:", (R > 0.01 * R.max()).sum())