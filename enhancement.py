import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('tryon1.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Konversi citra ke grayscale untuk clahe
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adjust grayscale untuk clahe
mean_val = np.mean(gray)

if mean_val < 80:
    alpha, beta = 1.2, 25
elif mean_val < 120:
    alpha, beta = 1.1, 15
elif mean_val < 180:
    alpha, beta = 1.05, 5
else:
    alpha, beta = 1.0, 0

print(f"Mean gray: {mean_val:.2f} -> alpha={alpha}, beta={beta}")

gray_auto = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# Konversi ke Gray + HE
gray_he = cv2.equalizeHist(gray)

# Konversi ke Gray + CLAHE
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray_auto)


# Warna via HSV (channel V)
# HE: asli | CLAHE: auto bright
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# HE warna (asli, tanpa auto)
v_he = cv2.equalizeHist(v)

# CLAHE warna (auto bright)
v_auto = cv2.convertScaleAbs(v, alpha=alpha, beta=beta)
v_clahe = clahe.apply(v_auto)

# Merge kembali & konversi ke RGB
hsv_he = cv2.merge([h, s, v_he])
hsv_clahe = cv2.merge([h, s, v_clahe])

rgb_he = cv2.cvtColor(hsv_he, cv2.COLOR_HSV2RGB)
rgb_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)

# Menampilkan hasi dengan frame 2x3
titles = [
    'RGB Asli', 'Gray + HE', 'Gray + CLAHE (Auto)',
    'Grayscale', 'Warna + HE', 'Warna + CLAHE (Auto)'
]

images = [
    img_rgb, gray_he, gray_clahe,
    gray, rgb_he, rgb_clahe
]

plt.figure(figsize=(12, 7))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    if i in [0, 4, 5]:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()