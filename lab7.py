
import time
import numpy as np
import cv2
from PIL import Image
from skimage import data


# Завантаження зображення у відтінках сірого
gray = np.array(
    Image.fromarray(data.astronaut()).convert("L"),
    dtype=np.float64
)


# Завдання 1. Функція двовимірної згортки
def conv2(img, K):
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2

    pad = np.pad(
        img,
        ((ph, ph), (pw, pw)),
        mode="edge"
    )

    out = np.zeros_like(img, dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            out += K[i, j] * pad[
                i:i + img.shape[0],
                j:j + img.shape[1]
            ]

    return out


# Створення одновимірного ядра Гауса
def gauss1d(sigma, n):
    x = np.arange(n) - n // 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


# Завдання 2. Розмиття за Гаусом
k = gauss1d(1.6, 7)
BLUR = np.outer(k, k)

blur = conv2(gray, BLUR)


# Оператор Собеля
SX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

SY = SX.T

gx = conv2(gray, SX)
gy = conv2(gray, SY)

edges = np.hypot(gx, gy)

if edges.max() > 0:
    edges = edges / edges.max() * 255

sobel = np.clip(edges, 0, 255)


# Підвищення різкості
detail = gray - blur
sharp = np.clip(gray + 1.2 * detail, 0, 255)


# Збереження зображень
Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).save(
    "original.png"
)

Image.fromarray(np.clip(blur, 0, 255).astype(np.uint8)).save(
    "blur.png"
)

Image.fromarray(sobel.astype(np.uint8)).save(
    "sobel.png"
)

Image.fromarray(sharp.astype(np.uint8)).save(
    "sharp.png"
)


# Перевірка середньої яскравості
average_kernel = np.ones((3, 3), dtype=np.float64) / 9
average_result = conv2(gray, average_kernel)

print("Завдання 1. Перевірка середньої яскравості")
print("Середня яскравість оригіналу:", gray.mean())
print("Середня яскравість результату:", average_result.mean())
print()


# Завдання 3. Роздільна згортка
def conv_sep(img, k):
    n = len(k) // 2

    # Горизонтальний прохід
    pad = np.pad(
        img,
        ((0, 0), (n, n)),
        mode="edge"
    )

    tmp = np.zeros_like(img, dtype=np.float64)

    for j, v in enumerate(k):
        tmp += v * pad[:, j:j + img.shape[1]]

    # Вертикальний прохід
    pad = np.pad(
        tmp,
        ((n, n), (0, 0)),
        mode="edge"
    )

    out = np.zeros_like(img, dtype=np.float64)

    for i, v in enumerate(k):
        out += v * pad[i:i + img.shape[0], :]

    return out


# Вимірювання найкращого часу
def best(f, r=5):
    times = []

    for _ in range(r):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)

    return min(times) * 1000


# Порівняння звичайної та роздільної згортки
print("Завдання 3. Порівняння часу")
print(f"{'Ядро':<10}{'Звичайна, мс':<18}"
      f"{'Роздільна, мс':<18}{'Виграш':<12}")

for n in (3, 5, 9, 15, 31):
    k = gauss1d(max(n / 5, 0.6), n)
    K = np.outer(k, k)

    ta = best(lambda: conv2(gray, K))
    tb = best(lambda: conv_sep(gray, k))

    difference = np.abs(
        conv2(gray, K) - conv_sep(gray, k)
    ).max()

    print(
        f"{n}x{n:<7}{ta:<18.3f}"
        f"{tb:<18.3f}{ta / tb:<12.2f}x"
    )

    print(f"Максимальна розбіжність: {difference:.3e}")

print()


# Завдання 4. Порівняння з OpenCV
n, sigma = 15, 3.0

k = gauss1d(sigma, n)

mine = conv_sep(gray, k)

lib = cv2.GaussianBlur(
    gray,
    (n, n),
    sigma,
    borderType=cv2.BORDER_REPLICATE
)

difference = np.abs(mine - lib).max()

print("Завдання 4. Порівняння з OpenCV")
print("Максимальна розбіжність:", difference)
print(f"Ваш час:  {best(lambda: conv_sep(gray, k)):.3f} мс")
print(
    f"OpenCV:   "
    f"{best(lambda: cv2.GaussianBlur(gray, (n, n), sigma, borderType=cv2.BORDER_REPLICATE)):.3f} мс"
)

print()
print("Зображення збережено: original.png, blur.png, sobel.png, sharp.png")
