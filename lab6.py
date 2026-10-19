# lab6.py
# Лабораторна робота 6 — JPEG-lite
# Спеціальність 122 Комп'ютерні науки, група КН-3/1
# Python + NumPy

import os
import zlib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import data
from skimage.metrics import structural_similarity


OUT_DIR = "lab6_results"
os.makedirs(OUT_DIR, exist_ok=True)

N = 8

Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


# Завдання 1. Матриця базису DCT
def dct_matrix(N=8):
    D = np.zeros((N, N), dtype=np.float64)

    for k in range(N):
        a = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)

        for n in range(N):
            D[k, n] = (
                a * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
            )

    return D


D = dct_matrix(N)


def dct2(block):
    return D @ block @ D.T


def idct2(coeffs):
    return D.T @ coeffs @ D


# Завдання 2. Таблиця квантування
def qtable(q):
    q = max(1, min(99, int(q)))

    S = 5000 / q if q < 50 else 200 - 2 * q

    return np.clip(
        np.floor((S * Q50 + 50) / 100),
        1,
        255
    )


# Завдання 3. Розбиття зображення на блоки 8x8
def blockify(img):
    h, w = img.shape

    if h % 8 != 0 or w % 8 != 0:
        raise ValueError(
            "Розміри зображення мають бути кратними 8"
        )

    return (
        img.reshape(h // 8, 8, w // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )


def unblockify(blocks, h, w):
    return (
        blocks.reshape(h // 8, w // 8, 8