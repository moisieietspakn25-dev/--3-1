import numpy as np
import os


OUTPUT_DIR = "lab2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Гама-корекція sRGB
# --------------------------------------------------

def to_light(code):
    """Перетворення sRGB-коду [0..255] у лінійну яскравість [0..1]."""
    c = np.asarray(code, dtype=np.float64) / 255.0
    return np.where(
        c <= 0.04045,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4
    )


def to_code(light):
    """Перетворення лінійної яскравості [0..1] у sRGB-код [0..255]."""
    light = np.clip(np.asarray(light, dtype=np.float64), 0.0, 1.0)
    c = np.where(
        light <= 0.0031308,
        12.92 * light,
        1.055 * (light ** (1.0 / 2.4)) - 0.055
    )
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------
# 2. Запис BMP 24-bit без сторонніх бібліотек
# --------------------------------------------------

def write_bmp24(filename, img):
    """Зберігає RGB-зображення або grayscale-зображення у BMP."""
    img = np.asarray(img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Зображення повинно мати форму HxWx3 або HxW.")

    img = np.clip(img, 0, 255).astype(np.uint8)
    height, width, _ = img.shape
    row_size = (width * 3 + 3) & ~3
    padding = b"\x00" * (row_size - width * 3)
    pixel_data = bytearray()

    for y in range(height - 1, -1, -1):
        row = img[y, :, ::-1].tobytes()
        pixel_data.extend(row)
        pixel_data.extend(padding)

    file_size = 54 + len(pixel_data)

    import struct

    with open(filename, "wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))
        f.write(struct.pack("<HHI", 0, 0, 54))
        f.write(struct.pack("<IiiHHIIiiII",
                            40, width, height, 1, 24, 0,
                            len(pixel_data), 2835, 2835, 0, 0))
        f.write(pixel_data)


# --------------------------------------------------
# 3. Створення шахівниці та зменшення зображення
# --------------------------------------------------

def make_board(size=512, cell=8):
    """Створює чорно-білу шахівницю."""
    y, x = np.indices((size, size))
    board = ((x // cell + y // cell) % 2) * 255
    return np.stack([board, board, board], axis=2).astype(np.uint8)


def box_average(img, k):
    """Зменшує зображення усередненням блоків k x k."""
    height, width = img.shape[:2]

    if height % k != 0 or width % k != 0:
        raise ValueError("Розміри зображення повинні ділитися на k.")

    h, w = height // k, width // k
    channels = 1 if img.ndim == 2 else img.shape[2]

    if img.ndim == 2:
        blocks = img.reshape(h, k, w, k)
        return blocks.mean(axis=(1, 3))

    blocks = img.reshape(h, k, w, k, channels)
    return blocks.mean(axis=(1, 3))


def save_grayscale(filename, img):
    """Зберігає grayscale-зображення у BMP."""
    write_bmp24(os.path.join(OUTPUT_DIR, filename), img)


def test_gamma_and_downsampling():
    board = make_board(512, 8)

    # Наївне усереднення без лінеаризації
    naive = box_average(board.astype(np.float64), 16)
    naive = np.clip(np.round(naive), 0, 255).astype(np.uint8)

    # Правильне усереднення у лінійному просторі світла
    linear = to_light(board)
    correct_linear = box_average(linear, 16)
    correct = to_code(correct_linear)

    save_grayscale("board_naive.bmp", naive[:, :, 0])
    save_grayscale("board_linear_correct.bmp", correct[:, :, 0])

    naive_light = np.mean(to_light(naive))
    correct_light = np.mean(to_light(correct))

    print("\n=== Гама-корекція та зменшення ===")
    print(f"Середній код наївного зображення: {naive.mean():.2f}")
    print(f"Середня фізична яскравість наївного: {naive_light * 100:.2f}%")
    print(f"Середній код правильного зображення: {correct.mean():.2f}")
    print(f"Середня фізична яскравість правильного: {correct_light * 100:.2f}%")
    print("Збережено board_naive.bmp та board_linear_correct.bmp")


# --------------------------------------------------
# 4. RGB та YCbCr
# --------------------------------------------------

def rgb_to_ycbcr(img):
    """Перетворення RGB у YCbCr."""
    rgb = img.astype(np.float64)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return np.stack([y, cb, cr], axis=2)


def ycbcr_to_rgb(img):
    """Перетворення YCbCr у RGB."""
    y, cb, cr = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)

    rgb = np.stack([r, g, b], axis=2)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def test_ycbcr():
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

    ycbcr = rgb_to_ycbcr(original)
    restored = ycbcr_to_rgb(ycbcr)

    diff = np.abs(original.astype(np.int16) - restored.astype(np.int16))
    print("\n=== RGB та YCbCr ===")
    print(f"Максимальна різниця після перетворення: {diff.max()}")
    print(f"Середня абсолютна різниця: {diff.mean():.4f}")


# --------------------------------------------------
# 5. Читання PPM P6
# --------------------------------------------------

def read_ppm(filename):
    """Читає PPM-зображення формату P6."""
    with open(filename, "rb") as f:
        if f.readline().strip() != b"P6":
            raise ValueError("Підтримується лише PPM P6.")

        def next_header_line():
            line = f.readline()
            while line.startswith(b"#") or not line.strip():
                line = f.readline()
            return line

        width, height = map(int, next_header_line().split())
        max_value = int(next_header_line())

        if max_value != 255:
            raise ValueError("Підтримуються PPM-файли з max_value = 255.")

        data = f.read(width * height * 3)
        if len(data) != width * height * 3:
            raise ValueError("Файл PPM пошкоджений або неповний.")

    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()


# --------------------------------------------------
# 6. HSV
# --------------------------------------------------

def rgb_to_hsv(img):
    """Перетворення RGB [0..255] у HSV: H [0..360], S,V [0..1]."""
    rgb = img.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc.copy()

    nonzero = maxc != 0
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    mask = delta != 0
    mr = mask & (maxc == r)
    mg = mask & (maxc == g)
    mb = mask & (maxc == b)

    h[mr] = 60.0 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    h[mg] = 60.0 * (((b[mg] - r[mg]) / delta[mg]) + 2)
    h[mb] = 60.0 * (((r[mb] - g[mb]) / delta[mb]) + 4)

    return np.stack([h, s, v], axis=2)


def hsv_to_rgb(hsv):
    """Перетворення HSV у RGB [0..255]."""
    h = hsv[:, :, 0] % 360.0
    s = np.clip(hsv[:, :, 1], 0.0, 1.0)
    v = np.clip(hsv[:, :, 2], 0.0, 1.0)

    c = v * s
    x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
    m = v - c

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    masks = [
        (h < 60),
        (h >= 60) & (h < 120),
        (h >= 120) & (h < 180),
        (h >= 180) & (h < 240),
        (h >= 240) & (h < 300),
        (h >= 300)
    ]

    values = [
        (c, x, np.zeros_like(h)),
        (x, c, np.zeros_like(h)),
        (np.zeros_like(h), c, x),
        (np.zeros_like(h), x, c),
        (x, np.zeros_like(h), c),
        (c, np.zeros_like(h), x)
    ]

    for mask, (rr, gg, bb) in zip(masks, values):
        r[mask] = rr[mask]
        g[mask] = gg[mask]
        b[mask] = bb[mask]

    rgb = np.stack([r + m, g + m, b + m], axis=2)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def test_hsv(img):
    hsv = rgb_to_hsv(img)

    # Зменшення насиченості у 0.3 раза
    low_sat = hsv.copy()
    low_sat[:, :, 1] *= 0.3
    write_bmp24(os.path.join(OUTPUT_DIR, "hsv_low_saturation.bmp"),
                hsv_to_rgb(low_sat))

    # Збільшення насиченості у 2 рази з обмеженням до 1
    high_sat = hsv.copy()
    high_sat[:, :, 1] = np.clip(high_sat[:, :, 1] * 2.0, 0.0, 1.0)
    write_bmp24(os.path.join(OUTPUT_DIR, "hsv_high_saturation.bmp"),
                hsv_to_rgb(high_sat))

    # Зміщення відтінку на 60 градусів
    shifted_hue = hsv.copy()
    shifted_hue[:, :, 0] = (shifted_hue[:, :, 0] + 60.0) % 360.0
    write_bmp24(os.path.join(OUTPUT_DIR, "hsv_shifted_hue.bmp"),
                hsv_to_rgb(shifted_hue))

    restored = hsv_to_rgb(hsv)
    diff = np.abs(img.astype(np.int16) - restored.astype(np.int16))

    print("\n=== HSV ===")
    print(f"Максимальна різниця RGB → HSV → RGB: {diff.max()}")
    print("Збережено три зображення з ефектами HSV.")


# --------------------------------------------------
# 7. Аналіз колірних просторів
# --------------------------------------------------

def analyze_image(img):
    ycbcr = rgb_to_ycbcr(img)

    # Канал Y
    y = np