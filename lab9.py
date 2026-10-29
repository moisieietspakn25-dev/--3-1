
import numpy as np
from PIL import Image
from pathlib import Path

OUTPUT = Path("bc1_results")
OUTPUT.mkdir(exist_ok=True)


# ==========================================
# ЗАВДАННЯ 1. ПАЛІТРА З ЧОТИРЬОХ КОЛЬОРІВ
# ==========================================

def make_palette(c0, c1):
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)

    return np.stack([
        c0,
        c1,
        (2 * c0 + c1) / 3,
        (c0 + 2 * c1) / 3
    ])


# Перетворення RGB у RGB565 з відновленням у RGB
def to565(color):
    color = np.asarray(color, dtype=float)

    r = np.round(color[..., 0] * 31 / 255) * 255 / 31
    g = np.round(color[..., 1] * 63 / 255) * 255 / 63
    b = np.round(color[..., 2] * 31 / 255) * 255 / 31

    return np.stack([r, g, b], axis=-1)


# ==========================================
# ЗАВДАННЯ 2. ВИБІР ОПОРНИХ КОЛЬОРІВ
# ТА КОДУВАННЯ БЛОКА 4x4
# ==========================================

def pick_endpoints(blk, mode="bbox"):
    blk = np.asarray(blk, dtype=float)

    if mode == "bbox":
        return blk.max(axis=0), blk.min(axis=0)

    elif mode == "pixel":
        luminance = blk @ np.array([0.299, 0.587, 0.114])
        return blk[luminance.argmax()], blk[luminance.argmin()]

    else:
        raise ValueError("Режим має бути bbox або pixel")


def encode_block(blk, mode="bbox"):
    blk = np.asarray(blk, dtype=float).reshape(16, 3)

    c0, c1 = pick_endpoints(blk, mode)

    palette = make_palette(to565(c0), to565(c1))

    distances = (
        (blk[:, None, :] - palette[None, :, :]) ** 2
    ).sum(axis=-1)

    indices = distances.argmin(axis=1)

    return palette, indices


# Додатковий метод: обираємо найкращий із двох
def encode_best(blk):
    best = None

    for mode in ("bbox", "pixel"):
        palette, indices = encode_block(blk, mode)
        error = ((palette[indices] - blk) ** 2).sum()

        if best is None or error < best[0]:
            best = (error, palette, indices)

    return best[1], best[2]


# ==========================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================

def prepare_image(image):
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))

    image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    if image.shape[2] == 4:
        image = image[..., :3]

    h, w = image.shape[:2]
    h = h // 4 * 4
    w = w // 4 * 4

    if h < 4 or w < 4:
        raise ValueError("Зображення має бути не менше 4x4")

    return image[:h, :w, :3].astype(np.uint8)


# ==========================================
# ЗАВДАННЯ 3. КОДУВАННЯ ВСЬОГО ЗОБРАЖЕННЯ
# ==========================================

def bc1(image, mode="bbox"):
    image = prepare_image(image).astype(float)

    h, w, _ = image.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(0, h, 4):
        for x in range(0, w, 4):
            block = image[y:y+4, x:x+4].reshape(16, 3)

            if mode == "best":
                palette, indices = encode_best(block)
            else:
                palette, indices = encode_block(block, mode)

            decoded = palette[indices].reshape(4, 4, 3)

            result[y:y+4, x:x+4] = np.clip(
                np.round(decoded), 0, 255
            ).astype(np.uint8)

    return result


def stats(original, reconstructed):
    original = prepare_image(original).astype(float)
    reconstructed = prepare_image(reconstructed).astype(float)

    difference = original - reconstructed

    mae = np.mean(np.abs(difference))
    mse = np.mean(difference ** 2)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def save_difference(original, reconstructed, filename):
    original = prepare_image(original).astype(float)
    reconstructed = prepare_image(reconstructed).astype(float)

    difference = np.abs(original - reconstructed)
    difference = np.clip(difference * 3, 0, 255).astype(np.uint8)

    Image.fromarray(difference).save(filename)


# ==========================================
# ЗАВДАННЯ 4. КУТОВА ПОХИБКА НОРМАЛЬ-МАПИ
# ТА ПОШУК ЗОБРАЖЕНЬ, ДЕ BC1 ЛАМАЄТЬСЯ
# ==========================================

def angle_error(original, reconstructed):
    original = prepare_image(original).astype(float) / 127.5 - 1
    reconstructed = prepare_image(reconstructed).astype(float) / 127.5 - 1

    original /= np.maximum(
        np.linalg.norm(original, axis=2, keepdims=True), 1e-12
    )
    reconstructed /= np.maximum(
        np.linalg.norm(reconstructed, axis=2, keepdims=True), 1e-12
    )

    dot = np.sum(original * reconstructed, axis=2)
    dot = np.clip(dot, -1, 1)

    return np.degrees(np.arccos(dot))


def create_test_images():
    size = 96
    y, x = np.mgrid[0:size, 0:size]

    # 1. Фотографія-подібне синтетичне зображення
    rng = np.random.default_rng(9)

    base = (
        110
        + 45 * np.sin(x / 11)
        + 35 * np.cos(y / 13)
    )

    photo = np.stack([
        base + 24 * np.sin((x + y) / 17),
        base * 0.88 + 20 * np.cos(x / 9),
        base * 0.72 + 18 * np.sin(y / 8)
    ], axis=-1)

    photo += rng.normal(0, 5, photo.shape)
    photo = np.clip(photo, 0, 255).astype(np.uint8)

    # 2. Градієнт
    gradient = np.stack([
        x * 255 / (size - 1),
        y * 255 / (size - 1),
        (x + y) * 255 / (2 * (size - 1))
    ], axis=-1).astype(np.uint8)

    # 3. Нормал-мапа
    nx = (x - size / 2) / (size / 2)
    ny = (y - size / 2) / (size / 2)
    nz = np.sqrt(np.maximum(
        0.05, 1 - np.minimum(0.95, nx**2 + ny**2)
    ))

    normals = np.stack([nx, -ny, nz], axis=-1)
    normals /= np.maximum(
        np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12
    )

    normal_map = np.clip(
        (normals + 1) * 127.5, 0, 255
    ).astype(np.uint8)

    # 4. Різкі межі кольорів
    stripes = np.zeros((size, size, 3), dtype=np.uint8)
    stripes[:, :37] = [255, 0, 0]
    stripes[:, 37:74] = [0, 255, 0]
    stripes[:, 74:] = [0, 0, 255]

    return {
        "photo": photo,
        "gradient": gradient,
        "normal_map": normal_map,
        "sharp_edges": stripes
    }


def run_tests():
    print("ПЕРЕВІРКА ЗАВДАНЬ")

    # Тест завдання 1
    palette = make_palette(
        [255, 255, 255],
        [0, 0, 0]
    )

    print("1. Палітра:", palette[:, 0].round().astype(int))
    assert np.allclose(
        palette[:, 0], [255, 0, 170, 85]
    )

    # Тест завдання 2: червоні та зелені пікселі
    block = np.array(
        [[255, 0, 0]] * 8 + [[0, 255, 0]] * 8,
        dtype=float
    )

    errors = {}

    for mode in ("bbox", "pixel"):
        palette, indices = encode_block(block, mode)
        errors[mode] = np.sqrt(
            np.mean(np.sum(
                (palette[indices] - block) ** 2, axis=1
            ))
        )

    print(f"2. RMSE bbox:  {errors['bbox']:.2f}")
    print(f"   RMSE pixel: {errors['pixel']:.2f}")

    assert errors["pixel"] < 1
    assert errors["bbox"] > 100

    print("Базові тести пройдено.")


def main():
    run_tests()

    images = create_test_images()

    print("\nПОРІВНЯННЯ МЕТОДІВ")
    print(f"{'Зображення':<16} {'Коробка':>10} {'Пікселі':>10}")

    for name, original in images.items():
        reconstructed_bbox = bc1(original, "bbox")
        reconstructed_pixel = bc1(original, "pixel")

        rmse_bbox = stats(original, reconstructed_bbox)[2]
        rmse_pixel = stats(original, reconstructed_pixel)[2]

        print(f"{name:<16} {rmse_bbox:>10.2f} {rmse_pixel:>10.2f}")

        # Зберігаємо результати двох методів
        Image.fromarray(original).save(
            OUTPUT / f"bc1_{name}_original.png"
        )

        Image.fromarray(reconstructed_bbox).save(
            OUTPUT / f"bc1_{name}_bbox.png"
        )

        Image.fromarray(reconstructed_pixel).save(
            OUTPUT / f"bc1_{name}_pixel.png"
        )

        save_difference(
            original, reconstructed_bbox,
            OUTPUT / f"bc1_{name}_bbox_diff.png"
        )

        save_difference(
            original, reconstructed_pixel,
            OUTPUT / f"bc1_{name}_pixel_diff.png"
        )

        # Додаткове кодування найкращим методом
        reconstructed_best = bc1(original, "best")

        Image.fromarray(reconstructed_best).save(
            OUTPUT / f"bc1_{name}_best.png"
        )

        # Кутова похибка нормал-мапи
        if name == "normal_map":
            angles = angle_error(original, reconstructed_bbox)

            print("\nКУТОВА ПОХИБКА НОРМАЛ-МАПИ")
            print(f"Середня: {angles.mean():.2f}°")
            print(f"Максимальна: {angles.max():.2f}°")
            print(
                f"Понад 2°: {(angles > 2).mean() * 100:.1f}%"
            )

    # Власний приклад: чотириколірне зображення
    custom = np.zeros((96, 96, 3), dtype=np.uint8)

    custom[:48, :48] = [255, 0, 0]
    custom[:48, 48:] = [0, 255, 0]
    custom[48:, :48] = [0, 0, 255]
    custom[48:, 48:] = [255, 255, 0]

    decoded = bc1(custom, "bbox")

    Image.fromarray(custom).save(
        OUTPUT / "bc1_custom_original.png"
    )

    Image.fromarray(decoded).save(
        OUTPUT / "bc1_custom.png"
    )

    save_difference(
        custom, decoded,
        OUTPUT / "bc1_custom_diff.png"
    )

    print("\nВласне зображення:")
    print(f"RMSE: {stats(custom, decoded)[2]:.2f}")

    print(f"\nУсі зображення збережено в папці: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
