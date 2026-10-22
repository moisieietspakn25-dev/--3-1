import numpy as np
from PIL import Image
from skimage import data


def to_light(c):
    """Перетворення sRGB у лінійне світло."""
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(
        c <= 0.04045,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4
    )


def to_code(L):
    """Перетворення лінійного світла у sRGB."""
    L = np.asarray(L, dtype=np.float64)
    c = np.where(
        L <= 0.0031308,
        L * 12.92,
        1.055 * np.maximum(L, 0) ** (1 / 2.4) - 0.055
    )
    return np.clip(c * 255, 0, 255)


def half_naive(img):
    """Зменшення зображення вдвічі звичайним усередненням."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    img = img[:h // 2 * 2, :w // 2 * 2]
    return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def half_linear(img):
    """Зменшення зображення з урахуванням гамми."""
    L = to_light(img)
    h, w = L.shape[:2]
    L = L[:h // 2 * 2, :w // 2 * 2]
    reduced = L.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return to_code(reduced)


def save_image(array, filename):
    """Збереження масиву як PNG."""
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    Image.fromarray(array).save(filename)


def checker(N=512, cell=4, ang=18):
    """Створення шахівниці під кутом."""
    y, x = np.indices((N, N)).astype(float) - N / 2
    a = np.deg2rad(ang)

    xr = x * np.cos(a) - y * np.sin(a)
    yr = x * np.sin(a) + y * np.cos(a)

    return (
        (np.floor(xr / cell) + np.floor(yr / cell)) % 2
    ) * 255


def main():
    # Завдання 1. Збільшення зображення.
    gray = np.array(
        Image.fromarray(data.astronaut()).convert("L")
    )

    small = gray[340:356, 372:388]
    im = Image.fromarray(small)

    im.resize(
        (320, 320), Image.Resampling.NEAREST
    ).save("up_nearest.png")

    im.resize(
        (320, 320), Image.Resampling.BILINEAR
    ).save("up_bilinear.png")

    im.resize(
        (320, 320), Image.Resampling.LANCZOS
    ).save("up_lanczos.png")

    print("Завдання 1 виконано.")

    # Завдання 2. Зменшення та аліасинг.
    ch = checker()
    k = 4

    nearest = ch[::k, ::k]
    area = ch.reshape(
        512 // k, k, 512 // k, k
    ).mean(axis=(1, 3))

    save_image(nearest, "ds_nearest.png")
    save_image(area, "ds_area.png")

    print("\nЗавдання 2:")
    print(
        "Прорідження: середнє %.1f, розкид %.1f"
        % (nearest.mean(), nearest.std())
    )
    print(
        "Усереднення: середнє %.1f, розкид %.1f"
        % (area.mean(), area.std())
    )

    # Завдання 3. Побудова ланцюжка міпмапів.
    g = gray.astype(float)
    level = 0

    while g.shape[0] >= 2 and g.shape[1] >= 2:
        save_image(g, f"mip_{level}.png")

        print(
            f"Рівень {level}: {g.shape[1]}x{g.shape[0]}, "
            f"середня яскравість {g.mean():.1f}"
        )

        g = half_naive(g)
        level += 1

    # Завдання 4. Порівняння впливу гамми.
    cb = (
        np.indices((256, 256)).sum(axis=0) % 2
    ) * 255.0

    print("\nЗавдання 4 — шахівниця:")
    print("Наївно: %.1f" % half_naive(cb).mean())
    print("Через лінійне світло: %.1f" % half_linear(cb).mean())

    # Порівняння на фотографії.
    gn = gray.astype(float)
    gl = gray.astype(float)

    print("\nТаблиця яскравості міпмапів:")
    print("Рівень | Розмір | Наївно | Правильно | Різниця")
    print(
        "0 | 512 | %.1f | %.1f | %.1f"
        % (gn.mean(), gl.mean(), gl.mean() - gn.mean())
    )

    for lv in range(1, 6):
        gn = half_naive(gn)
        gl = half_linear(gl)

        print(
            "%d | %d | %.1f | %.1f | %+.1f"
            % (
                lv,
                gn.shape[0],
                gn.mean(),
                gl.mean(),
                gl.mean() - gn.mean()
            )
        )

    print("\nУсі завдання виконано!")


if __name__ == "__main__":
    main()