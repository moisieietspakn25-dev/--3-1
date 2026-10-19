
import numpy as np
import struct
import sys
import os


# Завдання 1. Створення та читання PPM

def create_test_image():
    img = np.zeros((4, 6, 3), dtype=np.uint8)

    img[0:2, 0:3] = (255, 0, 0)
    img[0:2, 3:6] = (0, 255, 0)
    img[2:4, 0:3] = (0, 0, 255)
    img[2:4, 3:6] = (255, 255, 255)

    return img


def write_ppm(path, img):
    height, width, _ = img.shape

    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def read_ppm(path):
    with open(path, "rb") as f:
        if f.readline().strip() != b"P6":
            raise ValueError("Файл не є P6 PPM")

        # Пропускаємо коментарі в заголовку
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        width, height = map(int, line.split())
        max_value = int(f.readline().strip())

        if max_value != 255:
            raise ValueError("Підтримуються лише PPM з max value 255")

        data = f.read()
        expected = width * height * 3

        if len(data) != expected:
            raise ValueError("Неправильний розмір даних PPM")

        return np.frombuffer(data, dtype=np.uint8).reshape(
            height, width, 3
        ).copy()


# Завдання 2. Запис 24-бітного BMP

def write_bmp24(path, img):
    height, width, _ = img.shape

    stride = ((width * 24 + 31) // 32) * 4
    padding = stride - width * 3
    offbits = 54
    filesize = offbits + stride * height

    out = bytearray()

    out += struct.pack(
        "<2sIHHI", b"BM", filesize, 0, 0, offbits
    )

    out += struct.pack(
        "<IiiHHIIiiII",
        40, width, height, 1, 24,
        0, stride * height, 2835, 2835, 0, 0
    )

    for y in range(height - 1, -1, -1):
        for x in range(width):
            r, g, b = map(int, img[y, x])
            out += bytes((b, g, r))

        out += b"\x00" * padding

    with open(path, "wb") as f:
        f.write(out)


# Завдання 3. Читання 24-бітного BMP

def read_bmp24(path):
    with open(path, "rb") as f:
        data = f.read()

    if data[:2] != b"BM":
        raise ValueError("Файл не є BMP")

    offbits = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    signed_height = struct.unpack_from("<i", data, 22)[0]
    bits = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]

    if bits != 24 or compression != 0:
        raise ValueError("Потрібен нестиснений 24-бітний BMP")

    if width <= 0 or signed_height == 0:
        raise ValueError("Некоректні розміри BMP")

    height = abs(signed_height)
    stride = ((width * bits + 31) // 32) * 4

    if len(data) < offbits + stride * height:
        raise ValueError("Файл BMP пошкоджений або неповний")

    img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        start = offbits + y * stride

        for x in range(width):
            pos = start + x * 3
            b, g, r = data[pos:pos + 3]

            if signed_height > 0:
                img[height - 1 - y, x] = (r, g, b)
            else:
                img[y, x] = (r, g, b)

    return img


# Завдання 4. Створення палітрового 8-бітного BMP

def color_index(r, g, b):
    ri = min(5, int(r) // 51)
    gi = min(5, int(g) // 51)
    bi = min(5, int(b) // 51)

    return 36 * ri + 6 * gi + bi


def write_bmp8(path, img):
    height, width, _ = img.shape

    palette = bytearray()

    for i in range(256):
        if i < 216:
            r = (i // 36) * 51
            g = ((i // 6) % 6) * 51
            b = (i % 6) * 51
        else:
            r = g = b = 0

        palette += bytes((b, g, r, 0))

    stride = ((width * 8 + 31) // 32) * 4
    padding = stride - width
    offbits = 14 + 40 + 1024
    filesize = offbits + stride * height

    out = bytearray()

    out += struct.pack(
        "<2sIHHI", b"BM", filesize, 0, 0, offbits
    )

    out += struct.pack(
        "<IiiHHIIiiII",
        40, width, height, 1, 8,
        0, stride * height, 2835, 2835, 256, 0
    )

    out += palette

    for y in range(height - 1, -1, -1):
        for x in range(width):
            r, g, b = map(int, img[y, x])
            out.append(color_index(r, g, b))

        out += b"\x00" * padding

    with open(path, "wb") as f:
        f.write(out)


# Завдання 5. Конвертер PPM <-> BMP

def convert(src, dst):
    src_ext = os.path.splitext(src)[1].lower()
    dst_ext = os.path.splitext(dst)[1].lower()

    if src_ext == ".ppm" and dst_ext == ".bmp":
        img = read_ppm(src)
        write_bmp24(dst, img)

    elif src_ext == ".bmp" and dst_ext == ".ppm":
        img = read_bmp24(src)
        write_ppm(dst, img)

    else:
        raise ValueError(
            "Підтримуються конвертації PPM -> BMP "
            "і 24-бітний BMP -> PPM"
        )

    print(f"Конвертацію завершено: {src} -> {dst}")


def main():
    if len(sys.argv) == 3:
        try:
            convert(sys.argv[1], sys.argv[2])
        except (OSError, ValueError, struct.error) as e:
            print(f"Помилка: {e}")
            sys.exit(1)
        return

    # Автоматична перевірка завдань 1–4
    print("Лабораторна робота: PPM і BMP")
    print("-" * 40)

    img = create_test_image()
    write_ppm("test.ppm", img)

    a = read_ppm("test.ppm")

    print("Розмір масиву:", a.shape)
    print("Піксель [0, 0]:", a[0, 0])
    print("Піксель [3, 5]:", a[3, 5])

    write_bmp24("test.bmp", a)
    b = read_bmp24("test.bmp")

    print("PPM і BMP збігаються:", np.array_equal(a, b))
    print("Розмір test.ppm:", os.path.getsize("test.ppm"), "байт")
    print("Розмір test.bmp:", os.path.getsize("test.bmp"), "байт")

    write_bmp8("test8.bmp", a)

    print("Розмір test8.bmp:", os.path.getsize("test8.bmp"), "байт")
    print("Палітровий BMP створено.")

    print("-" * 40)
    print("Перевірку завершено.")


if __name__ == "__main__":
    main()
