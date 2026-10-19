import numpy as np
from PIL import Image
import os
import io

OUTPUT_DIR = "lab3_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 1. ПІДГОТОВКА ЧОТИРЬОХ ЗОБРАЖЕНЬ
# ==================================================

# Градієнт
grad = np.tile(
    np.linspace(0, 255, 512).astype(np.uint8),
    (384, 1)
)

# Білий шум
rng = np.random.default_rng(1)
noise = rng.integers(0, 256, (384, 512)).astype(np.uint8)


def load_grayscale(filename):
    """Завантажує зображення та перетворює його на grayscale."""
    if not os.path.exists(filename):
        return None

    img = Image.open(filename).convert("L").resize((512, 384))
    return np.array(img, dtype=np.uint8)


scr = load_grayscale("screen.png")
photo = load_grayscale("photo.jpg")

images = {
    "Градієнт": grad,
    "Білий шум": noise
}

if scr is not None:
    images["Скріншот"] = scr
else:
    print("Увага: файл screen.png не знайдено.")

if photo is not None:
    images["Фотографія"] = photo
else:
    print("Увага: файл photo.jpg не знайдено.")


# ==================================================
# 2. ЗБЕРЕЖЕННЯ ЗОБРАЖЕНЬ
# ==================================================

def save_image(filename, img):
    path = os.path.join(OUTPUT_DIR, filename)
    Image.fromarray(img.astype(np.uint8)).save(path)


save_image("gradient.png", grad)
save_image("white_noise.png", noise)

if scr is not None:
    save_image("screenshot.png", scr)

if photo is not None:
    save_image("photo.png", photo)


# ==================================================
# 3. ЕНТРОПІЯ
# ==================================================

def entropy(img):
    values, counts = np.unique(img, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ==================================================
# 4. RLE — ВЛАСНА РЕАЛІЗАЦІЯ
# ==================================================

def rle_size(img):
    data = img.ravel()
    n = len(data)
    i = 0
    out = 0

    while i < n:
        j = i

        while (
            j + 1 < n
            and data[j + 1] == data[i]
            and j - i + 1 < 255
        ):
            j += 1

        out += 2
        i = j + 1

    return out


# ==================================================
# 5. КОДУВАННЯ ХАФФМАНА
# ==================================================

def huffman_codes(freq):
    """Повертає коди Хаффмана для словника частот."""
    if len(freq) == 1:
        return {list(freq)[0]: "0"}

    nodes = [[n, {v: ""}] for v, n in freq.items()]

    while len(nodes) > 1:
        nodes.sort(key=lambda node: node[0])

        a = nodes.pop(0)
        b = nodes.pop(0)

        for v in a[1]:
            a[1][v] = "0" + a[1][v]

        for v in b[1]:
            b[1][v] = "1" + b[1][v]

        nodes.append([a[0] + b[0], {**a[1], **b[1]}])

    return nodes[0][1]


def huffman_stats(img):
    values, counts = np.unique(img, return_counts=True)

    freq = {
        int(v): int(c)
        for v, c in zip(values, counts)
    }

    codes = huffman_codes(freq)
    total = sum(freq.values())

    avg_length = sum(
        freq[v] * len(codes[v])
        for v in freq
    ) / total

    lengths = [len(c) for c in codes.values()]

    return avg_length, min(lengths), max(lengths)


# ==================================================
# 6. PNG — ВИМІРЮВАННЯ РОЗМІРУ
# ==================================================

def png_size(img):
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(
        buf,
        format="PNG",
        optimize=True
    )
    return len(buf.getvalue())


# ==================================================
# 7. АНАЛІЗ УСІХ ЗОБРАЖЕНЬ
# ==================================================

def analyze_images(images):
    results = []

    for name, img in images.items():
        original_size = img.nbytes

        ent = entropy(img)
        rle_bytes = rle_size(img)
        rle_percent = rle_bytes / original_size * 100

        huff_avg, huff_min, huff_max = huffman_stats(img)

        png_bytes = png_size(img)
        png_percent = png_bytes / original_size * 100

        results.append({
            "name": name,
            "entropy": ent,
            "huffman": huff_avg,
            "huffman_min": huff_min,
            "huffman_max": huff_max,
            "rle_bytes": rle_bytes,
            "rle_percent": rle_percent,
            "png_bytes": png_bytes,
            "png_percent": png_percent
        })

        print(f"\nЗображення: {name}")
        print(f"Ентропія: {ent:.2f} біт/піксель")
        print(
            f"Хаффман: {huff_avg:.2f} біт/піксель "
            f"(коди від {huff_min} до {huff_max} біт)"
        )
        print(f"RLE: {rle_bytes} байтів ({rle_percent:.2f}%)")
        print(f"PNG: {png_bytes} байтів ({png_percent:.2f}%)")

    return results


# ==================================================
# 8. ТАБЛИЦЯ РЕЗУЛЬТАТІВ
# ==================================================

def save_results(results):
    filename = os.path.join(OUTPUT_DIR, "results.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("ЗВІТ: ЕНТРОПІЙНЕ СТИСНЕННЯ\n\n")

        f.write(
            f"{'Зображення':<15}"
            f"{'Ентропія':>12}"
            f"{'Хаффман':>12}"
            f"{'RLE, %':>12}"
            f"{'PNG, %':>12}\n"
        )

        f.write("-" * 63 + "\n")

        for r in results:
            f.write(
                f"{r['name']:<15}"
                f"{r['entropy']:>12.2f}"
                f"{r['huffman']:>12.2f}"
                f"{r['rle_percent']:>12.2f}"
                f"{r['png_percent']:>12.2f}\n"
            )

        f.write("\nДЕТАЛЬНІ РЕЗУЛЬТАТИ ХАФФМАНА\n")

        for r in results:
            f.write(
                f"\n{r['name']}: середня довжина = "
                f"{r['huffman']:.2f} біт; "
                f"мінімум = {r['huffman_min']} біт; "
                f"максимум = {r['huffman_max']} біт.\n"
            )

        f.write("\nВІДПОВІДІ НА ЗАПИТАННЯ\n")

        f.write(
            "\n1. Для градієнта Хаффман не дає виграшу, "
            "оскільки всі 256 значень пікселів мають однакову частоту. "
            "Тому кожне значення отримує код довжиною 8 біт.\n"
        )

        f.write(
            "\n2. Для фотографії окремі рідкісні значення можуть отримати "
            "коди довші за 8 біт. Це не суперечить стисненню: часті значення "
            "отримують коротші коди, тому середня довжина може бути меншою "
            "за 8 біт.\n"
        )

        f.write("\nВИСНОВОК\n")
        f.write(
            "Ентропія за гістограмою враховує частоти окремих значень "
            "пікселів, але не враховує їхній порядок і залежності між "
            "сусідніми пікселями. Градієнт може мати ентропію 8 біт "
            "на піксель, хоча його сусідні пікселі змінюються передбачувано. "
            "PNG використовує фільтрацію та стиснення послідовностей даних, "
            "тому може ефективно використати цю просторову закономірність. "
            "Отже, ентропія окремих пікселів не є остаточною межею для "
            "методів, які враховують структуру зображення.\n"
        )


# ==================================================
# 9. ГОЛОВНА ФУНКЦІЯ
# ==================================================

def main():
    print("=" * 55)
    print("ЛАБОРАТОРНА РОБОТА №3")
    print("ЕНТРОПІЙНЕ СТИСНЕННЯ")
    print("=" * 55)

    print(f"\nКількість зображень: {len(images)}")
    print(f"Розмір одного зображення: {512 * 384} байтів")

    results = analyze_images(images)
    save_results(results)

    print("\n" + "=" * 55)
    print("Роботу завершено.")
    print(f"Результати збережено в папці: {OUTPUT_DIR}")
    print("Таблиця: results.txt")
    print("=" * 55)


if __name__ == "__main__":
    main()