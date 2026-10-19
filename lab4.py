import numpy as np
from PIL import Image
import os
from collections import Counter

OUTPUT_DIR = "lab4_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 1. ПІДГОТОВКА ЗОБРАЖЕНЬ
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
    if not os.path.exists(filename):
        return None

    return np.array(
        Image.open(filename).convert("L").resize((512, 384)),
        dtype=np.uint8
    )


scr = load_grayscale("screen.png")
photo = load_grayscale("photo.jpg")

images = {
    "Градієнт": grad,
    "Білий шум": noise
}

if scr is not None:
    images["Скріншот"] = scr
else:
    print("Попередження: screen.png не знайдено.")

if photo is not None:
    images["Фотографія"] = photo
else:
    print("Попередження: photo.jpg не знайдено.")


# ==================================================
# 2. ЕНТРОПІЯ
# ==================================================

def entropy(img):
    values, counts = np.unique(img, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ==================================================
# 3. ФІЛЬТРИ PNG: NONE, SUB, UP
# ==================================================

def filter_row(cur, prev, f):
    """
    cur  — поточний рядок uint8
    prev — попередній рядок оригінального зображення
    f    — 0 None, 1 Sub, 2 Up
    """
    a = np.concatenate(([0], cur[:-1])).astype(np.uint8)
    b = prev

    if f == 0:
        pred = np.zeros_like(cur)
    elif f == 1:
        pred = a
    elif f == 2:
        pred = b
    else:
        raise ValueError("Невідомий номер фільтра.")

    return (
        (cur.astype(np.int16) - pred.astype(np.int16)) % 256
    ).astype(np.uint8)


def filter_image(img, f):
    out = np.zeros_like(img)
    prev = np.zeros(img.shape[1], dtype=np.uint8)

    for y in range(img.shape[0]):
        out[y] = filter_row(img[y], prev, f)
        prev = img[y]

    return out


# ==================================================
# 4. ЗВОРОТНЕ ПЕРЕТВОРЕННЯ
# ==================================================

def unfilter_row(enc, prev, f):
    n = len(enc)
    out = np.zeros(n, dtype=np.uint8)

    for x in range(n):
        a = out[x - 1] if x > 0 else 0
        b = prev[x]

        if f == 0:
            pred = 0
        elif f == 1:
            pred = a
        elif f == 2:
            pred = b
        else:
            raise ValueError("Невідомий номер фільтра.")

        out[x] = (int(enc[x]) + int(pred)) % 256

    return out


def unfilter_image(img, f):
    out = np.zeros_like(img)
    prev = np.zeros(img.shape[1], dtype=np.uint8)

    for y in range(img.shape[0]):
        out[y] = unfilter_row(img[y], prev, f)
        prev = out[y]

    return out


# ==================================================
# 5. ВИБІР ФІЛЬТРА ДЛЯ КОЖНОГО РЯДКА
# ==================================================

def sum_abs(row):
    r = row.astype(np.int16)
    signed = np.where(r > 127, r - 256, r)
    return int(np.abs(signed).sum())


def filter_adaptive(img):
    out = np.zeros_like(img)
    chosen = []
    prev = np.zeros(img.shape[1], dtype=np.uint8)

    for y in range(img.shape[0]):
        best_filter = 0
        best_row = None
        best_score = None

        for f in (0, 1, 2):
            candidate = filter_row(img[y], prev, f)
            score = sum_abs(candidate)

            if best_score is None or score < best_score:
                best_score = score
                best_filter = f
                best_row = candidate

        out[y] = best_row
        chosen.append(best_filter)
        prev = img[y]

    return out, chosen


def unfilter_adaptive(filtered, chosen):
    out = np.zeros_like(filtered)
    prev = np.zeros(filtered.shape[1], dtype=np.uint8)

    for y in range(filtered.shape[0]):
        out[y] = unfilter_row(filtered[y], prev, chosen[y])
        prev = out[y]

    return out


# ==================================================
# 6. ЗБЕРЕЖЕННЯ ЗОБРАЖЕНЬ
# ==================================================

def save_image(filename, img):
    path = os.path.join(OUTPUT_DIR, filename)
    Image.fromarray(img.astype(np.uint8)).save(path)


# ==================================================
# 7. ТЕСТ ОДНОГО РЯДКА
# ==================================================

def test_filter_row():
    row = np.array(
        [40, 42, 44, 46, 48, 50, 52, 54],
        dtype=np.uint8
    )
    zeros = np.zeros(8, dtype=np.uint8)

    print("\n=== Перевірка одного рядка ===")
    print("Sub: ", filter_row(row, zeros, 1))
    print("None:", filter_row(row, zeros, 0))
    print("Up:  ", filter_row(row, row, 2))


# ==================================================
# 8. АНАЛІЗ ЕНТРОПІЇ
# ==================================================

def analyze_fixed_filters(images):
    names = {0: "None", 1: "Sub", 2: "Up"}
    results = []

    print("\n=== Ентропія після фільтрації ===")

    for name, img in images.items():
        original_entropy = entropy(img)

        row_result = {
            "name": name,
            "before": original_entropy
        }

        print(f"\n{name}:")
        print(f"До фільтрації: {original_entropy:.2f} біт/піксель")

        for f in (0, 1, 2):
            filtered = filter_image(img, f)
            ent = entropy(filtered)
            row_result[names[f]] = ent

            print(f"{names[f]:>5}: {ent:.2f} біт/піксель")

            restored = unfilter_image(filtered, f)
            print(
                f"  Відновлення: "
                f"{np.array_equal(img, restored)}"
            )

            save_image(
                f"{name.lower().replace(' ', '_')}_{names[f].lower()}.png",
                filtered
            )

        results.append(row_result)

    return results


# ==================================================
# 9. АДАПТИВНА ФІЛЬТРАЦІЯ
# ==================================================

def analyze_adaptive(images):
    results = []

    print("\n=== Адаптивний вибір фільтра ===")

    for name, img in images.items():
        filtered, chosen = filter_adaptive(img)
        restored = unfilter_adaptive(filtered, chosen)

        before = entropy(img)
        after = entropy(filtered)
        gain = (before - after) / before * 100 if before > 0 else 0

        counts = Counter(chosen)
        filter_names = {0: "None", 1: "Sub", 2: "Up"}

        most_common = counts.most_common(1)[0][0]
        optimal = filter_names[most_common]

        results.append({
            "name": name,
            "optimal": optimal,
            "before": before,
            "after": after,
            "gain": gain,
            "counts": counts
        })

        print(f"\n{name}:")
        print(f"Ентропія до: {before:.2f}")
        print(f"Ентропія після: {after:.2f}")
        print(f"Виграш: {gain:.2f}%")
        print(f"Розподіл фільтрів: {dict(counts)}")
        print(f"Найчастіший фільтр: {optimal}")
        print(f"Відновлення без втрат: {np.array_equal(img, restored)}")

        save_image(
            f"{name.lower().replace(' ', '_')}_adaptive.png",
            filtered
        )

    return results


# ==================================================
# 10. ЗВІТ І ТАБЛИЦІ
# ==================================================

def save_report(fixed_results, adaptive_results):
    report_path = os.path.join(OUTPUT_DIR, "results.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ЛАБОРАТОРНА РОБОТА №4 — ПРЕДИКТОРИ PNG\n\n")

        f.write("ТАБЛИЦЯ 1. ЕНТРОПІЯ ПІСЛЯ ФІЛЬТРАЦІЇ\n")
        f.write(
            f"{'Зображення':<15}"
            f"{'До':>10}"
            f"{'None':>10}"
            f"{'Sub':>10}"
            f"{'Up':>10}\n"
        )
        f.write("-" * 55 + "\n")

        for r in fixed_results:
            f.write(
                f"{r['name']:<15}"
                f"{r['before']:>10.2f}"
                f"{r['None']:>10.2f}"
                f"{r['Sub']:>10.2f}"
                f"{r['Up']:>10.2f}\n"
            )

        f.write("\nТАБЛИЦЯ 2. АДАПТИВНИЙ ВИБІР\n")
        f.write(
            f"{'Зображення':<15}"
            f"{'Фільтр':>10}"
            f"{'До':>10}"
            f"{'Після':>10}"
            f"{'Виграш %':>12}\n"
        )
        f.write("-" * 57 + "\n")

        for r in adaptive_results:
            f.write(
                f"{r['name']:<15}"
                f"{r['optimal']:>10}"
                f"{r['before']:>10.2f}"
                f"{r['after']:>10.2f}"
                f"{r['gain']:>12.2f}\n"
            )

        f.write("\nВИСНОВОК\n")
        f.write(
            "Фільтри PNG зменшують ентропію, коли між сусідніми "
            "пікселями є закономірності. Для градієнта фільтр Up "
            "ефективний, тому що сусідні рядки однакові, а різниця "
            "між ними дорівнює нулю. Для білого шуму сусідні пікселі "
            "практично не пов'язані, тому фільтрація не дає суттєвого "
            "виграшу. Фотографії містять як закономірності, так і "
            "деталі, тому результат зазвичай перебуває між градієнтом "
            "і шумом. На відміну від ентропії окремих пікселів, "
            "предиктори враховують залежність між сусідніми пікселями.\n"
        )

        f.write("\nВІДПОВІДІ НА ЗАПИТАННЯ\n")
        f.write(
            "1. Лівий сусід формується з cur[:-1] із нулем на початку. "
            "Для першого пікселя лівий сусід дорівнює нулю.\n"
        )
        f.write(
            "2. Up дає майже нуль для градієнта, оскільки рядки однакові. "
            "Sub залишає різницю між сусідніми пікселями.\n"
        )
        f.write(
            "3. Перетворення оборотне, бо віднімання та додавання по "
            "модулю 256 взаємно компенсуються.\n"
        )
        f.write(
            "4. Значення понад 127 трактуються як від'ємні різниці "
            "для коректного вибору фільтра за сумою модулів.\n"
        )
        f.write(
            "5. Білий шум не має виражених закономірностей між сусідніми "
            "пікселями, тому предиктори не допомагають.\n"
        )
        f.write(
            "6. Різні рядки можуть мати різні закономірності, тому "
            "адаптивний вибір дає кращий результат.\n"
        )
        f.write(
            "7. Номер фільтра потребує одного байта на рядок: для "
            "384 рядків це 384 байти.\n"
        )
        f.write(
            "8. Ентропія 8 біт за гістограмою не враховує порядок "
            "пікселів. Після предиктора дані містять більше нульових "
            "та малих різниць, і ентропія зменшується.\n"
        )


# ==================================================
# 11. ГОЛОВНА ФУНКЦІЯ
# ==================================================

def main():
    print("=" * 60)
    print("ЛАБОРАТОРНА РОБОТА №4")
    print("ПРЕДИКТОРИ PNG")
    print("=" * 60)

    test_filter_row()

    fixed_results = analyze_fixed_filters(images)
    adaptive_results = analyze_adaptive(images)

    save_report(fixed_results, adaptive_results)

    print("\nРоботу завершено.")
    print(f"Результати збережено в папці: {OUTPUT_DIR}")
    print("Таблиці та висновки: results.txt")


if __name__ == "__main__":
    main()