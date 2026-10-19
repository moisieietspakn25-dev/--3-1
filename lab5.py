import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import data
import matplotlib.pyplot as plt
import os

OUT_DIR = "lab5_results"
os.makedirs(OUT_DIR, exist_ok=True)


# Завдання 1. Перетворення RGB <-> YCbCr
def rgb2ycbcr(img):
    a = img.astype(np.float64)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B

    return Y, Cb, Cr


def ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).round().astype(np.uint8)


# Завдання 2. Субдискретизація 4:2:2
def down422(C):
    h, w = C.shape
    w2 = w // 2 * 2
    C = C[:, :w2]
    return C.reshape(h, w2 // 2, 2).mean(axis=2)


def up422(C, w):
    return np.repeat(C, 2, axis=1)[:, :w]


# Завдання 3. Субдискретизація 4:2:0
def down420(C):
    h, w = C.shape
    h2 = h // 2 * 2
    w2 = w // 2 * 2
    C = C[:h2, :w2]
    return C.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def up420(C, h, w):
    return np.repeat(
        np.repeat(C, 2, axis=0), 2, axis=1
    )[:h, :w]


def scheme(img, mode):
    img = np.asarray(img, dtype=np.uint8)
    h, w = img.shape[:2]

    # Доповнення краю для зображень із непарними розмірами.
    pad_h = h % 2
    pad_w = w % 2

    if pad_h or pad_w:
        img_work = np.pad(
            img,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge"
        )
    else:
        img_work = img

    Y, Cb, Cr = rgb2ycbcr(img_work)
    hp, wp = Y.shape

    if mode == "4:4:4":
        pass

    elif mode == "4:2:2":
        Cb = up422(down422(Cb), wp)
        Cr = up422(down422(Cr), wp)

    elif mode == "4:2:0":
        Cb = up420(down420(Cb), hp, wp)
        Cr = up420(down420(Cr), hp, wp)

    else:
        raise ValueError("Невідома схема: " + str(mode))

    return ycbcr2rgb(Y, Cb, Cr)[:h, :w]


# Пошук шрифту для напису
def find_font(size=48):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass

    return ImageFont.load_default()


# Тестове зображення: червоний текст на чорному тлі
def redtext(W=320, H=96):
    im = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = find_font(48)

    d.text((12, 22), "ЧЕРВОНИЙ", font=f, fill=(255, 0, 0))
    return np.array(im)


# Тестове зображення: тонкі кольорові лінії
def lines(W=320, H=160):
    im = np.zeros((H, W, 3), np.uint8)

    cols = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]

    for i, c in enumerate(cols):
        x = 24 + i * 48
        im[12:70, x:x + 2] = c
        im[12:70, x + 22:x + 23] = c

    for i, c in enumerate(cols[:4]):
        y = 90 + i * 16
        im[y:y + 2, 24:296] = c
        im[y + 7:y + 8, 24:296] = c

    return im


# Завдання 4. Карта різниці
def diffmap(orig, out):
    d = np.abs(
        out.astype(np.int16) - orig.astype(np.int16)
    )
    return d.max(axis=2)


def save_diffmap(orig, out, filename, title):
    dm = diffmap(orig, out)

    plt.figure(figsize=(8, 4))
    plt.imshow(
        np.clip(dm * 2, 0, 255),
        cmap="inferno",
        vmin=0,
        vmax=255
    )
    plt.title(title + " (похибка ×2)")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT_DIR, filename),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


# Завдання 5. Вимірювання похибок
def measure(orig, mode):
    out = scheme(orig, mode)

    d = np.abs(
        out.astype(np.int16) - orig.astype(np.int16)
    )
    mx = d.max(axis=2)

    return (
        int(d.max()),
        float(d.mean()),
        float((mx > 0).mean() * 100),
        float((mx > 30).mean() * 100),
    )


def save_comparison(orig, mode, filename):
    out = scheme(orig, mode)
    comparison = np.concatenate([orig, out], axis=0)

    Image.fromarray(comparison).save(
        os.path.join(OUT_DIR, filename)
    )
    return out


def main():
    print("Лабораторна робота 5 — Субдискретизація кольору")
    print("=" * 60)

    # Перевірка RGB -> YCbCr -> RGB
    photo = data.astronaut()

    Y, Cb, Cr = rgb2ycbcr(photo)
    back = ycbcr2rgb(Y, Cb, Cr)

    error = np.abs(
        back.astype(np.int16) - photo.astype(np.int16)
    ).max()

    print("RGB -> YCbCr -> RGB, максимальна похибка:", error)

    # Створення тестових зображень
    txt = redtext()
    lin = lines()

    tests = {
        "червоний текст": txt,
        "тонкі лінії": lin,
        "фотографія": photo,
    }

    Image.fromarray(txt).save(
        os.path.join(OUT_DIR, "text_original.png")
    )
    Image.fromarray(lin).save(
        os.path.join(OUT_DIR, "lines_original.png")
    )
    Image.fromarray(photo).save(
        os.path.join(OUT_DIR, "photo_original.png")
    )

    # Зображення тексту після субдискретизації
    save_comparison(txt, "4:2:2", "text422.png")
    save_comparison(txt, "4:2:0", "text420.png")

    # Карти різниці для тексту та фотографії
    text420 = scheme(txt, "4:2:0")
    save_diffmap(
        txt, text420, "diff_text.png",
        "Червоний текст, 4:2:0"
    )

    photo420 = scheme(photo, "4:2:0")
    save_diffmap(
        photo, photo420, "diff_photo.png",
        "Фотографія, 4:2:0"
    )

    # Порівняння тонких ліній у трьох схемах
    trio = [
        scheme(lin, "4:4:4"),
        scheme(lin, "4:2:2"),
        scheme(lin, "4:2:0"),
    ]

    Image.fromarray(np.concatenate(trio, axis=1)).save(
        os.path.join(OUT_DIR, "lines420.png")
    )

    # Таблиця вимірювань
    header = (
        f"{'Зображення':<20} {'Схема':<8} {'Макс':>5} "
        f"{'Середня':>8} {'Змінених':>10} {'Помітних':>10}"
    )

    print("\n" + header)
    print("-" * len(header))

    results = [header, "-" * len(header)]

    for name, im in tests.items():
        for mode in ("4:4:4", "4:2:2", "4:2:0"):
            mx, avg, changed, visible = measure(im, mode)

            row = (
                f"{name:<20} {mode:<8} {mx:>5d} "
                f"{avg:>8.2f} {changed:>9.1f}% "
                f"{visible:>9.2f}%"
            )

            print(row)
            results.append(row)

    # Висновок та відповіді на контрольні запитання
    report = [
        "",
        "ВИСНОВОК",
        "Субдискретизація кольору зберігає канал яскравості Y, "
        "а канали Cb і Cr усереднюються та відновлюються "
        "повторенням значень. Похибка найбільше помітна на "
        "різких кольорових межах і тонких лініях.",
        "",
        "ВІДПОВІДІ НА КОНТРОЛЬНІ ЗАПИТАННЯ",
        "",
        "1. На квадраті 2×2 схема 4:4:4 зберігає 12 чисел, "
        "4:2:2 — 8 чисел (економія 33,3%), 4:2:0 — 6 чисел "
        "(економія 50%).",
        "",
        "3. На карті різниці похибка зосереджується на контурах "
        "тексту та кольорових ліній, оскільки сусідні пікселі "
        "мають різні кольори. На однорідних ділянках сусідні "
        "значення схожі, тому похибка менша.",
        "",
        "4. У фотографії змінюється багато пікселів, але похибка "
        "зазвичай мала й непомітна. У тексті зміни зосереджені "
        "на чітких межах червоних букв і чорного тла, тому "
        "їх легше помітити.",
        "",
        "Примітка: на картах різниці похибка підсилена множником 2.",
    ]

    results.extend(report)

    with open(
        os.path.join(OUT_DIR, "results.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(results))

    print("\nРезультати збережено в папці:", OUT_DIR)
    print(
        "Створені файли: text422.png, text420.png, "
        "diff_text.png, diff_photo.png, lines420.png, results.txt"
    )


if __name__ == "__main__":
    main()