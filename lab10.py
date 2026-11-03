BPP = {
    'RGBA32': 32,
    'ASTC 4x4': 8,
    'ASTC 6x6': 128 / 36,
    'ASTC 8x8': 2
}

def mem(side, fmt, mips=True):
    b = side * side * BPP[fmt] / 8
    if mips:
        b *= 4 / 3
    return b / 2**20

scene = [
    ('Ground_Albedo', 2048),
    ('Ground_Normal', 2048),
]

total = sum(mem(s, 'RGBA32') for _, s in scene)

print(f'Пам’ять текстур: {total:.2f} МБ')