import numpy as np
from PIL import Image

width, height = 1920, 1080
base_hex = "#0074B0"
noise_hex = "#42A8BE"
output_path = "perlin_wallpaper.png"

def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

base = hex_to_rgb(base_hex)
noise = hex_to_rgb(noise_hex)

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    return a + t * (b - a)

def generate_perlin(width, height, res=(6, 4), octaves=5, persistence=0.5, lacunarity=2):
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    frequency_x, frequency_y = res
    max_amp = 0.0

    for _ in range(octaves):
        gx = int(frequency_x) + 1
        gy = int(frequency_y) + 1

        angles = 2 * np.pi * np.random.RandomState(42 + int(frequency_x * 10 + frequency_y)).rand(gy, gx)
        gradients = np.dstack((np.cos(angles), np.sin(angles))).astype(np.float32)

        xs = np.linspace(0, frequency_x, width, endpoint=False, dtype=np.float32)
        ys = np.linspace(0, frequency_y, height, endpoint=False, dtype=np.float32)
        xi = xs.astype(int)
        yi = ys.astype(int)
        xf = xs - xi
        yf = ys - yi

        u = fade(xf)[None, :]
        v = fade(yf)[:, None]

        g00 = gradients[yi[:, None], xi[None, :]]
        g10 = gradients[yi[:, None], (xi + 1)[None, :]]
        g01 = gradients[(yi + 1)[:, None], xi[None, :]]
        g11 = gradients[(yi + 1)[:, None], (xi + 1)[None, :]]

        dx = xf[None, :]
        dy = yf[:, None]

        n00 = g00[..., 0] * dx + g00[..., 1] * dy
        n10 = g10[..., 0] * (dx - 1) + g10[..., 1] * dy
        n01 = g01[..., 0] * dx + g01[..., 1] * (dy - 1)
        n11 = g11[..., 0] * (dx - 1) + g11[..., 1] * (dy - 1)

        nx0 = lerp(n00, n10, u)
        nx1 = lerp(n01, n11, u)
        value = lerp(nx0, nx1, v)

        total += value * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency_x *= lacunarity
        frequency_y *= lacunarity

    return total / max_amp

noise_map = generate_perlin(width, height, res=(5, 3), octaves=6, persistence=0.55, lacunarity=2.0)

# Normalize and keep it subtle
noise_norm = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
blend = np.clip((noise_norm ** 1.5) * 0.85, 0, 1)[..., None]

img = (base * (1 - blend) + noise * blend).astype(np.uint8)
Image.fromarray(img, mode="RGB").save(output_path)

print(f"Saved wallpaper to: {output_path}")
