"""Does a latent upscale keep the draw? (AB-A-0257, the AB-R-0795 method.)

Each upscaled image is Lanczos'd back to the base draw's size and compared with the base decode: DISTS (piq, mean over
a grid of 512² crops) and PSNR. Low DISTS = the same picture. The AB-R-0795 band for one ×2 pass from a Klein 4B 1024²
draw is 0.015–0.034 (0.031–0.061 for ×4).
usage: python drift.py <base.png> <upscaled.png> [...]   (prints one JSON line per image)
"""
import json, sys
import numpy as np, torch, piq
from PIL import Image

D = piq.DISTS(reduction="none")


def arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255


def dists(a, b, s=512):
    H, W = a.shape[:2]
    v = []
    for y in range(0, H - s + 1, s):
        for x in range(0, W - s + 1, s):
            ta = torch.from_numpy(a[y:y + s, x:x + s]).permute(2, 0, 1)[None]
            tb = torch.from_numpy(b[y:y + s, x:x + s]).permute(2, 0, 1)[None]
            v.append(D(ta, tb).item())
    return float(np.mean(v))


base = Image.open(sys.argv[1])
b = arr(base)
for p in sys.argv[2:]:
    im = Image.open(p)
    d = arr(im.resize(base.size, Image.LANCZOS))
    print(json.dumps({"image": p, "size": list(im.size), "dists": round(dists(d, b), 4),
                      "psnr": round(float(10 * np.log10(1 / np.mean((d - b) ** 2))), 2)}))
