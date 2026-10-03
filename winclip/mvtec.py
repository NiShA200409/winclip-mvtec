import os
import numpy as np
import torch
from PIL import Image

MVTEC_OBJECTS = {
    "bottle": "bottle", "cable": "cable", "capsule": "capsule", "carpet": "carpet",
    "grid": "grid", "hazelnut": "hazelnut", "leather": "leather", "metal_nut": "metal nut",
    "pill": "pill", "screw": "screw", "tile": "tile", "toothbrush": "toothbrush",
    "transistor": "transistor", "wood": "wood", "zipper": "zipper",
}
_EXT = (".png", ".jpg", ".jpeg", ".bmp")


class MVTecTest:
    """Test split. Good images get an all-zero mask (they MUST be part of pixel metrics)."""
    def __init__(self, root, cat, preprocess, size):
        self.preprocess, self.size, self.items = preprocess, size, []
        test = os.path.join(root, cat, "test")
        for d in sorted(os.listdir(test)):
            p = os.path.join(test, d)
            if not os.path.isdir(p):
                continue
            for f in sorted(os.listdir(p)):
                if not f.lower().endswith(_EXT):
                    continue
                mp = None if d == "good" else os.path.join(
                    root, cat, "ground_truth", d, os.path.splitext(f)[0] + "_mask.png")
                self.items.append((os.path.join(p, f), mp, int(d != "good")))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp, lab = self.items[i]
        img = self.preprocess(Image.open(ip).convert("RGB"))
        if mp is not None:
            m = np.array(Image.open(mp).convert("L").resize((self.size, self.size), Image.NEAREST)) > 0
        else:
            m = np.zeros((self.size, self.size), bool)
        return img, m, lab


def reference_images(root, cat, preprocess, k, seed):
    d = os.path.join(root, cat, "train", "good")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(_EXT))
    pick = np.random.RandomState(seed).choice(len(files), k, replace=False)
    return torch.stack([preprocess(Image.open(os.path.join(d, files[i])).convert("RGB")) for i in pick])
