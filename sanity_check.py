"""Run this FIRST. It catches the silent bugs that break reproduction.
  python sanity_check.py --data_root /path/to/mvtec_anomaly_detection
"""
import argparse, torch
import torch.nn.functional as F
from winclip import WinCLIPModel, MVTecTest, MVTEC_OBJECTS

ap = argparse.ArgumentParser(); ap.add_argument("--data_root", required=True); a = ap.parse_args()
wm = WinCLIPModel("cuda" if torch.cuda.is_available() else "cpu")
ds = MVTecTest(a.data_root, "bottle", wm.preprocess, wm.img_size)
img = ds[0][0][None].to(wm.device)

with torch.no_grad():
    ref = F.normalize(wm.model.encode_image(img), dim=-1)
    x = wm._tokens(img)
    mine, _ = wm._encode(x)
    d1 = (ref - mine).abs().max().item()
    f, pos = wm.window_feats(x, wm.grid)             # one window covering the whole image
    d2 = (ref - f).abs().max().item()
print(f"[1] manual ViT forward vs open_clip encode_image : max|diff| = {d1:.2e}  (must be < 1e-3)")
print(f"[2] full-size window  vs open_clip encode_image  : max|diff| = {d2:.2e}  (must be < 1e-3)")
assert d1 < 1e-3 and d2 < 1e-3, "ViT forward is wrong -> fix before running anything else"

T = wm.text_features(MVTEC_OBJECTS["metal_nut"])
print("[3] text feats", tuple(T.shape), "| cos(normal, anomalous) =", round(float(T[0] @ T[1]), 4))
print("[4] example prompt:", "a photo of a {}.".format("flawless metal nut"))
print("OK - model side is sane.")
