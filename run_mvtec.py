"""Run WinCLIP (0-shot) and WinCLIP+ (k-shot) on MVTec-AD and write results.

  python run_mvtec.py --data_root /path/to/mvtec_anomaly_detection --shots 0 1 2 4 --seeds 5
"""
import argparse, json, os
import numpy as np, torch
from tqdm import tqdm
from winclip import (WinCLIPModel, MVTecTest, reference_images, MVTEC_OBJECTS,
                     upsample_maps, image_metrics, pixel_metrics)


def mm(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def pick(raw, mode):
    """mode: lang | vis | fused.  Fusion = mean of per-category min-max normalised scores
    (the two raw scales differ ~10x; the paper does not specify the calibration)."""
    if "vis_map" not in raw or mode == "lang":
        return raw["lang_map"], raw["lang_img"]
    if mode == "vis":
        return raw["vis_map"], raw["vis_max"]
    return (0.5 * (mm(raw["lang_map"]) + mm(raw["vis_map"])),
            0.5 * (mm(raw["lang_img"]) + mm(raw["vis_max"])))


def run_category(wm, root, cat, shots, seed, cache_dir):
    f = os.path.join(cache_dir, f"{cat}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return {k: z[k] for k in z.files}
    ds = MVTecTest(root, cat, wm.preprocess, wm.img_size)
    T = wm.text_features(MVTEC_OBJECTS[cat])
    mem = wm.build_memory(reference_images(root, cat, wm.preprocess, shots, seed)) if shots else None
    outs, masks, labels = [], [], []
    for i in tqdm(range(len(ds)), desc=f"{cat} {shots}-shot s{seed}", leave=False):
        img, m, l = ds[i]
        outs.append(wm.infer(img[None], T, mem)); masks.append(m); labels.append(l)
    raw = {"lang_map": np.stack([o["lang_map"] for o in outs]),
           "lang_img": np.array([o["lang_img"] for o in outs]),
           "masks": np.stack(masks), "labels": np.array(labels)}
    if mem is not None:
        raw["vis_map"] = np.stack([o["vis_map"] for o in outs])
        raw["vis_max"] = np.array([o["vis_max"] for o in outs])
    np.savez_compressed(f, **raw)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["fused"], help="fused lang vis")
    ap.add_argument("--cats", nargs="+", default=list(MVTEC_OBJECTS))
    ap.add_argument("--no_pro", action="store_true")
    a = ap.parse_args()

    wm = WinCLIPModel("cuda" if torch.cuda.is_available() else "cpu")
    for shots in a.shots:
        seeds = [0] if shots == 0 else list(range(a.seeds))
        for mode in a.modes:
            if shots == 0 and mode != a.modes[0]:
                continue
            per_seed = []
            for s in seeds:
                cache = os.path.join(a.out, "raw", f"{shots}shot_seed{s}"); os.makedirs(cache, exist_ok=True)
                rows = {}
                for cat in a.cats:
                    raw = run_category(wm, a.data_root, cat, shots, s, cache)
                    seg, ac = pick(raw, mode)
                    seg = upsample_maps(seg, wm.img_size)
                    rows[cat] = {**image_metrics(ac, raw["labels"]),
                                 **pixel_metrics(seg, raw["masks"], not a.no_pro)}
                    print(f"[{shots}-shot|{mode}|seed{s}] {cat:11s} " +
                          " ".join(f"{k}={v:5.1f}" for k, v in rows[cat].items()))
                keys = next(iter(rows.values())).keys()
                rows["MEAN"] = {k: float(np.mean([rows[c][k] for c in a.cats])) for k in keys}
                per_seed.append(rows)
            keys = per_seed[0]["MEAN"].keys()
            summary = {k: (float(np.mean([r["MEAN"][k] for r in per_seed])),
                           float(np.std([r["MEAN"][k] for r in per_seed]))) for k in keys}
            print(f"\n=== {shots}-shot | {mode} | mean over {len(seeds)} seed(s) ===")
            for k, (m, sd) in summary.items():
                print(f"  {k:7s} {m:5.1f} +- {sd:3.1f}")
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, f"{shots}shot_{mode}.json"), "w") as f:
                json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)


if __name__ == "__main__":
    main()
