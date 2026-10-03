import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.measure import label as cc_label
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def upsample_maps(maps, size, sigma=4):
    t = torch.from_numpy(np.asarray(maps, dtype=np.float32))[:, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()
    return np.stack([gaussian_filter(m, sigma=sigma) for m in t])


def _f1max(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float((2 * p * r / np.maximum(p + r, 1e-12)).max())


def image_metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    return {"AUROC": 100 * roc_auc_score(labels, scores),
            "AUPR": 100 * average_precision_score(labels, scores),
            "F1max": 100 * _f1max(labels, scores)}


def pro_score(masks, maps, max_fpr=0.3, n_thr=300):
    masks = masks.astype(bool)
    regions = []
    for n, m in enumerate(masks):
        if m.any():
            lab = cc_label(m, connectivity=2)
            for r in range(1, lab.max() + 1):
                regions.append((n, np.flatnonzero((lab == r).reshape(-1))))
    neg_total = (~masks).sum()
    thrs = np.linspace(maps.min(), maps.max(), n_thr)
    flat = maps.reshape(len(maps), -1)
    fprs, pros = [], []
    for t in thrs:
        pred = flat >= t
        fprs.append((pred & ~masks.reshape(len(maps), -1)).sum() / neg_total)
        pros.append(np.mean([pred[n][idx].mean() for n, idx in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= max_fpr
    order = np.argsort(fprs[keep])
    f, p = fprs[keep][order], pros[keep][order]
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(p, f) / max_fpr)


def pixel_metrics(maps, masks, with_pro=True):
    y, s = masks.reshape(-1).astype(np.uint8), maps.reshape(-1).astype(np.float32)
    out = {"pAUROC": 100 * roc_auc_score(y, s), "pF1max": 100 * _f1max(y, s)}
    if with_pro:
        out["PRO"] = 100 * pro_score(masks, maps)
    return out
