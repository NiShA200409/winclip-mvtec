"""WinCLIP / WinCLIP+ (CVPR 2023) re-implementation on OpenCLIP ViT-B/16+ (LAION-400M).

Key points that follow the paper:
  * Compositional Prompt Ensemble (state words x templates), 2 classes: normal / anomalous
  * Window features = only the tokens inside each window (+CLS) go through the ViT,
    keeping their ORIGINAL positional embeddings (no crop-and-resize)   [Eq. 2]
  * Every window is scored first, then scores are harmonically averaged per pixel [Eq. 3]
  * Multi-scale: small (2x2 patches), mid (3x3 patches), image-scale (CLS of the full image)
  * WinCLIP+: reference memories for patch / small / mid features, M = min 0.5*(1-cos) [Eq. 4,5]
  * AC score for WinCLIP+ = 1/2 (ascore0 + max M)                          [Eq. 6]
"""
import numpy as np
import torch
import torch.nn.functional as F
import open_clip

# NOTE: word lists adapted from public re-implementations of the paper's supplementary
# (Fig. 6). Check them against the supplementary PDF if you want an exact match.
NORMAL_STATES = ["{}", "flawless {}", "perfect {}", "unblemished {}",
                 "{} without flaw", "{} without defect", "{} without damage"]
ABNORMAL_STATES = ["damaged {}", "broken {}", "{} with flaw",
                   "{} with defect", "{} with damage"]
TEMPLATES = [
    "a cropped photo of the {}.", "a cropped photo of a {}.",
    "a close-up photo of the {}.", "a close-up photo of a {}.",
    "a bright photo of the {}.", "a bright photo of a {}.",
    "a dark photo of the {}.", "a dark photo of a {}.",
    "a jpeg corrupted photo of the {}.", "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.", "a blurry photo of a {}.",
    "a photo of the {}.", "a photo of a {}.",
    "a photo of a small {}.", "a photo of the small {}.",
    "a photo of a large {}.", "a photo of the large {}.",
    "a photo of the {} for visual inspection.", "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.", "a photo of a {} for anomaly detection.",
]


def spread(vals, pos, k, g, how):
    """Distribute per-window values onto the g x g patch grid, aggregating overlaps."""
    vals = vals.detach().float().cpu().numpy()
    acc = np.zeros((g, g)); cnt = np.zeros((g, g))
    for v, (i, j) in zip(vals, pos):
        acc[i:i + k, j:j + k] += (1.0 / (v + 1e-8)) if how == "harmonic" else v
        cnt[i:i + k, j:j + k] += 1
    return cnt / acc if how == "harmonic" else acc / cnt


class WinCLIPModel:
    def __init__(self, device="cuda", model_name="ViT-B-16-plus-240", pretrained="laion400m_e32"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        v = self.model.visual
        self.grid = v.grid_size[0]            # 15 for 240px / patch 16
        self.img_size = v.image_size[0]       # 240
        self.scale = float(self.model.logit_scale.exp())

    # ---------------- text side (CPE) ----------------
    @torch.no_grad()
    def text_features(self, obj):
        def enc(states):
            per_state = []
            for s in states:
                phrase = s.format(obj)
                toks = self.tokenizer([t.format(phrase) for t in TEMPLATES]).to(self.device)
                f = F.normalize(self.model.encode_text(toks), dim=-1).mean(0)
                per_state.append(f)
            return F.normalize(torch.stack(per_state).mean(0), dim=-1)
        return torch.stack([enc(NORMAL_STATES), enc(ABNORMAL_STATES)])   # [2, D]

    # ---------------- visual side ----------------
    def _tokens(self, img):
        """Tokens after patch-embed + CLS + positional emb + ln_pre: [B, 1+N, W]."""
        v = self.model.visual
        x = v.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
        return v.ln_pre(x)

    def _encode(self, x):
        """Run transformer on a token sequence. Returns normalized (cls [B,D], patches [B,L-1,D])."""
        v = self.model.visual
        t = v.transformer
        if getattr(t, "batch_first", False):
            x = t(x)
        else:
            x = t(x.permute(1, 0, 2)).permute(1, 0, 2)
        cls, patches = v.ln_post(x[:, 0]), v.ln_post(x[:, 1:])
        if v.proj is not None:
            cls, patches = cls @ v.proj, patches @ v.proj
        return F.normalize(cls, dim=-1), F.normalize(patches, dim=-1)

    def window_feats(self, x, k):
        """Eq.2: for every k x k window feed [CLS + window tokens] (original positions)."""
        g = self.grid
        cls_tok, patch = x[:, :1], x[:, 1:].reshape(1, g, g, -1)
        wins, pos = [], []
        for i in range(g - k + 1):
            for j in range(g - k + 1):
                wins.append(torch.cat([cls_tok, patch[:, i:i + k, j:j + k].reshape(1, k * k, -1)], 1))
                pos.append((i, j))
        wins = torch.cat(wins, 0)
        feats = [self._encode(wins[b:b + 256])[0] for b in range(0, len(wins), 256)]
        return torch.cat(feats), pos

    # ---------------- WinCLIP+ memory ----------------
    @torch.no_grad()
    def build_memory(self, imgs):
        P, S, M = [], [], []
        for im in imgs:
            x = self._tokens(im[None].to(self.device))
            _, pf = self._encode(x)
            P.append(pf[0]); S.append(self.window_feats(x, 2)[0]); M.append(self.window_feats(x, 3)[0])
        return {"P": torch.cat(P), "S": torch.cat(S), "M": torch.cat(M)}

    @staticmethod
    def _assoc(q, mem):                      # Eq. 4
        return 0.5 * (1.0 - (q @ mem.T).max(dim=1).values)

    # ---------------- inference ----------------
    @torch.no_grad()
    def infer(self, img, T, mem=None):
        g = self.grid
        x = self._tokens(img.to(self.device))
        cls_f, patch_f = self._encode(x)
        prob = lambda f: torch.softmax(self.scale * f @ T.T, dim=-1)[:, 1]   # P(anomalous)
        img_score = prob(cls_f)[0].item()

        wf, maps = {}, []
        for k in (2, 3):
            f, pos = self.window_feats(x, k)
            wf[k] = (f, pos)
            maps.append(spread(prob(f), pos, k, g, "harmonic"))              # Eq. 3
        maps.append(np.full((g, g), img_score))                              # image-scale
        lang_map = len(maps) / sum(1.0 / (m + 1e-8) for m in maps)           # multi-scale harmonic

        out = {"lang_map": lang_map.astype(np.float32), "lang_img": img_score}
        if mem is not None:
            mp = self._assoc(patch_f[0], mem["P"]).view(g, g).cpu().numpy()
            ms = spread(self._assoc(wf[2][0], mem["S"]), wf[2][1], 2, g, "mean")
            mm = spread(self._assoc(wf[3][0], mem["M"]), wf[3][1], 3, g, "mean")
            vis = (mp + ms + mm) / 3.0                                       # Eq. 5
            out["vis_map"] = vis.astype(np.float32)
            out["vis_max"] = float(vis.max())
        return out
