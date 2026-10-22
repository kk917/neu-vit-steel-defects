# %% [markdown]
# # NEU steel defects: frozen vs fine-tuned CLIP, v5 experiment suite
# MAAI901 Applied Vision Transformer Report. Run on a GPU runtime (Colab: Runtime > Change runtime type > T4 GPU), top to bottom.
#
# **What is new compared with v4** (each item answers a weakness of the first run):
# 1. **5-fold cross-validation** instead of one test split: every image is a test image once (1,800 test predictions per setting, not 360).
# 2. **Error rate on a log scale, plus calibration** (log-loss, ECE, confident mistakes): these still separate methods when accuracy hits ~100%.
# 3. **Rare-defect experiment**: one defect type has very few labels while the others have plenty (realistic factory scarcity).
# 4. **Novel-defect (open-set) experiment**: can the model notice a defect type it was never trained on?
# 5. **Robustness at several severities**, and a model trained with blur/noise augmentation.
# 6. **Tuning checks**: tuned linear probe, learning-rate sensitivity (so the conclusion does not depend on one fixed setting).
# 7. **Zero-shot done properly**: prompts chosen on training images only, with a confusion matrix.
# 8. **Difficulty check**: how similar are test images to training images, and does accuracy drop for the unusual ones?
# 9. **Occlusion maps**: where in the image does each model look?
#
# Everything is **resumable**: results are saved after each step. If Colab disconnects, run all cells again and finished work is skipped.
# To keep results if the VM is recycled, mount Google Drive and set `OUT_DIR` (see the next cell).

# %%
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")   # avoids a Mac/conda OpenMP crash
# Optional settings. Uncomment what you need BEFORE running the cells below.
# os.environ["PROFILE"] = "local"     # for a laptop (Mac M-series): lightest, 2 folds, no k=240, no ImageNet ViT
# os.environ["PROFILE"] = "lite"      # ~1.3 h instead of ~3 h: 3 folds, 1 seed. Default is "full".
# os.environ["SMOKE"] = "1"           # 3-minute code check on tiny settings, then remove and restart the runtime
# from google.colab import drive; drive.mount("/content/drive"); os.environ["OUT_DIR"] = "/content/drive/MyDrive/vit_results_v5"
# os.environ["DATA_DIR"] = "/content/NEU-DET"   # only if you uploaded the dataset yourself

# %%
import sys
if os.environ.get("FAKE") != "1":
    import importlib.util
    if any(importlib.util.find_spec(m) is None for m in ("open_clip", "timm", "kagglehub", "seaborn")):
        os.system("pip -q install open_clip_torch timm scikit-learn seaborn kagglehub")

# %%
if os.environ.get("FAKE") != "1":
    import torch, open_clip          # import PyTorch before sklearn/numpy-heavy libraries (avoids a Mac/conda OpenMP crash)
import glob, re, json, time, copy, math, random, collections, pickle, itertools, shutil
import numpy as np, pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import matplotlib
if os.environ.get("FAKE") == "1": matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAKE = os.environ.get("FAKE") == "1"            # logic test with synthetic data and numpy stand-ins (not used for real results)
SMOKE = os.environ.get("SMOKE") == "1" or FAKE
PROFILE = os.environ.get("PROFILE", "full")
OUT = os.environ.get("OUT_DIR", "vit_results_v5"); os.makedirs(OUT, exist_ok=True)

CFG = dict(
    data_dir=os.environ.get("DATA_DIR", "NEU-DET"),
    n_folds=5, test_per_class=60, img_size=224,
    budgets=[1, 2, 3, 5, 10, 20, 50, 240], small_max=20, seeds_small=[0, 1], seeds_big=[0],
    ft_enc_lr=1e-5, ft_head_lr=1e-3, ft_wd=0.01, ft_batch=32, ft_min_steps=100, ft_epochs=10,
    lr_C=10.0, C_grid=[0.01, 0.1, 1, 10, 100, 1000], imagenet=True, partial_from=10,
    clip_model="ViT-B-16-quickgelu", clip_pretrained="openai",
    rare_classes=[1, 3, 5], rare_ks=[1, 5, 20], rare_others=50, rare_seeds=[0, 1],
    open_k=20, open_seeds=[0, 1],
    rob_ks=[20, 240], rob_seeds=[0, 1],
    sens_ks=[5, 20], sens_seeds=[0, 1], sens_lrs=[3e-6, 1e-5, 3e-5], sens_Cs=[0.1, 1, 10, 100],
)
if PROFILE == "lite":
    CFG.update(n_folds=3, budgets=[1, 2, 5, 10, 20, 50, 240], seeds_small=[0], rare_seeds=[0], open_seeds=[0], rob_seeds=[0], sens_seeds=[0])
if PROFILE == "local":      # for a laptop (Apple M-series / CPU): fewer folds, no k=240, no ImageNet reference
    CFG.update(n_folds=2, budgets=[1, 2, 5, 10, 20, 50], seeds_small=[0], seeds_big=[0], imagenet=False, rare_classes=[1, 3], rare_ks=[1, 20],
               rare_seeds=[0], open_seeds=[0], rob_ks=[20], rob_seeds=[0], sens_ks=[5], sens_seeds=[0])
if SMOKE:
    CFG.update(n_folds=2, test_per_class=4, budgets=[1, 5], small_max=5, seeds_small=[0], seeds_big=[0], ft_min_steps=2, ft_epochs=1,
               imagenet=False, rare_classes=[1], rare_ks=[1], rare_others=5, rare_seeds=[0], open_k=3, open_seeds=[0],
               rob_ks=[5], rob_seeds=[0], sens_ks=[5], sens_seeds=[0], sens_lrs=[1e-5], sens_Cs=[1, 10])
print("FAKE" if FAKE else ("SMOKE" if SMOKE else PROFILE), CFG)

# %% [markdown]
# ## 1. Utilities (plain numpy: metrics, splits, result store)

# %%
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NAMES = ["crazing", "inclusion", "patches", "pitting", "rolled-in scale", "scratches"]
PREFIX = {"cr": 0, "in": 1, "pa": 2, "pi": 3, "ps": 3, "ro": 4, "rs": 4, "sc": 5}

def set_seed_np(s): random.seed(s); np.random.seed(s)

def proba_full(clf, X, n):
    """Class probabilities as an (len(X), n) array, even if a class was absent from training."""
    p = clf.predict_proba(X); out = np.zeros((len(X), n)); out[:, clf.classes_] = p; return out

def calib(probs, y):
    """Accuracy, macro-F1, log-loss, expected calibration error (10 bins), share of test images misclassified with confidence >= 0.9."""
    pred, conf = probs.argmax(1), probs.max(1)
    nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    edges = np.linspace(0, 1, 11); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return dict(acc=float((pred == y).mean()), macro_f1=float(f1_score(y, pred, average="macro")), nll=nll, ece=float(ece),
                conf_err=float(((pred != y) & (conf >= 0.9)).mean()))

class Store:
    """Result rows saved to CSV after every step; rows whose key already exists are skipped on a rerun."""
    def __init__(self, name, key_cols):
        self.path, self.key_cols = f"{OUT}/{name}.csv", key_cols
        self.rows = pd.read_csv(self.path).to_dict("records") if os.path.exists(self.path) else []
        self.done = {self._k(r) for r in self.rows}
    def _k(self, d): return tuple(str(d[c]) for c in self.key_cols)
    def has(self, **kw): return self._k(kw) in self.done
    def add(self, row): self.rows.append(row); self.done.add(self._k(row))
    def save(self): pd.DataFrame(self.rows).to_csv(self.path, index=False)

PRED_PATH = f"{OUT}/predictions.pkl"
PRED = pickle.load(open(PRED_PATH, "rb")) if os.path.exists(PRED_PATH) else {}
def save_pred(): pickle.dump(PRED, open(PRED_PATH, "wb"))

def fold_split(f, perm, T):
    test, pool = [], []
    for c in range(6):
        ids = perm[c]; t = list(ids[f * T:(f + 1) * T]); ts = set(t)
        test += t; pool += [i for i in ids if i not in ts]
    return np.array(test), np.array(pool)

def sample_per_class(pool_idx, labels, k_by_class, seed, tag):
    """k_by_class: dict class -> number of images to draw (capped by availability)."""
    out = []
    for c, k in k_by_class.items():
        ids = pool_idx[labels[pool_idx] == c]
        r = np.random.RandomState(10007 * seed + 131 * k + 7 * c + 1009 * tag)
        out += list(r.choice(ids, size=min(k, len(ids)), replace=False))
    return np.array(out, dtype=int)

def sample_val(pool_idx, tr, labels, per_class=60):
    rest = np.setdiff1d(pool_idx, tr); out = []
    for c in range(6):
        ids = rest[labels[rest] == c]; out += list(ids[:per_class])
    return np.array(out, dtype=int)

def timer(): t0 = time.time(); return lambda: f"{time.time() - t0:.0f}s"

# %% [markdown]
# ## 2. Dataset
# Downloaded automatically (public Kaggle copy of the NEU surface defect database; see the report for the official source). It must contain
# 1,800 images, 300 per class. The cell stops if that is not the case.

# %%
def find_images(root):
    exts = (".bmp", ".jpg", ".jpeg", ".png")
    files = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True) if f.lower().endswith(exts)]
    seen, out = set(), []
    for f in sorted(files):
        stem = os.path.splitext(os.path.basename(f))[0].lower()
        if stem in seen: continue
        seen.add(stem); out.append(f)
    return out

if FAKE:                                            # synthetic stripe images, only to test the code
    CFG["data_dir"] = f"{OUT}/fake_data"; os.makedirs(CFG["data_dir"], exist_ok=True)
    rs = np.random.RandomState(0)
    for c in range(6):
        for i in range(24):
            xx = np.tile(np.arange(200), (200, 1)); img = 128 + 60 * np.sin(xx * (0.05 + 0.04 * c)) + rs.randn(200, 200) * 25
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(f"{CFG['data_dir']}/{CLASSES[c]}_{i}.png")
elif not os.path.isdir(CFG["data_dir"]):
    try:
        import kagglehub
        CFG["data_dir"] = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
    except Exception as e:
        raise SystemExit(f"Dataset not found at {CFG['data_dir']} ({e}). Upload/unzip it there or set DATA_DIR.")

paths = find_images(CFG["data_dir"]); labels, keep = [], []
for p in paths:
    pre = ""
    for cand in (os.path.basename(p).lower(), os.path.basename(os.path.dirname(p)).lower()):
        m = re.match(r"([a-z]+)", cand)
        if m and m.group(1)[:2] in PREFIX: pre = m.group(1)[:2]; break
    if pre: keep.append(p); labels.append(PREFIX[pre])
paths, labels = keep, np.array(labels)
print("images:", len(paths), "per class:", dict(zip(CLASSES, np.bincount(labels, minlength=6).tolist())))
print("image sizes:", collections.Counter(Image.open(p).size for p in paths[:200]))
if not SMOKE: assert len(paths) == 1800 and set(np.bincount(labels)) == {300}, "Unexpected dataset size; inspect the folder"

S = CFG["img_size"]
IMGS = [Image.open(p).convert("RGB").resize((S, S), Image.BICUBIC) for p in paths]
_rng = np.random.RandomState(0)
PERM = [_rng.permutation(np.where(labels == c)[0]) for c in range(6)]
assert CFG["n_folds"] * CFG["test_per_class"] <= min(len(p) for p in PERM)
def get_fold(f): return fold_split(f, PERM, CFG["test_per_class"])
print("fold 0: test", len(get_fold(0)[0]), "pool", len(get_fold(0)[1]))

# %% [markdown]
# ## 3. Models and training code
# Real backend: CLIP ViT-B/16 (OpenAI weights, QuickGELU variant) and an ImageNet-21k ViT-B/16 from `timm`. (In the code-logic test these are replaced by numpy stand-ins.)

# %%
if not FAKE:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch, torch.nn as nn, torch.nn.functional as F, torchvision.transforms as T, open_clip
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    AC = "cuda" if DEVICE == "cuda" else "cpu"       # mixed precision only on CUDA
    print("device:", DEVICE)
    def set_seed(s): set_seed_np(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

    class Backbone(nn.Module):
        def __init__(self, kind):
            super().__init__(); self.kind = kind
            if kind == "clip":
                m, _, _ = open_clip.create_model_and_transforms(CFG["clip_model"], pretrained=CFG["clip_pretrained"], device="cpu")
                self.model, self.dim = m, m.visual.output_dim
                self.mean, self.std = open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD
                self.tok = open_clip.get_tokenizer(CFG["clip_model"])
            else:
                import timm
                name = "vit_tiny_patch16_224" if SMOKE else "vit_base_patch16_224.augreg_in21k_ft_in1k"
                self.model = timm.create_model(name, pretrained=not SMOKE, num_classes=0)
                self.dim = self.model.num_features
                cfgp = self.model.pretrained_cfg; self.mean, self.std = cfgp.get("mean", (0.5,) * 3), cfgp.get("std", (0.5,) * 3)
        def forward(self, x): return self.model.encode_image(x) if self.kind == "clip" else self.model(x)

    class AddNoise:
        def __call__(self, x): return x + torch.randn_like(x) * random.uniform(0.0, 0.12)
    def eval_tf(bb): return T.Compose([T.ToTensor(), T.Normalize(bb.mean, bb.std)])
    def train_tf(bb, distort=False):
        t = [T.RandomResizedCrop(S, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
        if distort: t.append(T.RandomApply([T.GaussianBlur(9, sigma=(0.3, 3.0))], p=0.5))
        t.append(T.ToTensor())
        if distort: t.append(T.RandomApply([AddNoise()], p=0.5))
        t.append(T.Normalize(bb.mean, bb.std)); return T.Compose(t)

    class DS(torch.utils.data.Dataset):
        def __init__(self, imgs, ys, tf): self.imgs, self.ys, self.tf = imgs, ys, tf
        def __len__(self): return len(self.imgs)
        def __getitem__(self, i): return self.tf(self.imgs[i]), int(self.ys[i])

    clip = Backbone("clip"); BASE = {"clip": clip}
    TEXT_PARTS = ("model.transformer", "model.token_embedding", "model.positional_embedding", "model.ln_final", "model.text_projection", "model.logit_scale")
    PARTIAL_PREFIX = tuple(f"model.visual.transformer.resblocks.{i}." for i in range(CFG["partial_from"], 12)) + ("model.visual.ln_post", "model.visual.proj")

    @torch.no_grad()
    def embed_clip(imgs, bs=64):
        clip.eval().to(DEVICE); tf = eval_tf(clip); out = []
        for i in range(0, len(imgs), bs):
            x = torch.stack([tf(im) for im in imgs[i:i + bs]]).to(DEVICE)
            with torch.autocast(AC, enabled=DEVICE == "cuda"): f = clip(x)
            out.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def text_weights(names, prompts):
        clip.eval().to(DEVICE); W = []
        for n in names:
            t = clip.tok([p.format(n) for p in prompts]).to(DEVICE)
            e = F.normalize(clip.model.encode_text(t).float(), dim=-1).mean(0); W.append(F.normalize(e, dim=-1).cpu())
        return torch.stack(W).numpy()

    def train_ft(kind, imgs, ys, seed, mode="full", n_classes=6, enc_lr=None, class_weights=None, distort=False):
        """Fine-tune a copy of the pre-trained model. mode='full': all weights; 'partial': last blocks + final norm/projection. Returns a handle."""
        set_seed(seed)
        if kind not in BASE: BASE[kind] = Backbone(kind)
        bb = copy.deepcopy(BASE[kind]).cpu(); head = nn.Linear(bb.dim, n_classes)
        if mode == "partial":
            for n_, p_ in bb.named_parameters(): p_.requires_grad = n_.startswith(PARTIAL_PREFIX)
            assert any(p_.requires_grad for p_ in bb.parameters()), "partial-FT prefix matched nothing"
        bb.to(DEVICE); head.to(DEVICE)
        n = len(imgs); bs = min(CFG["ft_batch"], n); total = max(CFG["ft_min_steps"], CFG["ft_epochs"] * math.ceil(n / bs))
        dl = torch.utils.data.DataLoader(DS(imgs, ys, train_tf(bb, distort)), batch_size=bs, shuffle=True, num_workers=2 if DEVICE == "cuda" else 0)
        opt = torch.optim.AdamW([{"params": [p for n_, p in bb.named_parameters() if p.requires_grad and not n_.startswith(TEXT_PARTS)], "lr": enc_lr or CFG["ft_enc_lr"]},
                                 {"params": head.parameters(), "lr": CFG["ft_head_lr"]}], weight_decay=CFG["ft_wd"])
        warm = max(1, int(0.1 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
        cw = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE) if class_weights is not None else None
        step = 0; bb.train()
        while step < total:
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(AC, enabled=DEVICE == "cuda"): loss = F.cross_entropy(head(bb(x).float()), y, weight=cw)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); step += 1
                if step >= total: break
        return dict(bb=bb, head=head)

    @torch.no_grad()
    def predict_probs(H, imgs, bs=64):
        bb, head = H["bb"], H["head"]; bb.eval(); head.eval(); tf = eval_tf(bb); out = []
        for i in range(0, len(imgs), bs):
            x = torch.stack([tf(im) for im in imgs[i:i + bs]]).to(DEVICE)
            with torch.autocast(AC, enabled=DEVICE == "cuda"): out.append(F.softmax(head(bb(x).float()).float(), dim=1).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    def release(H):
        if H is not None:
            H.clear()
        if DEVICE == "cuda": torch.cuda.empty_cache()
        elif DEVICE == "mps": torch.mps.empty_cache()
else:
    # ---- numpy stand-ins used ONLY to test the control flow of this notebook (no real results come from this branch) ----
    def _feat(im):
        v = np.asarray(im.convert("L").resize((8, 8)), dtype=np.float32).ravel(); v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-6)
    def embed_clip(imgs, bs=64): return np.stack([_feat(i) for i in imgs])
    def text_weights(names, prompts):
        w = np.random.RandomState(len(names) + len(prompts)).randn(len(names), 64); return w / np.linalg.norm(w, axis=1, keepdims=True)
    def train_ft(kind, imgs, ys, seed, mode="full", n_classes=6, enc_lr=None, class_weights=None, distort=False):
        clf = LogisticRegression(C=3.0, max_iter=2000).fit(embed_clip(imgs), ys); return dict(clf=clf, n=n_classes)
    def predict_probs(H, imgs, bs=64): return proba_full(H["clf"], embed_clip(imgs), H["n"])
    def release(H): pass
    def set_seed(s): set_seed_np(s)

# %% [markdown]
# ## 4. Embeddings, occlusion helper, and the experiment recipe
# All 1,800 images are embedded once by frozen CLIP. `METHODS_FT` lists the fine-tuned variants compared throughout.

# %%
t = timer(); Z = embed_clip(IMGS); print("embedded", Z.shape, t())
METHODS_FT = [("finetune_clip", "clip", "full"), ("partial_finetune_clip", "clip", "partial")] + ([("finetune_imagenet_vit", "imagenet", "full")] if CFG["imagenet"] else [])

def fit_lr(tr, y, C=None, **kw): return LogisticRegression(C=C or CFG["lr_C"], max_iter=5000, **kw).fit(Z[tr], y)

def occlusion_map(prob_fn, img, cls, patch=56, stride=28):
    """How much does the predicted probability of `cls` drop when a grey square hides each part of the image?"""
    base = prob_fn([img])[0, cls]; pos = list(range(0, img.size[0] - patch + 1, stride)); ims, idx = [], []
    for yy in pos:
        for xx in pos:
            im = img.copy(); im.paste((128, 128, 128), (xx, yy, xx + patch, yy + patch)); ims.append(im); idx.append((pos.index(yy), pos.index(xx)))
    P = prob_fn(ims)[:, cls]; M = np.zeros((len(pos), len(pos)))
    for (a, b), p in zip(idx, P): M[a, b] = base - p
    return M, float(base)

# %% [markdown]
# ## 5. Experiment 1: learning curves with 5-fold cross-validation
# For each fold, the 60 test images per class are held out and k labelled images per class are drawn from the other 240. Methods: frozen CLIP + logistic regression
# (fixed strength, and with the strength chosen on separate validation images from the pool), partial and full fine-tuning of CLIP, and the ImageNet ViT reference.
# Metrics per run: accuracy, macro-F1, log-loss, calibration error, confident mistakes.

# %%
store1 = Store("exp1_learning_curves", ["method", "k", "fold", "seed"])
def record1(method, P, y, k, fold, seed, **extra):
    store1.add(dict(method=method, k=k, fold=fold, seed=seed, **calib(P, y), **extra))
    PRED[f"e1|{method}|{k}|{fold}|{seed}"] = P.argmax(1).astype(np.int8)

t = timer()
for fold in range(CFG["n_folds"]):
    test_idx, pool_idx = get_fold(fold); y_te = labels[test_idx]; te_imgs = [IMGS[i] for i in test_idx]
    for k in CFG["budgets"]:
        for seed in (CFG["seeds_small"] if k <= CFG["small_max"] else CFG["seeds_big"]):
            tr = sample_per_class(pool_idx, labels, {c: k for c in range(6)}, seed, fold)
            if not store1.has(method="frozen_clip+LR", k=k, fold=fold, seed=seed):
                record1("frozen_clip+LR", proba_full(fit_lr(tr, labels[tr]), Z[test_idx], 6), y_te, k, fold, seed)
            if not store1.has(method="frozen_clip+LR_tuned", k=k, fold=fold, seed=seed):
                val = sample_val(pool_idx, tr, labels)
                if len(set(labels[val])) == 6 and len(val) >= 60:
                    sc = {C: (lambda pv: ((pv.argmax(1) == labels[val]).mean(), -calib(pv, labels[val])["nll"]))(proba_full(fit_lr(tr, labels[tr], C), Z[val], 6)) for C in CFG["C_grid"]}
                    bestC = max(sc, key=sc.get)
                else: bestC = CFG["lr_C"]                      # no spare validation images (k = 240): keep the default
                record1("frozen_clip+LR_tuned", proba_full(fit_lr(tr, labels[tr], bestC), Z[test_idx], 6), y_te, k, fold, seed, C_chosen=bestC)
            for name, kind, mode in METHODS_FT:
                if store1.has(method=name, k=k, fold=fold, seed=seed): continue
                H = train_ft(kind, [IMGS[i] for i in tr], labels[tr], seed, mode)
                record1(name, predict_probs(H, te_imgs), y_te, k, fold, seed); release(H)
            store1.save(); save_pred(); print(f"exp1 fold={fold} k={k:>3} seed={seed} done ({t()})", flush=True)
E1 = pd.DataFrame(store1.rows); print(E1.groupby(["method", "k"])[["acc", "nll", "ece"]].mean().round(4).to_string())

# %% [markdown]
# ## 6. Experiment 2: one rare defect type
# Five defect types have `rare_others` labelled images each; one type has only `k_rare`. We measure how often the rare defect is caught (recall) and how often
# a prediction of it is right (precision). Variants with class-balanced weighting are included, since that is the standard remedy.

# %%
store2 = Store("exp2_rare_defect", ["method", "rare_class", "k_rare", "seed"])
test_idx, pool_idx = get_fold(0); y_te = labels[test_idx]; te_imgs = [IMGS[i] for i in test_idx]
def record2(method, P, rc, kr, seed):
    pred = P.argmax(1); tp = int(((pred == rc) & (y_te == rc)).sum())
    store2.add(dict(method=method, rare_class=rc, k_rare=kr, seed=seed, **calib(P, y_te), rare_recall=tp / max(1, int((y_te == rc).sum())),
                    rare_precision=tp / max(1, int((pred == rc).sum()))))
t = timer()
for rc in CFG["rare_classes"]:
    for kr in CFG["rare_ks"]:
        for seed in CFG["rare_seeds"]:
            kmap = {c: (kr if c == rc else CFG["rare_others"]) for c in range(6)}
            tr = sample_per_class(pool_idx, labels, kmap, seed, 100 + rc); ytr = labels[tr]
            cnt = np.bincount(ytr, minlength=6); w = (len(ytr) / (6 * np.maximum(cnt, 1))).tolist()
            if not store2.has(method="frozen_clip+LR", rare_class=rc, k_rare=kr, seed=seed):
                record2("frozen_clip+LR", proba_full(fit_lr(tr, ytr), Z[test_idx], 6), rc, kr, seed)
            if not store2.has(method="frozen_clip+LR_balanced", rare_class=rc, k_rare=kr, seed=seed):
                record2("frozen_clip+LR_balanced", proba_full(fit_lr(tr, ytr, class_weight="balanced"), Z[test_idx], 6), rc, kr, seed)
            for name, cw in [("finetune_clip", None), ("finetune_clip_weighted", w)]:
                if store2.has(method=name, rare_class=rc, k_rare=kr, seed=seed): continue
                H = train_ft("clip", [IMGS[i] for i in tr], ytr, seed, "full", class_weights=cw)
                record2(name, predict_probs(H, te_imgs), rc, kr, seed); release(H)
            store2.save(); print(f"exp2 rare={CLASSES[rc]} k={kr} seed={seed} done ({t()})", flush=True)
E2 = pd.DataFrame(store2.rows); print(E2.groupby(["method", "rare_class", "k_rare"])[["rare_recall", "rare_precision"]].mean().round(3).to_string())

# %% [markdown]
# ## 7. Experiment 3: a defect type the model has never seen (open-set)
# Train on five defect types, then show the model the sixth. A trustworthy model should be *less confident* on it. We score this with AUROC
# (1.0 = perfectly separates novel from known) using 1 - max probability, and, for frozen CLIP, the distance to the nearest training image.

# %%
store3 = Store("exp3_open_set", ["method", "novel_class", "seed"])
def fpr_at_95(y, s): fpr, tpr, _ = roc_curve(y, s); return float(fpr[np.searchsorted(tpr, 0.95)])
t = timer()
for nc in range(6):
    known = [c for c in range(6) if c != nc]; remap = {c: i for i, c in enumerate(known)}
    is_novel = (labels[test_idx] == nc).astype(int); kn = is_novel == 0
    y_known = np.array([remap[l] for l in labels[test_idx][kn]])
    for seed in CFG["open_seeds"]:
        tr = sample_per_class(pool_idx, labels, {c: CFG["open_k"] for c in known}, seed, 200 + nc)
        ytr = np.array([remap[l] for l in labels[tr]])
        def rec3(method, score, P=None):
            row = dict(method=method, novel_class=nc, seed=seed, auroc=float(roc_auc_score(is_novel, score)), fpr95=fpr_at_95(is_novel, score))
            if P is not None: row["acc_known"] = float((P[kn].argmax(1) == y_known).mean())
            store3.add(row)
        if not store3.has(method="frozen_clip+LR_maxprob", novel_class=nc, seed=seed):
            P = proba_full(fit_lr(tr, ytr), Z[test_idx], 5); rec3("frozen_clip+LR_maxprob", 1 - P.max(1), P)
        if not store3.has(method="frozen_clip_knn_distance", novel_class=nc, seed=seed):
            rec3("frozen_clip_knn_distance", 1 - (Z[test_idx] @ Z[tr].T).max(1))
        for name, kind, mode in METHODS_FT[:2]:
            if store3.has(method=name + "_maxprob", novel_class=nc, seed=seed): continue
            H = train_ft(kind, [IMGS[i] for i in tr], ytr, seed, mode, n_classes=5)
            P = predict_probs(H, te_imgs); rec3(name + "_maxprob", 1 - P.max(1), P); release(H)
        store3.save(); print(f"exp3 novel={CLASSES[nc]} seed={seed} done ({t()})", flush=True)
E3 = pd.DataFrame(store3.rows); print(E3.groupby("method")[["auroc", "fpr95", "acc_known"]].mean().round(3).to_string())

# %% [markdown]
# ## 8. Experiment 4: robustness at several severities, with and without distortion-aware training
# Models trained on clean images (k = `rob_ks`) are tested on blurred, noisy, brighter/darker and low-contrast copies of the same test images.
# `finetune_clip_distaug` is full fine-tuning with random blur and noise added during training.

# %%
SHIFTS = {"clean": ("clean", 0, lambda im: im)}
for s_ in [0.5, 1, 2, 3]: SHIFTS[f"blur_{s_}"] = ("blur", s_, (lambda s_: lambda im: im.filter(ImageFilter.GaussianBlur(s_)))(s_))
for s_ in [5, 15, 30]:
    SHIFTS[f"noise_{s_}"] = ("noise", s_, (lambda s_: lambda im: Image.fromarray(np.clip(np.asarray(im, dtype=np.float32) + np.random.RandomState(0).normal(0, s_, (S, S, 3)), 0, 255).astype(np.uint8)))(s_))
for s_ in [0.6, 0.8, 1.2, 1.4]: SHIFTS[f"brightness_{s_}"] = ("brightness", s_, (lambda s_: lambda im: ImageEnhance.Brightness(im).enhance(s_))(s_))
for s_ in [0.5, 0.75]: SHIFTS[f"contrast_{s_}"] = ("contrast", s_, (lambda s_: lambda im: ImageEnhance.Contrast(im).enhance(s_))(s_))
store4 = Store("exp4_robustness", ["method", "k", "seed", "shift"])
ZS = {n: embed_clip([fn(im) for im in te_imgs]) for n, (_, _, fn) in SHIFTS.items()}
KEEP = {}
ROB_METHODS = METHODS_FT + [("finetune_clip_distaug", "clip", "full")]
t = timer()
for k in CFG["rob_ks"]:
    for seed in CFG["rob_seeds"]:
        tr = sample_per_class(pool_idx, labels, {c: k for c in range(6)}, seed, 300); keep_this = (k == max(CFG["rob_ks"]) and seed == CFG["rob_seeds"][0])
        clf = fit_lr(tr, labels[tr])
        if keep_this: KEEP["frozen"] = clf
        if not store4.has(method="frozen_clip+LR", k=k, seed=seed, shift="clean"):
            for n, (fam, lvl, fn) in SHIFTS.items():
                store4.add(dict(method="frozen_clip+LR", k=k, seed=seed, shift=n, family=fam, level=lvl, acc=float((clf.predict(ZS[n]) == y_te).mean())))
        for name, kind, mode in ROB_METHODS:
            if store4.has(method=name, k=k, seed=seed, shift="clean") and not keep_this: continue
            if store4.has(method=name, k=k, seed=seed, shift="clean") and name in KEEP: continue
            H = train_ft(kind, [IMGS[i] for i in tr], labels[tr], seed, mode, distort=name.endswith("distaug"))
            if not store4.has(method=name, k=k, seed=seed, shift="clean"):
                for n, (fam, lvl, fn) in SHIFTS.items():
                    P = predict_probs(H, [fn(im) for im in te_imgs])
                    store4.add(dict(method=name, k=k, seed=seed, shift=n, family=fam, level=lvl, acc=float((P.argmax(1) == y_te).mean())))
            if keep_this and name in ("finetune_clip", "finetune_clip_distaug"): KEEP[name] = H
            else: release(H)
        store4.save(); print(f"exp4 k={k} seed={seed} done ({t()})", flush=True)
E4 = pd.DataFrame(store4.rows); print(E4[E4.k == max(CFG["rob_ks"])].pivot_table(index="shift", columns="method", values="acc").round(3).to_string())

# %% [markdown]
# ## 9. Experiment 5: do the conclusions depend on the fixed settings?
# Full fine-tuning at three encoder learning rates, and the frozen classifier at four regularisation strengths (fold 0). If the ranking of methods does not change, it is robust.

# %%
store5 = Store("exp5_sensitivity", ["method", "k", "seed", "param"])
t = timer()
for k in CFG["sens_ks"]:
    for seed in CFG["sens_seeds"]:
        tr = sample_per_class(pool_idx, labels, {c: k for c in range(6)}, seed, 400)
        for C in CFG["sens_Cs"]:
            if not store5.has(method="frozen_clip+LR", k=k, seed=seed, param=f"C={C}"):
                store5.add(dict(method="frozen_clip+LR", k=k, seed=seed, param=f"C={C}", **calib(proba_full(fit_lr(tr, labels[tr], C), Z[test_idx], 6), y_te)))
        for lr in CFG["sens_lrs"]:
            if store5.has(method="finetune_clip", k=k, seed=seed, param=f"enc_lr={lr}"): continue
            H = train_ft("clip", [IMGS[i] for i in tr], labels[tr], seed, "full", enc_lr=lr)
            store5.add(dict(method="finetune_clip", k=k, seed=seed, param=f"enc_lr={lr}", **calib(predict_probs(H, te_imgs), y_te))); release(H)
        store5.save(); print(f"exp5 k={k} seed={seed} done ({t()})", flush=True)
E5 = pd.DataFrame(store5.rows); print(E5.groupby(["method", "k", "param"])[["acc", "nll"]].mean().round(4).to_string())

# %% [markdown]
# ## 10. Experiment 6: zero-shot CLIP, done properly
# v4 gave 12.5% (below the 16.7% of guessing). Here, eight prompt/name combinations are compared **on the training pool only**; the best is then scored on the
# test images. All combinations are reported, so nothing is hidden. The confusion matrix shows what CLIP actually predicts.

# %%
NAME_SETS = {"plain": NAMES, "descriptive": ["fine surface cracks", "dark embedded impurity spots", "irregular patchy blotches", "small pits and holes", "rolled-in scale streaks", "long thin scratch lines"]}
PROMPT_SETS = {"photo": ["a photo of {}."], "microscope": ["a microscope image of {} on a steel surface."],
               "texture": ["a grayscale texture image of hot-rolled steel with {}."],
               "ensemble": ["a photo of {}, a defect on a hot-rolled steel surface.", "a grayscale microscope image of {} on steel.", "steel strip with {}."]}
rows6 = []
for (nn, names), (pn, prompts) in itertools.product(NAME_SETS.items(), PROMPT_SETS.items()):
    W = text_weights(names, prompts)
    rows6.append(dict(names=nn, prompts=pn, pool_acc=float(((Z[pool_idx] @ W.T).argmax(1) == labels[pool_idx]).mean()),
                      test_acc=float(((Z[test_idx] @ W.T).argmax(1) == y_te).mean()), all_images_acc=float(((Z @ W.T).argmax(1) == labels).mean())))
E6 = pd.DataFrame(rows6).sort_values("pool_acc", ascending=False); E6["chosen_on_pool"] = [i == 0 for i in range(len(E6))]
E6.to_csv(f"{OUT}/exp6_zero_shot.csv", index=False); print(E6.round(3).to_string(index=False))
best = E6.iloc[0]; W = text_weights(NAME_SETS[best["names"]], PROMPT_SETS[best["prompts"]]); pz = (Z[test_idx] @ W.T).argmax(1)
cm = np.zeros((6, 6), int)
for a, b in zip(y_te, pz): cm[a, b] += 1
np.savetxt(f"{OUT}/exp6_zero_shot_confusion.csv", cm, fmt="%d", delimiter=","); print("zero-shot confusion (rows = true class):\n", cm)

# %% [markdown]
# ## 11. Experiment 7: how unusual are the test images? (difficulty / near-duplicate check)
# For every test image: cosine similarity of its CLIP embedding to the closest training-pool image. Later analysis checks whether the *most unusual* test images are
# classified worse (accuracy by similarity band), which is a harder test than average accuracy.

# %%
sim = {}
for fold in range(CFG["n_folds"]):
    ti, pi = get_fold(fold); S_ = Z[ti] @ Z[pi].T; sim[f"test_idx_{fold}"] = ti; sim[f"nn_sim_{fold}"] = S_.max(1); sim[f"nn_pool_{fold}"] = pi[S_.argmax(1)]
np.savez(f"{OUT}/nn_similarity.npz", **sim)
s0, ti0 = sim["nn_sim_0"], sim["test_idx_0"]; order = np.argsort(s0); sel = list(order[:4]) + list(order[-4:])
fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
for j, i in enumerate(sel):
    ax[0, j].imshow(IMGS[ti0[i]]); ax[0, j].set_title(f"test: {CLASSES[labels[ti0[i]]]}\nsim {s0[i]:.3f}", fontsize=7)
    ax[1, j].imshow(IMGS[sim["nn_pool_0"][i]]); ax[1, j].set_title("closest training image", fontsize=7)
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.suptitle("Left four: least similar test images. Right four: most similar test images (fold 0)", fontsize=9); plt.tight_layout()
plt.savefig(f"{OUT}/nearest_training_images.png", dpi=170); plt.show()
print("similarity quantiles [min,25%,50%,75%,max]:", np.quantile(np.concatenate([sim[f"nn_sim_{f}"] for f in range(CFG["n_folds"])]), [0, .25, .5, .75, 1]).round(4))

# %% [markdown]
# ## 12. Experiment 8: occlusion maps (where does each model look?)
# A grey square is slid over the image. The brighter the map, the more the predicted probability of the true class drops when that area is hidden.

# %%
try:
    if "finetune_clip" not in KEEP:
        tr = sample_per_class(pool_idx, labels, {c: max(CFG["rob_ks"]) for c in range(6)}, CFG["rob_seeds"][0], 300)
        KEEP["frozen"] = fit_lr(tr, labels[tr]); KEEP["finetune_clip"] = train_ft("clip", [IMGS[i] for i in tr], labels[tr], CFG["rob_seeds"][0], "full")
    fns = {"Frozen CLIP + LR": lambda ims: proba_full(KEEP["frozen"], embed_clip(ims), 6), "Fully fine-tuned CLIP": lambda ims: predict_probs(KEEP["finetune_clip"], ims)}
    if "finetune_clip_distaug" in KEEP: fns["Fine-tuned with blur/noise aug."] = lambda ims: predict_probs(KEEP["finetune_clip_distaug"], ims)
    picks = [int(np.where(labels[test_idx] == c)[0][0]) for c in range(6)]
    fig, ax = plt.subplots(6, 1 + len(fns), figsize=(3 * (1 + len(fns)), 17))
    for r, (c, pi_) in enumerate(zip(range(6), picks)):
        im = te_imgs[pi_]; ax[r, 0].imshow(im); ax[r, 0].set_title(CLASSES[c], fontsize=9); ax[r, 0].axis("off")
        for j, (nm, fn) in enumerate(fns.items(), 1):
            M, base = occlusion_map(fn, im, c)
            ax[r, j].imshow(im); ax[r, j].imshow(np.kron(M, np.ones((32, 32))), cmap="inferno", alpha=0.55, extent=(0, S, S, 0))
            ax[r, j].set_title(f"{nm}\np(true)={base:.2f}", fontsize=8); ax[r, j].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/occlusion_maps.png", dpi=150); plt.show()
except Exception as e:
    print("Occlusion maps failed (the rest of the results are unaffected):", repr(e))

# %% [markdown]
# ## 13. Package everything
# Download `vit_results_v5.zip` and send it back. It contains all CSVs, predictions and figures.

# %%
json.dump(dict(CFG=CFG, classes=CLASSES, profile=("fake" if FAKE else "smoke" if SMOKE else PROFILE)), open(f"{OUT}/config.json", "w"), default=str)
shutil.make_archive("vit_results_v5", "zip", OUT)
print("wrote vit_results_v5.zip with:", sorted(os.listdir(OUT)))
