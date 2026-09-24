"""
Data: download, DINOv2 feature extraction, loading, forget/retain splits.
────────────────────────────────────────────────────────────────────────────
Heads (linear / mlp1 / mlp2) read cached DINOv2-small CLS features;
ResNet-18 reads normalised raw CIFAR images. A *cell* is one
`{dataset}_{arch}_{scenario}_{k}` configuration:

    CLASS     forget every training sample of class k
    SUBCLASS  train on superclasses, forget fine class k
    RANDOM    forget k training samples drawn with np.random.default_rng(split)

    python -m unlearning.data --dataset cifar10 cifar100 [--no-features]
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from unlearning import config as CFG

_NORM = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))}


# ── Download / extraction ────────────────────────────────────────────────────

def _tv_dataset(dataset, train, transform=None, download=False):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    return cls(CFG.RAW_DIR, train=train, transform=transform, download=download)


def raw_ready(dataset):
    sub = "cifar-10-batches-py" if dataset == "cifar10" else "cifar-100-python"
    return os.path.isdir(os.path.join(CFG.RAW_DIR, sub))


def features_ready(dataset):
    d = os.path.join(CFG.FEATURE_DIR, dataset)
    return all(os.path.exists(os.path.join(d, f"{s}_{k}.pt"))
               for s in ("train", "test") for k in ("features", "labels"))


def download_raw(dataset):
    os.makedirs(CFG.RAW_DIR, exist_ok=True)
    for train in (True, False):
        _tv_dataset(dataset, train, download=True)


def fetch_resnet_weights():
    """Cache the ImageNet ResNet-18 weights (torchvision hub cache)."""
    from torchvision.models import ResNet18_Weights, resnet18
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


class _Collate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        px = self.processor(images=list(imgs), return_tensors="pt")["pixel_values"]
        return px, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def _encode(model, processor, ds, device, batch_size, num_workers):
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, collate_fn=_Collate(processor),
                     pin_memory=(device.type == "cuda"))
    feats, labels = [], []
    for px, y in ldr:
        out = model(pixel_values=px.to(device, non_blocking=True))
        cls = getattr(out, "pooler_output", None)
        if cls is None:
            cls = out.last_hidden_state[:, 0, :]
        feats.append(cls.detach().cpu())
        labels.append(y)
    return torch.cat(feats, 0).float(), torch.cat(labels, 0).long()


def extract_features(dataset, device=None, batch_size=64, num_workers=2):
    """DINOv2-small CLS token of every CIFAR image (HF processor: resize 256,
    centre crop 224, ImageNet normalisation) -> FEATURE_DIR/<dataset>/*.pt."""
    from transformers import AutoImageProcessor, AutoModel
    device = torch.device(device) if device else default_device()
    processor = AutoImageProcessor.from_pretrained(CFG.DINOV2_MODEL)
    model = AutoModel.from_pretrained(CFG.DINOV2_MODEL).to(device).eval()
    out = os.path.join(CFG.FEATURE_DIR, dataset)
    os.makedirs(out, exist_ok=True)
    for split, train in (("train", True), ("test", False)):
        ds = _tv_dataset(dataset, train)
        try:
            X, y = _encode(model, processor, ds, device, batch_size, num_workers)
        except NotImplementedError:          # MPS without the CPU fallback enabled
            print("[data] a DINOv2 kernel is missing on MPS; extracting on CPU")
            device = torch.device("cpu")
            model = model.to(device)
            X, y = _encode(model, processor, ds, device, batch_size, num_workers)
        torch.save(X, os.path.join(out, f"{split}_features.pt"))
        torch.save(y, os.path.join(out, f"{split}_labels.pt"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"dataset": dataset, "model": CFG.DINOV2_MODEL,
                   "device": str(device)}, f, indent=1)


def prepare(dataset, features=True, resnet=True, device=None):
    """Everything a cell of `dataset` needs, downloaded once."""
    need_raw = resnet or (features and not features_ready(dataset))
    if need_raw and not raw_ready(dataset):
        print(f"[data] downloading {dataset} -> {CFG.RAW_DIR}")
        download_raw(dataset)
    if features and not features_ready(dataset):
        print(f"[data] extracting {CFG.DINOV2_MODEL} features for {dataset} "
              f"(one-off; minutes on a GPU, longer on CPU)")
        extract_features(dataset, device=device)
    if resnet:
        fetch_resnet_weights()


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Loading ──────────────────────────────────────────────────────────────────

def load_features(dataset):
    d = os.path.join(CFG.FEATURE_DIR, dataset)
    if not features_ready(dataset):
        raise FileNotFoundError(f"no DINOv2 features in {d}: run "
                                f"`python -m unlearning.data --dataset {dataset}`")
    ld = lambda n: torch.load(os.path.join(d, n), weights_only=False)
    return (ld("train_features.pt").float(), ld("train_labels.pt").long(),
            ld("test_features.pt").float(), ld("test_labels.pt").long())


def _to_tensors(ds):
    Xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=CFG.BATCH_SIZE, shuffle=False):
        Xs.append(x)
        ys.append(y)
    return torch.cat(Xs).float(), torch.cat(ys).long()


def load_raw(dataset):
    """Normalised 3x32x32 float tensors (no augmentation)."""
    import torchvision.transforms as T
    if not raw_ready(dataset):
        raise FileNotFoundError(f"no raw {dataset} in {CFG.RAW_DIR}: run "
                                f"`python -m unlearning.data --dataset {dataset}`")
    tf = T.Compose([T.ToTensor(), T.Normalize(*_NORM[dataset])])
    X_tr, y_tr = _to_tensors(_tv_dataset(dataset, True, tf))
    X_te, y_te = _to_tensors(_tv_dataset(dataset, False, tf))
    return X_tr, y_tr, X_te, y_te


_CACHE = {}


def load(dataset, arch):
    """(X_train, y_train, X_test, y_test), memoised (read-only, shared)."""
    key = (dataset, "resnet18" if arch == "resnet18" else "head")
    if key not in _CACHE:
        if len(_CACHE) >= 2:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = load_raw(dataset) if arch == "resnet18" else load_features(dataset)
    return _CACHE[key]


# ── Cells ────────────────────────────────────────────────────────────────────

@dataclass
class Cell:
    name: str
    dataset: str
    arch: str
    scenario: str
    to_forget: int
    split_seed: int
    X_train: torch.Tensor = field(repr=False)
    y_train: torch.Tensor = field(repr=False)     # training labels (superclass under SUBCLASS)
    X_test: torch.Tensor = field(repr=False)
    y_test: torch.Tensor = field(repr=False)
    forget_mask: torch.Tensor = field(repr=False)
    num_classes: int = 0
    label: str = ""
    y_train_fine: torch.Tensor = field(default=None, repr=False)
    y_test_fine: torch.Tensor = field(default=None, repr=False)

    @property
    def retain_mask(self):
        return ~self.forget_mask

    @property
    def X_forget(self):
        return self.X_train[self.forget_mask]

    @property
    def y_forget(self):
        return self.y_train[self.forget_mask]

    @property
    def X_retain(self):
        return self.X_train[self.retain_mask]

    @property
    def y_retain(self):
        return self.y_train[self.retain_mask]

    def loaders(self):
        """(forget_loader, retain_loader) handed to the unlearning methods."""
        f = DataLoader(TensorDataset(self.X_forget, self.y_forget),
                       batch_size=CFG.BATCH_SIZE, shuffle=True)
        r = DataLoader(TensorDataset(self.X_retain, self.y_retain),
                       batch_size=CFG.BATCH_SIZE, shuffle=True)
        return f, r

    def retain_eval_idx(self):
        rng = np.random.default_rng(CFG.RETAIN_EVAL_SEED)
        n_r = int(self.retain_mask.sum())
        return np.sort(rng.choice(n_r, min(CFG.RETAIN_EVAL_N, n_r), replace=False))

    def eval_sets(self):
        """{name: (X, y)}: full test set, full forget set, retain subsample."""
        idx = torch.from_numpy(self.retain_eval_idx())
        return {"test": (self.X_test, self.y_test),
                "forget": (self.X_forget, self.y_forget),
                "retain": (self.X_retain[idx], self.y_retain[idx])}


def build_cell(cell, split_seed=CFG.SPLIT_SEED):
    dataset, arch, scenario, k = CFG.parse_cell(cell)
    X_tr, y_tr, X_te, y_te = load(dataset, arch)
    n_fine = CFG.NUM_CLASSES[dataset]
    y_train, y_test, num_classes = y_tr, y_te, n_fine
    if scenario == "CLASS":
        forget = (y_tr == k)
        label = (f"CLASS {k} ({CFG.class_name(dataset, k)}), "
                 f"{int(forget.sum())} samples")
    elif scenario == "SUBCLASS":
        smap = CFG.superclass_map(dataset)
        mapping = torch.zeros(len(smap), dtype=torch.long)
        for fine, coarse in smap.items():
            mapping[fine] = coarse
        y_train, y_test = mapping[y_tr], mapping[y_te]
        forget = (y_tr == k)
        num_classes = len(CFG.superclass_names(dataset))
        sup = int(smap[k])
        label = (f"SUBCLASS {CFG.class_name(dataset, k)} (fine {k}) "
                 f"→ superclass {CFG.superclass_names(dataset)[sup]} ({sup}), "
                 f"{int(forget.sum())} samples")
    else:
        rng = np.random.default_rng(split_seed)
        nf = max(1, int(k))
        chosen = rng.choice(len(y_tr), nf, replace=False)
        forget = torch.zeros(len(y_tr), dtype=torch.bool)
        forget[chosen] = True
        label = f"RANDOM {nf} samples (split_seed={split_seed})"
    return Cell(name=cell, dataset=dataset, arch=arch, scenario=scenario,
                to_forget=k, split_seed=split_seed, X_train=X_tr,
                y_train=y_train, X_test=X_te, y_test=y_test,
                forget_mask=forget, num_classes=num_classes, label=label,
                y_train_fine=y_tr, y_test_fine=y_te)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", default=CFG.DATASETS, choices=CFG.DATASETS)
    p.add_argument("--no-features", action="store_true",
                   help="skip DINOv2 extraction (ResNet cells only need raw images)")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    for ds in args.dataset:
        prepare(ds, features=not args.no_features, device=args.device)
    print("[data] ready")


if __name__ == "__main__":
    main()
