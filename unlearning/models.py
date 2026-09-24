"""
Models and reference training.
────────────────────────────────────────────────────────────────────────────
    linear   Linear(384, C)                         on DINOv2-small features
    mlp1     384 -> 256 -> C                         (ReLU)
    mlp2     384 -> 256 -> 256 -> C                  (ReLU)
    resnet18 ImageNet ResNet-18, 32x32 inputs upsampled to 64x64, new C-way fc
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from unlearning import config as CFG


class CifarResNet18(nn.Module):
    def __init__(self, num_classes=10, input_size=CFG.RESNET_INPUT_SIZE):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.input_size = int(input_size)
        net.fc = nn.Linear(512, num_classes)
        # backbone[j]: 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4-7 layer1-4, 8 avgpool
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3,
                                      net.layer4, net.avgpool)
        self.fc = net.fc
        self.num_classes = num_classes

    def _maybe_resize(self, x):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear",
                              align_corners=False)
        return x

    def forward(self, x):
        return self.fc(self.backbone(self._maybe_resize(x)).flatten(1))

    def __getitem__(self, idx):
        """model[-1] is the classifier, as for the nn.Sequential heads."""
        if idx == -1:
            return self.fc
        raise IndexError("CifarResNet18 only supports model[-1]")


def make_model(arch, num_classes):
    arch = arch.replace("dinov2-", "")
    if arch == "resnet18":
        return CifarResNet18(num_classes=num_classes)
    d, h = CFG.FEATURE_DIM, CFG.HIDDEN_DIM
    if arch == "linear":
        return nn.Sequential(nn.Linear(d, num_classes))
    if arch == "mlp1":
        return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, num_classes))
    if arch == "mlp2":
        return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h),
                             nn.ReLU(), nn.Linear(h, num_classes))
    raise ValueError(f"unknown arch {arch!r}")


def fresh(base_weights, device, arch, num_classes):
    m = make_model(arch, num_classes).to(device)
    m.load_state_dict(base_weights)
    return m


# ── Reference training ───────────────────────────────────────────────────────

def train_head(model, X, y, epochs, device, seed):
    """Adam + cosine + CE, reshuffled each epoch; tensors held on the device,
    batches are slices of a CPU-drawn permutation (device-independent order)."""
    model = model.to(device)
    X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
    opt = optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    n = X.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, CFG.BATCH_SIZE):
            idx = perm[i:i + CFG.BATCH_SIZE]
            opt.zero_grad()
            crit(model(X[idx]), y[idx]).backward()
            opt.step()
        sch.step()
    return model


def train_resnet(model, loader, max_epochs, device, acc_target, verbose=False):
    """Adam + cosine over `max_epochs`, stopped as soon as the running train
    accuracy reaches `acc_target` (%). Returns (model, stats)."""
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=CFG.RESNET_LR, weight_decay=CFG.RESNET_WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.CrossEntropyLoss()
    acc, ep, t0 = float("nan"), 0, time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            crit(out, y).backward()
            opt.step()
            correct += (out.detach().argmax(1) == y).sum().item()
            total += y.numel()
        sch.step()
        acc = 100.0 * correct / max(1, total)
        if verbose:
            print(f"      epoch {ep:2d}  train_acc={acc:6.2f}%  ({time.time()-t0:.0f}s)")
        if acc >= float(acc_target):
            break
    return model, {"epochs_run": ep, "acc_train": acc,
                   "acc_target": float(acc_target),
                   "acc_target_reached": bool(acc >= float(acc_target))}


def train_reference(arch, model, X, y, train_epochs, device, seed, verbose=False):
    """Heads: `train_epochs` epochs. ResNet: until 99 % train accuracy,
    `train_epochs` being the cap. Returns (model, stats)."""
    if arch == "resnet18":
        torch.manual_seed(int(seed))
        loader = DataLoader(TensorDataset(X, y), batch_size=CFG.RESNET_BATCH_SIZE,
                            shuffle=True)
        return train_resnet(model, loader, train_epochs, device,
                            CFG.RESNET_TRAIN_ACC_TARGET, verbose=verbose)
    model = train_head(model, X, y, train_epochs, device, seed)
    return model, {"epochs_run": train_epochs, "acc_train": None,
                   "acc_target": None, "acc_target_reached": None}


def default_train_epochs(arch):
    return CFG.RESNET_MAX_EPOCHS if arch == "resnet18" else CFG.EPOCHS
