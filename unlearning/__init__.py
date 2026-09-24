"""Behavioral unlearning from proxies: references, methods, per-epoch metrics."""
import os

# MPS lacks a few kernels (bicubic upsampling in DINOv2, some linalg); fall
# back to CPU for those. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
