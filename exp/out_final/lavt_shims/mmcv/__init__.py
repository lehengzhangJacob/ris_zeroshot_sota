"""Minimal mmcv shim for LAVT inference on the glod-ris environment.

LAVT's vendored mmcv_custom.checkpoint imports mmcv only for checkpoint I/O
helpers.  test.py loads the checkpoint with torch.load directly, so the shim
exposes only the symbols needed for import and falls back to torch for file I/O.
"""
__version__ = "0.0.0-lavt-shim"
