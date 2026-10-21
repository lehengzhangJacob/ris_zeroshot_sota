"""Multi-target battery for the official GRES evaluator (ReLA code).

The 5-case battery in metric_check.py covers single-target, no-target and
targeted-FP conventions.  This battery adds the case that matters for the
GREx paper: two target instances in one expression, predicted as a union,
partially, or not at all.  Numeric output is written to
exp/literature/runs/multitarget_check.json.
"""
import sys, types, json
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
d2 = types.ModuleType("detectron2"); d2.__path__ = []
ev = types.ModuleType("detectron2.evaluation"); ev.__path__ = []
evm = types.ModuleType("detectron2.evaluation.evaluator")
class DatasetEvaluator: pass
evm.DatasetEvaluator = DatasetEvaluator
ut = types.ModuleType("detectron2.utils"); ut.__path__ = []
comm = types.ModuleType("detectron2.utils.comm")
comm.all_gather = lambda x: [x]
comm.is_main_process = lambda: True
comm.synchronize = lambda: None
io = types.ModuleType("detectron2.utils.file_io")
class PathManager:
    @staticmethod
    def open(p, mode="r"):
        return open(p, mode)
io.PathManager = PathManager
for name, mod in [("detectron2", d2), ("detectron2.evaluation", ev),
                  ("detectron2.evaluation.evaluator", evm),
                  ("detectron2.utils", ut), ("detectron2.utils.comm", comm),
                  ("detectron2.utils.file_io", io)]:
    sys.modules[name] = mod

import importlib.util
spec = importlib.util.spec_from_file_location(
    "refer_evaluation", ROOT / "exp/literature/repos/rela/gres_model/evaluation/refer_evaluation.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def make_case(pred_cells, nt=False):
    """GT has two 2x2 squares; pred_cells lists which squares are predicted."""
    g = np.zeros((12, 12), np.int8)
    g[0:2, 0:2] = 1
    g[6:8, 6:8] = 1
    p = np.zeros((12, 12), np.int8)
    for (r, c) in pred_cells:
        p[r:r+2, c:c+2] = 1
    seg = torch.zeros(2, 12, 12)
    seg[1] = torch.from_numpy(p.astype(np.float32))
    inp = {"image_id": 0, "source": "grefcoco",
           "gt_mask_merged": torch.from_numpy(g), "sentence": {"raw": "two squares"},
           "empty": False}
    out = {"ref_seg": seg, "nt_label": torch.tensor([1.0, 0.0]) if not nt else torch.tensor([0.0, 1.0])}
    return inp, out


cases = {
    "two_target_union_exact": [((0, 0)), ((6, 6))],
    "two_target_partial": [((0, 0))],
    "two_target_wrong_plus_miss": [((10, 10))],
    "two_target_abstain_FP": [((0, 0)), ((6, 6))],
}
e = mod.ReferEvaluator("grefcoco", distributed=False, output_dir=None)
e.reset()
for name, cells in cases.items():
    i, o = make_case(cells, nt=(name == "two_target_abstain_FP"))
    e.process([i], [o])
res = e.evaluate()["grefcoco"]
out = {
    "cases": list(cases),
    "official": {k: round(float(v), 4) for k, v in res.items()},
    "note": ("two-target GT union (8 px each square, 16 px total). Exact union "
             "should give cIoU 100; partial gives I=4, U=12 -> 33.33; wrong-plus-miss "
             "gives 0; the abstention-FP case keeps the discarded union in the "
             "denominator, so it is not credited."),
}
print(json.dumps(out, indent=2))
(ROOT / "exp/literature/runs/multitarget_check.json").write_text(json.dumps(out, indent=2))
