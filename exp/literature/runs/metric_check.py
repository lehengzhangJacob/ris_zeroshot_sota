"""Run the official GRES evaluator (ReLA) on a hand-built case battery.

Detectron2 is not installed; the evaluator only needs DatasetEvaluator and
comm/io helpers, so a minimal stub is installed in sys.modules first.  The
five cases isolate the official cIoU/gIoU/N_acc/T_acc conventions.
"""
import sys, types, json, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]

# --- minimal detectron2 stub -------------------------------------------------
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

import importlib.util, torch
import numpy as np
spec = importlib.util.spec_from_file_location(
    "refer_evaluation",
    ROOT / "exp/literature/repos/rela/gres_model/evaluation/refer_evaluation.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def case(gt, pred, pred_nt):
    g = np.zeros((8, 8), np.int8); g[gt[0]:gt[0]+2, gt[1]:gt[1]+2] = 1
    p = np.zeros((8, 8), np.int8)
    if pred is not None:
        p[pred[0]:pred[0]+2, pred[1]:pred[1]+2] = 1
    seg = torch.zeros(2, 8, 8)
    seg[1] = torch.from_numpy(p.astype(np.float32))
    nt = torch.tensor([1.0, 0.0]) if not pred_nt else torch.tensor([0.0, 1.0])
    inp = {"image_id": 0, "source": "grefcoco",
           "gt_mask_merged": torch.from_numpy(g), "sentence": {"raw": "x"},
           "empty": bool(g.sum() == 0)}
    out = {"ref_seg": seg, "nt_label": nt}
    return inp, out

cases = [
    ("targeted_exact",      (0, 0), (0, 0), False),
    ("targeted_wrong",      (0, 0), (4, 4), False),
    ("targeted_abstain_FP", (0, 0), (4, 4), True),
    ("notarget_TP",         None,   None,   True),
    ("notarget_FN",         None,   (4, 4), False),
]
e = mod.ReferEvaluator("grefcoco", distributed=False, output_dir=None)
e.reset()
for name, gt, pred, nt in cases:
    i, o = case(gt or (0, 0), pred, nt)
    if gt is None:
        i["gt_mask_merged"] = torch.zeros(8, 8)
        i["empty"] = True
    e.process([i], [o])
res = e.evaluate()
r = dict(res["grefcoco"])
# our convention: an abstention contributes I=0, U=gt_area, iou=0/1
our_I = 4.0; our_U = 4 + 8 + 4 + 0 + 4
our = dict(cIoU=100 * our_I / our_U, gIoU=100 * (1 + 0 + 0 + 1 + 0) / 5,
           N_acc=100.0 * 1 / 2, T_acc=100.0 * 2 / 3)
out = dict(cases=[c[0] for c in cases], official={k: round(v, 4) for k, v in r.items()},
           our_convention_on_same_cases={k: round(v, 4) for k, v in our.items()},
           note=("official targeted-FP case adds the discarded mask's union to the "
                 "cIoU denominator (I=0); an empty-prediction abstention (our "
                 "convention) adds only the GT area. gIoU agrees; N_acc/T_acc agree up to the official code's fraction (not percent) scale."))
print(json.dumps(out, indent=2))
(ROOT / "exp/literature/runs/metric_check.json").write_text(json.dumps(out, indent=2))
