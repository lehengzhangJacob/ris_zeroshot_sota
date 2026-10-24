#!/usr/bin/env python
"""EXP-190 — pooled-cIoU-oriented selector head.

Motivation: the shipped classifier head maximizes per-proposal
correctness and measures 64.66 pooled oIoU on RefCOCO val.  PinPoint
(training-free, Qwen2.5-VL + SAM2) reports 75.9 cIoU on the same split;
the pool oracle over our cached SAM pool is 80.77.  This experiment fits
a continuous-IoU regressor with sample weights proportional to the GT
area (the cIoU weighting), selects the weighting by a ref-id hash holdout
carved from the TRAIN tune records, then applies the selected model to the
official val split exactly once.

No official validation/test record is used for fitting or model selection.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/msj_team/Jacob/research/ris')
sys.path.insert(0, str(ROOT / 'exp'))
spec = importlib.util.spec_from_file_location('e159', ROOT / 'exp/exp159_head_rebuild.py')
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier


def holdout_flag(r):
    return (hash((int(r['ref_id']), int(r.get('sent_idx', 0)))) % 5) == 0


def tune_matrices(dataset, te, box_path):
    from glod.data import RefDataset
    from vlm_boxes import QwenBoxer
    ds = RefDataset(E.REPRO / 'data/refer', E.DS_NAME[dataset], 'train')
    ref_by_id = {r['ref_id']: r for r in ds.refs}
    store, order = {}, []
    boxer = QwenBoxer.__new__(QwenBoxer)
    if box_path is None:
        box_path = boxer._path
    recs = []
    for r, fn, masks, fg in E.iterate_records(dataset, 'tune', ref_by_id, ds, store, order, te, None):
        bp = box_path(fn, r['parsed']['sentence'].strip())
        box = json.loads(bp.read_text())['box'] if bp.is_file() else None
        our = fg @ te(r['parsed']['sentence'])
        feats = E.arm_c_features(masks, fg, r, box, our)
        X = E.feat_matrix(feats, E.ALL_COLS)
        y = np.asarray(r['gt_iou'][:len(fg)], np.float32)
        recs.append(dict(X=X, y=y, area=float(r.get('gt_area', 0.0)), hold=holdout_flag(r)))
    X = np.concatenate([x['X'] for x in recs]); y = np.concatenate([x['y'] for x in recs])
    w_area = np.concatenate([np.full(len(x['y']), max(x['area'], 1e-6)) for x in recs])
    w_sqrt = np.sqrt(w_area)
    hold_idx = np.concatenate([np.full(len(x['y']), x['hold'], bool) for x in recs])
    print('tune rows', len(y), 'holdout rows', int(hold_idx.sum()), 'pos', float((y >= .5).mean()), flush=True)
    return X, y, w_area, w_sqrt, hold_idx, recs


def fit_candidate(kind, X, y, w_area, w_sqrt):
    if kind == 'reg':
        return HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=.08, random_state=0).fit(X, y)
    if kind == 'reg_area':
        return HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=.08, random_state=0).fit(X, y, sample_weight=w_area)
    if kind == 'reg_sqrt':
        return HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=.08, random_state=0).fit(X, y, sample_weight=w_sqrt)
    if kind == 'clf_area':
        return HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=.08, random_state=0).fit(X, (y >= .5).astype(np.int8), sample_weight=w_area)
    if kind == 'clf_plain':
        return HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=.08, random_state=0).fit(X, (y >= .5).astype(np.int8))
    raise ValueError(kind)


def score_proposals(model, X, kind):
    if kind.startswith('clf'):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def holdout_metric(model, kind, recs):
    I = U = 0.0; ious = []
    for rec in recs:
        if not rec['hold']:
            continue
        pred = score_proposals(model, rec['X'], kind)
        j = int(np.argmax(pred))
        inter = float(rec['inter'][j]); union = float(rec['union'][j])
        I += inter; U += union; ious.append(float(rec['y'][j]))
    return 100.0 * I / max(U, 1e-9), 100.0 * float(np.mean(ious))


def add_gt_arrays(recs, dataset):
    # recs currently lack inter/union; reload the tune jsonl in order
    path = ROOT / E.TUNE[dataset]
    rows = [json.loads(l) for l in open(path)]
    # iterate_records skipped rows without cache/ref; match by ref_id/sent_idx
    by = {(int(r['ref_id']), int(r.get('sent_idx', 0))): r for r in rows}
    for rec in recs:
        j = by[(int(rec['ref_id']), int(rec.get('sent_idx', 0)))]
        rec['inter'] = j['inter_gt']; rec['union'] = j['union_gt']


def val_eval(dataset, te, model, kind, box_path):
    from glod.data import RefDataset
    from vlm_boxes import QwenBoxer
    ds = RefDataset(E.REPRO / 'data/refer', E.DS_NAME[dataset], 'val')
    ref_by_id = {r['ref_id']: r for r in ds.refs}
    E.build_map([ds.images[r['image_id']]['file_name'] for r in ds.refs])
    store, order = {}, []
    boxer = QwenBoxer.__new__(QwenBoxer)
    if box_path is None: box_path = boxer._path
    I = U = 0.0; ious = []; n = 0
    for r, fn, masks, fg in E.iterate_records(dataset, 'val', ref_by_id, ds, store, order, te, None):
        bp = box_path(fn, r['parsed']['sentence'].strip())
        box = json.loads(bp.read_text())['box'] if bp.is_file() else None
        our = fg @ te(r['parsed']['sentence'])
        feats = E.arm_c_features(masks, fg, r, box, our)
        X = E.feat_matrix(feats, E.ALL_COLS)
        pred = score_proposals(model, X, kind)
        j = int(np.argmax(pred))
        I += float(r['inter_gt'][j]); U += float(r['union_gt'][j]); ious.append(float(r['gt_iou'][j]))
        n += 1
        if n % 1000 == 0: print('val', n, 'ciou', round(100*I/U,3), flush=True)
    res = dict(n=n, mIoU=100.0*float(np.mean(ious)), cIoU=100.0*I/max(U,1e-9))
    print('VAL', dataset, res, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='refcoco', choices=list(E.RECORDS))
    ap.add_argument('--out', default='exp/out_final/ciou_head_refcoco.json')
    ap.add_argument('--val', action='store_true')
    args = ap.parse_args()
    te = E.TextEncoder(device='cuda:0')
    X, y, w_area, w_sqrt, hold_idx, recs = tune_matrices(args.dataset, te, None)
    # attach identifiers/inter/union for holdout scoring
    tune_rows = [json.loads(l) for l in open(ROOT / E.TUNE[args.dataset])]
    by = {(int(r['ref_id']), int(r.get('sent_idx', 0))): r for r in tune_rows}
    # rebuild per-record grouping from iterate order is unavailable; instead
    # reconstruct records from tune_rows and the feature rows in the same order
    # via a second pass over feature row lengths is not stored.  Re-run iterator
    # only to recover grouping metadata in the same order (cheap relative to fit).
    from glod.data import RefDataset
    ds = RefDataset(E.REPRO / 'data/refer', E.DS_NAME[args.dataset], 'train')
    ref_by_id = {r['ref_id']: r for r in ds.refs}
    store, order = {}, []
    idx = 0; recs = []
    from vlm_boxes import QwenBoxer
    boxer = QwenBoxer.__new__(QwenBoxer); box_path = boxer._path
    for r, fn, masks, fg in E.iterate_records(args.dataset, 'tune', ref_by_id, ds, store, order, te, None):
        m = len(fg)
        if idx + m > len(y): break
        recs.append(dict(ref_id=r['ref_id'], sent_idx=r.get('sent_idx', 0), hold=holdout_flag(r),
                         inter=np.asarray(r['inter_gt'][:m], float), union=np.asarray(r['union_gt'][:m], float),
                         y=y[idx:idx+m], X=X[idx:idx+m]))
        idx += m
    assert idx == len(y), (idx, len(y))
    print('recovered records', len(recs), flush=True)
    candidates = ['reg', 'reg_area', 'reg_sqrt', 'clf_area', 'clf_plain']
    best = None
    for kind in candidates:
        model = fit_candidate(kind, X[~hold_idx], y[~hold_idx], w_area[~hold_idx], w_sqrt[~hold_idx])
        c, m = holdout_metric(model, kind, recs)
        print('holdout', kind, 'cIoU', round(c,3), 'mIoU', round(m,3), flush=True)
        if best is None or c > best[1]: best = (kind, c, m)
    kind = best[0]
    print('SELECTED', best, flush=True)
    model = fit_candidate(kind, X, y, w_area, w_sqrt)
    out = dict(dataset=args.dataset, kind=kind, holdout_cIoU=best[1], holdout_mIoU=best[2],
               tune_rows=len(y), selected_on='train-carved ref-id hash holdout; official val untouched')
    if args.val:
        out['val'] = val_eval(args.dataset, te, model, kind, None)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
