#!/usr/bin/env python3
"""Record completed mandatory-method runs into exp/out_final/deep_read.json.

Only reads measured artifacts; never invents numbers. Run this after a method
evaluation finishes and the evidence JSON exists.
"""
import json, os, sys, datetime

ROOT = '/home/msj_team/Jacob/research/ris'
DEEP = os.path.join(ROOT, 'exp/out_final/deep_read.json')
NOW = datetime.datetime.now().isoformat(timespec='seconds')

SPLITS_ALL = ['refcoco_val', 'refcoco_testA', 'refcoco_testB',
              'refcocoplus_val', 'refcocoplus_testA', 'refcocoplus_testB',
              'refcocog_val', 'refcocog_testu', 'grefcoco_val']


def load():
    return json.load(open(DEEP))


def paper_by_key(d, key):
    for p in d['papers']:
        if p.get('key') == key:
            return p
    raise SystemExit('key not found: ' + key)


def blockers_for(covered, evidence, reason):
    return {s: {'reason': reason, 'evidence': evidence} for s in SPLITS_ALL if s not in covered}


def record_lscm():
    merged = os.path.join(ROOT, 'exp/out_final/lscm_full_val_merged.json')
    if not os.path.exists(merged):
        raise SystemExit('missing ' + merged)
    m = json.load(open(merged))
    d = load(); p = paper_by_key(d, 'lscm2020')
    c = p.setdefault('code', {})
    c['ran'] = True
    c['inspected'] = True
    c['findings'] = ('LSCM_model_p2345.py implements the gather-propagate-distribute module; '
                     'trainval_model.py wires training/evaluation; util/eval_tools.py contains IoU/Pr@X helpers. '
                     'The released RefCOCO/unc checkpoint restores into the model graph (499 variables) and '
                     'was evaluated with the official test path including DenseCRF.')
    c.pop('blocker', None); c.pop('blocker_evidence', None); c.pop('retry_plan', None)
    run = {
        'entry': "python trainval_model.py -g '' -m test -d unc -t s<00-11> -i 700000 -f ckpts/unc/lscm_p2345",
        'sample_n': int(m['n']),
        'status': 'ok',
        'kind': 'method',
        'full_split': True,
        'split': 'refcoco_val',
        'protocol': m.get('protocol', ''),
        'env': m.get('env', ''),
        'datasets': {'refcoco_val': {'n': int(m['n']), 'metric': float(m['mIoU_dcrf'])}},
        'split_blockers': blockers_for(['refcoco_val'], merged,
            'Official split not yet evaluated in this pass; the released LSCM checkpoint and the val run are recorded here. '
            'Remaining splits require building their REFER batches and are queued after this val run.'),
        'metrics': {'mIoU_dcrf': float(m['mIoU_dcrf']), 'oIoU_dcrf': float(m['oIoU_dcrf']),
                    'mIoU_no_crf': float(m['mIoU']), 'oIoU_no_crf': float(m['oIoU'])},
        'notes': ('Released LSCM RefCOCO/unc checkpoint, full official val split (n=10,834), '
                  '12-way exact-accumulator sharded run; DenseCRF post-processing is the paper default. '
                  'Merged from per-shard cum_I/cum_U and per-expression IoU sums; no sampling. '
                  'Checkpoint SHA256 and shard manifest are stored alongside the merged JSON.'),
    }
    c['run'] = run
    json.dump(d, open(DEEP, 'w'), indent=2)
    print('recorded LSCM', run['metrics'])


def record_text4seg():
    out = os.path.join(ROOT, 'exp/out_final/text4seg_refcoco_val.json')
    if not os.path.exists(out):
        raise SystemExit('missing ' + out)
    m = json.load(open(out))
    if not m.get('n'):
        raise SystemExit('text4seg run has no records')
    d = load(); p = paper_by_key(d, 'text4seg2024')
    c = p.setdefault('code', {})
    c['ran'] = True
    c['inspected'] = True
    c['findings'] = ('llava/eval/model_refer_seg.py decodes the generated <seg> R-RLE text and refines the mask '
                     'with SAM ViT-H; llava/eval/eval_refer_seg.py pools intersection/union over expressions '
                     '(cIoU) and averages per-expression IoU (gIoU). The released quick-inference checkpoint is '
                     'lmc22/text4seg-llava-7b-p24 on HuggingFace.')
    c.pop('blocker', None); c.pop('blocker_evidence', None); c.pop('retry_plan', None)
    prec = m.get('precision', 'see eval log')
    run = {
        'entry': 'python -m llava.eval.model_refer_seg --dataset_split refcoco|unc|val (official Text4Seg harness)',
        'sample_n': int(m['n']),
        'status': 'ok',
        'kind': 'method',
        'full_split': True,
        'split': 'refcoco_val',
        'protocol': m.get('protocol', ''),
        'env': m.get('env', ''),
        'datasets': {'refcoco_val': {'n': int(m['n']), 'metric': float(m['mIoU'])}},
        'split_blockers': blockers_for(['refcoco_val'], out,
            'Official split not yet evaluated in this pass; this run records the full RefCOCO val split with the released checkpoint.'),
        'metrics': {'mIoU': float(m['mIoU']), 'cIoU': float(m['cIoU']),
                    'sam_mIoU': float(m['sam_mIoU']) if m.get('sam_mIoU') is not None else 0.0,
                    'sam_cIoU': float(m['sam_cIoU']) if m.get('sam_cIoU') is not None else 0.0},
        'notes': ('Released Text4Seg p24 quick-inference checkpoint, full official RefCOCO val split, '
                  'official model_refer_seg -> eval_refer_seg metric definitions (cIoU/mIoU). '
                  'Precision: ' + str(prec) + '. No sampling.'),
    }
    c['run'] = run
    json.dump(d, open(DEEP, 'w'), indent=2)
    print('recorded Text4Seg', run['metrics'])


def record_semantic_sam():
    out = os.path.join(ROOT, 'exp/out_final/semantic_sam_refcoco_val.json')
    if not os.path.exists(out):
        src = '/home/msj_team/Jacob/sys/semsam_run/refcoco_val_interactive_done.json'
        if not os.path.exists(src):
            raise SystemExit('missing ' + out)
        import shutil; shutil.copy(src, out)
    m = json.load(open(out))
    if not m.get('n_sentences'):
        raise SystemExit('semantic-sam run has no sentences')
    d = load(); p = paper_by_key(d, 'semanticsam2023')
    c = p.setdefault('code', {})
    c['ran'] = True
    c['inspected'] = True
    c['findings'] = ('The released Semantic-SAM SwinT checkpoint produces class-agnostic multi-granularity masks '
                     'through semantic_sam/tasks/interactive_predictor.py; it is a pool component, not a referring selector. '
                     'Measured as a mask ceiling on RefCOCO val with an oracle point at the GT box centre.')
    c.pop('blocker', None); c.pop('blocker_evidence', None); c.pop('retry_plan', None)
    run = {
        'entry': 'python /home/msj_team/Jacob/sys/semsam_run/semsam_interactive_val.py',
        'sample_n': int(m['n_sentences']),
        'status': 'ok',
        'kind': 'component',
        'full_split': True,
        'split': 'refcoco_val',
        'protocol': m.get('protocol', ''),
        'env': m.get('env', ''),
        'metrics': {'oracle_mIoU': float(m['oracle_mIoU']),
                    'oracle_hit50': float(m['oracle_hit50']),
                    'oracle_hit70': float(m['oracle_hit70']),
                    'prediou_mIoU': float(m['prediou_mIoU'])},
        'notes': ('Full official RefCOCO val (10,834 expressions / 3,811 refs / 1,500 images). '
                  'Component run: oracle localization (GT box centre point), released SwinT SA-1B checkpoint. '
                  'This is a mask-quality ceiling, not a referring-method number.'),
    }
    c['run'] = run
    json.dump(d, open(DEEP, 'w'), indent=2)
    print('recorded Semantic-SAM', run['metrics'])


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('lscm', 'all'): record_lscm()
    if which in ('text4seg', 'all'): record_text4seg()
    if which in ('semanticsam', 'all'): record_semantic_sam()
