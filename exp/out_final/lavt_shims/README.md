# LAVT reproduction shims (2026-10-22)

The released LAVT `test.py` imports `mmcv` and `mmseg` only for checkpoint-I/O
helpers and a logger. The environment has neither, so `mmcv/` and `mmseg/` here
are minimal import shims; put them on the repository root (or PYTHONPATH) when
running the released code. The released LAVT checkpoint is unchanged.

BERT: convert the HF `bert-base-uncased` safetensors snapshot to
`bert_base_local/pytorch_model.bin` with `safetensors.torch.load_file` +
`torch.save`; `config.json` and `vocab.txt` are copied here for reference.

Run (from the LAVT repo root):

```
CUDA_VISIBLE_DEVICES=<gpu> python test.py --model lavt --dataset refcoco \
  --split val --resume checkpoints/lavt_refcoco.pth --ck_bert bert_base_local \
  --refer_data_root /home/msj_team/Jacob/sys/glod_ris_repro/data/refer \
  --device cuda:0 --workers 2 --batch-size 1 --window12 --ddp_trained_weights
```

Result: mIoU 74.46 / oIoU 72.73 on all 10,834 RefCOCO val expressions,
exactly the published LAVT row. Evidence: `exp/out_final/lavt_refcoco_full.json`.
