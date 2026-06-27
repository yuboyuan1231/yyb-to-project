# CONQUER-RLEM C0 Baseline Audit

Date: 2026-06-21

Status: `C0_BASELINE_REPRODUCED`

## Scope

C0 only: establish the isolated project workspace, bind the downloaded TVR
features, import the official CONQUER checkpoint, and reproduce the frozen
CONQUER validation baseline. No C1 evidence export or RLEM training was run.

## Frozen paths

- Project root: `/home/a/yybcounter/CONQUER-RLEM-c2c3`
- Source archive: `/home/a/yybcounter/plan/CONQUER-RLEM-c2c3.zip`
- TVR feature root (read-only): `/home/a/data/yyb/tvr_feature_release`
- Historical official result: `results/tvr-conquer_general_paper_performance`
- C0 reproduction result: `results/tvr-conquer_c0_repro_20260621`
- C0 log: `c0_audit/logs/baseline_inference_nms_0.7.log`

## Integrity

| Artifact | SHA256 |
|---|---|
| `CONQUER-RLEM-c2c3.zip` | `7cd88654f8dd9f57519c1dd43a930fb1068bdc8adf4a46302652505836e3a24e` |
| official `model.ckpt` | `bf7aa90b6b2db212233c37d27b81a1e9b254b2a7be6bee7cbf75bb269cdf0264` |
| reproduced NMS prediction JSON | `74cacd0b240124eaf3598462ef9fefe7931fafca0e23cb5875a398e2584ea750` |
| reproduced NMS metrics JSON | `4bd8704888627252d2099d63cac73d227da8eba6b229b5931d7b043edd6050eb` |
| complete inference log | `2c2f2312b9bcec9cd2a4c53f41a1e035c599dc4ad42402dce65e8372b8c45fa5` |

The imported checkpoint hash exactly matches the checkpoint stored in the
official `tvr-conquer_general_paper_performance.tar.gz` archive. It loads as
epoch 9 with 122 model state entries.

## Frozen evaluation protocol

```text
split             = val
queries           = 10,895
max_vcmr_video    = 10
candidate rows    = 200
effective NMS input = score top-100
max_after_nms     = 100
nms_thd           = 0.7
clip_length       = 1.5
tasks             = VR SVMR VCMR
```

Command:

```bash
cd /home/a/yybcounter/CONQUER-RLEM-c2c3
bash scripts/inference.sh tvr-conquer_c0_repro_20260621 0 --nms_thd 0.7
```

Runtime: 144 seconds on GPU 0.

## VCMR reproduction gate

| Metric | Historical | Reproduced | Delta |
|---|---:|---:|---:|
| R@1 IoU 0.5 | 13.68 | 13.68 | 0.00 |
| R@5 IoU 0.5 | 27.21 | 27.21 | 0.00 |
| R@10 IoU 0.5 | 33.75 | 33.75 | 0.00 |
| R@100 IoU 0.5 | 46.23 | 46.23 | 0.00 |
| R@1 IoU 0.7 | 7.76 | 7.76 | 0.00 |
| R@5 IoU 0.7 | 17.22 | 17.22 | 0.00 |
| R@10 IoU 0.7 | 22.49 | 22.49 | 0.00 |
| R@100 IoU 0.7 | 35.17 | 35.17 | 0.00 |

All eight headline VCMR metrics exactly match the historical official result.
All VCMR/SVMR/VR outputs contain 10,895 queries and the submission contains a
2,179-video index.

Audit note: although the CLI value is `max_before_nms=200`, the original
`get_submission_top_n` mutates the raw result to top-100 before the NMS branch.
Consequently the historical official metrics use 200 generated rows, then the
score top-100 as effective NMS input. C1 preserves all 200 evidence rows, while
base reconstruction must reproduce the effective top-100 NMS input.

One secondary SVMR value differs by 0.01 (`R@5 IoU 0.7`: 47.72 historical,
47.71 reproduced); every other SVMR and VR metric matches. This does not fail
the VCMR C0 gate and is retained as an environment-level numerical note.

## Compatibility-only edits

- `config/tvr_data_config.json`: changed only `root_path`; the archive version
  is preserved as `config/tvr_data_config.json.bak`.
- `inference.py`: supplied NumPy `int`/`bool` aliases removed in NumPy 1.24.
- `config/config.py`: escaped literal percent signs so `python inference.py -h`
  works with current Python argparse.
- Reproduction `opt.json`: changed only `results_dir` to isolate new outputs.

These edits do not alter model weights, QDF, QAL, ML, VS, candidate generation,
NMS, split, or evaluator behavior.

## Gate decision

`C0_PASS`. The official CONQUER checkpoint is usable and baseline training is
not needed. Stop here. C1 is not authorized by this report.
