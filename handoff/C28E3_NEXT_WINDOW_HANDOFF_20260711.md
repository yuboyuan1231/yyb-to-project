# C28E-3 Next Window Handoff - 2026-07-11

## 0. New Window Entry Point

Repository:

```bash
cd /home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3
```

Current branch:

```text
c28e-code-review-clip-late-interaction
```

Current HEAD:

```text
9caac01d119bae11242698b7daab9ba51a8dfefc
Archive C28E-3 full training results
```

Open pull request:

```text
https://github.com/yuboyuan1231/yyb-to-project/pull/9
base: c28c-cleanroom-e2e-blueprint-v2
head: c28e-code-review-clip-late-interaction
```

Suggested first instruction for a new Codex window:

```text
你现在接手 /home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3 项目。
先完整阅读 handoff/C28E3_NEXT_WINDOW_HANDOFF_20260711.md，严格按其中的
当前状态、禁止事项、证据边界和下一步顺序继续。C28E-3 已经完整结束，
不要重跑或覆盖旧产物。先做 teacher-free 检索取证和代码修复，完成自审、
测试并上传 GitHub 供我审核；未经审核不要启动下一轮正式长训练，绝对不要
运行 calib_holdout 或 official validation。
```

## 1. One-Screen Current State

Promoted official system remains:

```text
C7-B6 R1SelectiveTop1
```

C28E-3 state:

```text
run_status = C28E_FULL_E2E_TRAINING_COMPLETE
archive_conclusion = C28E-3_NO_PROMOTION_CONTINUE_RETRIEVAL_REPAIR
epochs_complete = 8/8 (0 through 7)
selection_split = calib_select
calib_holdout_run = false
official_val_used = false
evaluator_modified = false
nms_modified = false
```

There is no C28E training process to resume. The C28E tmux session has ended. At handoff time GPU0 was idle and GPU1 was occupied by another user's process; recheck GPU ownership before any future run.

The C28E-3 implementation is no longer a fake-name/retriever-only shell. The full model, late interaction, dynamic candidates, coupled losses, candidate-specific duration handling, masks, long-video grid, in-batch negatives, score audits, and crash recovery all ran at configured scale. The remaining failure is model behavior and curriculum/selection design, not missing mechanism names.

Do not continue by adding another patch to an old C17/C24/C26 runner. Keep work in the clean `blueprint_e2e_v2` ownership path, and use a new C28F-style experiment namespace for any materially changed run.

## 2. User Requirements That Must Survive Context Loss

These are hard constraints from the user:

1. R1@0.7 is the primary metric. Broader recall is useful but cannot hide a failed R1@0.7.
2. Do not reduce proposal count, final candidate count, broad candidate count, sequence length, or model size to save time or memory in a formal run.
3. Current formal scale is a floor unless a reviewed design explicitly changes it:
   - final candidate topK: 200
   - broad candidate topK: 1,000
   - max spans/video: 64
   - target length: 64
   - hidden dimension: 384
   - effective batch: 32
4. Memory optimization may use GPU microbatches, gradient accumulation, chunked scoring, multiple GPUs, SSD staging, or cached/preencoded banks, but must preserve formal semantics and accuracy.
5. Prefer GPU. Vectorize Python loops when practical. Use CPU multiprocessing only for genuinely CPU-bound work.
6. Long runtime is acceptable. Do not silently simplify the design because a run is slow.
7. Training must be independent of the Codex connection, normally via detached tmux plus append-only logs and step recovery.
8. Server crashes are expected. Every formal run needs deterministic step recovery and durable candidate/checkpoint records.
9. Explain every encountered error and repair in the final report. Do not hide restarts or route changes.
10. Preserve user files and the dirty worktree. Do not reset, clean, or add unrelated historical artifacts.
11. Upload reviewed code/results to the branch/PR, but do not upload large checkpoints, feature caches, or candidate matrices.
12. Do not spend protected holdout or official validation unless all selection gates pass and the user gives explicit authorization.

## 3. Read These Files in Order

Start with the current evidence, then consult older design material only as needed.

### Current authoritative evidence

```text
handoff/PROJECT_EXPERIMENT_ARCHIVE_C1_TO_C28E3_20260711.md
blueprint_e2e_v2/reports/c28e_3_training/README.md
blueprint_e2e_v2/reports/c28e_3_training/C28E_3_FULL_E2E_TRAINING_DECISION.json
blueprint_e2e_v2/reports/c28e_3_training/C28C_FULL_medium_seed2026.training_log.json
blueprint_e2e_v2/reports/c28e_3_training/C28E3_formal_cuda0_20260709_stream.log
```

The user's updated archive source is also available at:

```text
/home/a/.codex/attachments/39069d9f-e176-4aea-9bce-81f890eaeeb3/pasted-text.txt
```

### Blueprint and external review material

```text
/home/a/yybwork/yybcounter/plan/改进版.txt
/home/a/yybwork/yybcounter/plan/路线微调.txt
/home/a/yybwork/yybcounter/plan/CONQUER-RLEM_方案蓝图_v1.2_权威整合版.docx
/home/a/yybwork/yybcounter/advices/43.txt
/home/a/yybwork/yybcounter/advices/advice1.txt
/home/a/yybwork/yybcounter/advices/advice2.txt
/home/a/yybwork/yybcounter/paper/
/home/a/.codex/attachments/5f62d057-8141-4aca-a23b-b17a23947162/pasted-text.txt
```

### C28 code-review evidence

```text
blueprint_e2e_v2/reports/c28e_0_code_review/C28E_ISSUE_REPAIR_MATRIX.md
blueprint_e2e_v2/reports/c28e_0_code_review/C28E_CODE_REVIEW_CHECKLIST.md
blueprint_e2e_v2/configs/c28e_code_review.yaml
run_c28e_clip_late_interaction_retriever.py
blueprint_e2e_v2/engine/train.py
blueprint_e2e_v2/engine/evaluate.py
blueprint_e2e_v2/engine/refresh_hard_negatives.py
blueprint_e2e_v2/engine/c28e_late_interaction.py
blueprint_e2e_v2/models/full_model.py
blueprint_e2e_v2/losses/full_loss.py
```

## 4. What Was Implemented Before the Formal Run

The important branch commits are:

| commit | purpose |
|---|---|
| `9f7ed1d` | initial C28E clip late-interaction code-review branch |
| `68a94b8` | tightened blueprint mechanism integration |
| `2b940a2` | repaired full E2E blueprint wiring |
| `1f1c0b0` | closed major self-audit mechanism gaps |
| `3d18824` | added multi-metric gates and score-scale audits |
| `ff383ea` | skipped incompatible resume checkpoints safely |
| `d9378dd` | preencoded clip bank once per candidate refresh |
| `205ab3e` | added GPU microbatches without reducing formal batch/candidates |
| `b04ba4d` | stabilized streaming training memory |
| `434315a` | avoided raw sequence cache duplication during bank build |
| `7b0833b` | added deterministic step recovery checkpoints |
| `963b95d` | recovered an unlogged epoch-finalization/evaluation state |
| `9caac01` | archived final C28E-3 results and evidence |

Major blueprint gaps already repaired:

- Full C28E-3 owns `run_full_training` and `compute_full_loss`; retriever-only training is diagnostic only.
- Candidate eval topK is separate from hard-negative sample count.
- Candidate proposals use each candidate video's own duration.
- Real clip-level visual/subtitle/joint late interaction is inside the full graph.
- Localizer and feedback consume late retrieval evidence, not a stale pooled side channel.
- Clip masks protect all padded positions.
- Long videos are duration-aware mean-bin-resampled over the full timeline.
- Training scores raw candidate clips through the trainable video encoder.
- Candidate scoring and train/eval are chunked/microbatched without reducing candidate scale.
- ActiveMoment is candidate-video-conditioned.
- In-batch retrieval negatives are duplicate-video safe.
- Checkpoint selection uses `calib_select` rather than train metrics.
- C28E-4 uses multiple readiness gates.
- Score-scale and loss-coupling audits are emitted.
- Explicit `cuda:N` disables accidental DataParallel use of another GPU.
- Dynamic refresh preencodes the student clip bank once per refresh.

Do not reopen these questions without concrete code evidence. The new work starts from the measured C28E-3 failure, not from the old claim that the task was never implemented.

## 5. Exact C28E-3 Run Configuration

Formal command used after recovery support was active:

```bash
env PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0 \
  C28C_TMP_ROOT=/home/a/yybwork/yybcounter/c28c_score_cache/CONQUER-RLEM-c2c3 \
  /home/a/miniconda3/envs/conquer-rlem/bin/python \
  run_c28e_clip_late_interaction_retriever.py \
  --stage c28e_3 \
  --config blueprint_e2e_v2/configs/c28e_code_review.yaml \
  --mode medium \
  --seed 2026 \
  --resume \
  --device cuda:0
```

Important: this command is historical evidence. Do not rerun it for the next design. A new curriculum/model experiment must use a new output root and artifact identifier, start from a deliberate initialization, and must not overwrite or accidentally resume the C28E-3 checkpoint.

Formal scale:

| item | value |
|---|---:|
| train queries | 69,428 |
| selection queries | 8,677 |
| videos | 17,435 |
| epochs | 8 |
| final candidates | 200 |
| broad candidates | 1,000 |
| target length | 64 |
| max spans/video | 64 |
| hidden dimension | 384 |
| effective batch | 32 |
| CPU microbatch | 2 |
| GPU microbatch | 2 |
| checkpoint interval | 100 steps |
| device | physical GPU0 |

No proposal/candidate/model reduction was used.

## 6. Final Measured Results

All metrics below are percentages on `calib_select`, not `calib_holdout` and not official validation.

### Per-epoch trajectory

| epoch | loss | select score | VCMR R@1@0.7 | R@5@0.7 | R@10@0.7 | R@100@0.7 | VR R@100 | wrong-video top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 44.7158 | -47.1361 | 0.0115 | 0.0230 | 0.0346 | 0.1383 | 2.6507 | 99.9424 |
| 1 | 45.1482 | -45.8165 | 0.0000 | 0.0115 | 0.0346 | 0.2881 | 4.0682 | 99.9770 |
| 2 | 44.7955 | -46.3294 | 0.0000 | 0.0230 | 0.0461 | 0.2305 | 3.4920 | 99.9654 |
| 3 | 43.4786 | -45.0271 | 0.0000 | 0.0000 | 0.0461 | 0.4725 | 4.8750 | 99.9885 |
| 4 | 43.3751 | -43.5865 | 0.0000 | 0.0461 | 0.0807 | 0.5301 | 6.0620 | 99.8963 |
| 5 | 42.4838 | -38.9939 | 0.0115 | 0.0691 | 0.1729 | 1.1870 | 10.2801 | 99.7465 |
| 6 | 41.4286 | -37.5014 | **0.0461** | 0.1959 | 0.3918 | 1.9362 | 10.7756 | 99.6658 |
| 7 | **39.7513** | **-31.8428** | 0.0230 | **0.3457** | **0.5532** | **2.9273** | **15.6390** | **99.4353** |

### Checkpoint interpretation

- Automatic composite best: epoch 7, selection score `-31.8428`.
- Primary R1@0.7 measured peak: epoch 6, `0.0461`.
- Epoch 7 R1@0.7: `0.0230`.
- Epoch 7 latest and automatic-best checkpoint files are identical.
- Epoch 6 weights were overwritten by the rolling policy and are not separately available.

### Epoch 7 automatic-best metrics

| metric | value |
|---|---:|
| VCMR R@1 IoU0.5 | 0.1613 |
| VCMR R@5 IoU0.5 | 0.8183 |
| VCMR R@10 IoU0.5 | 1.1986 |
| VCMR R@100 IoU0.5 | 5.5664 |
| VCMR R@1 IoU0.7 | 0.0230 |
| VCMR R@5 IoU0.7 | 0.3457 |
| VCMR R@10 IoU0.7 | 0.5532 |
| VCMR R@100 IoU0.7 | 2.9273 |
| VR R@1 | 0.5647 |
| VR R@5 | 2.3395 |
| VR R@10 | 3.9530 |
| VR R@100 | 15.6390 |
| wrong-video top1 | 99.4353 |
| high-score false-positive | 99.4353 |
| top1 mean IoU | 0.00224 |

### Gate result

Every declared gate failed:

| gate | required | observed |
|---|---:|---:|
| VCMR R@1@0.7 | >= 3.0 | 0.0230 |
| VCMR R@5@0.7 | >= 8.0 | 0.3457 |
| VCMR R@10@0.7 | >= 12.0 | 0.5532 |
| VR R@100 | >= 80.0 | 15.6390 |
| wrong-video top1 | <= 95.0 | 99.4353 |
| high-score false-positive | <= 95.0 | 99.4353 |

Do not promote, run holdout, or run official validation.

## 7. Root-Cause Analysis That Should Drive the Next Work

### 7.1 Front-rank video retrieval is the dominant failure

At epoch 7:

```text
wrong video at top1                 = 99.4353
correct video but wrong span        = 0.4034
correct video with IoU >= 0.5       = 0.1613
```

Nearly all top1 failures happen before localization can matter. Do not start by redesigning span proposals again. First determine why the student cannot rank the correct video.

### 7.2 Training and evaluation candidate distributions are mismatched

Epochs 6 and 7 training audits show:

```text
teacher_warm_start_query_rate = 100.0
teacher_warm_topk = 40
gt_insert_for_training_loss = true
```

Evaluation shows:

```text
teacher_warm_start_query_rate = 0.0
gt_insert_for_training_loss = false
```

The configured schedule is:

```text
epoch <= 2: teacher topK 120
epoch <= 5: teacher topK 80
epoch <= 8: teacher topK 40
else:       teacher topK 0
```

An eight-epoch run uses epochs 0-7, so the teacher-free phase is unreachable. The model was selected in a student-only evaluation regime it never experienced as its final training candidate regime. This is a concrete implementation/configuration mismatch, not speculation.

### 7.3 The model learned deeper recall but not reliable top1

From epoch 0 to 7:

```text
loss:                44.7158 -> 39.7513
VR R@100:             2.6507 -> 15.6390
VCMR R@100@0.7:       0.1383 -> 2.9273
VCMR R@1@0.7:         0.0115 -> 0.0230
```

Optimization is functioning. The problem is ranking concentration/calibration at the front, not total absence of learning.

### 7.4 Composite selection and the primary metric are misaligned

The composite objective preferred epoch 7 because R@5/R@10, VR R@100, and wrong-video rate improved. R1@0.7 peaked at epoch 6 and then halved. Even the epoch 6 peak is unusably low, but checkpoint policy must still preserve the primary metric independently.

### 7.5 Artifact finalization has two defects

1. Only latest and composite-best checkpoints are retained; there is no primary-R1-best or immutable per-epoch checkpoint.
2. Raw `training_log.json` remains top-level `status: running` after completion. The authoritative decision JSON and stream log correctly show completion, but future runs should write final status explicitly.

### 7.6 Mechanism scale is not the explanation

The run used broad 1,000, final 200, 64 spans/video, target length 64, hidden 384, late interaction, and all full losses. Do not attribute the negative result to a reduced proposal shortcut or missing late-interaction wiring.

## 8. Exact Next Phase: Evidence First, Then C28F Repair

Use a new clearly named phase/module/report namespace such as `c28f_teacher_free_retrieval_repair`. Do not mutate the historical C28E-3 result package.

### Phase 0: Freeze C28E-3 evidence

- Treat commit `9caac01` and the current result package as immutable evidence.
- Do not alter the C28E decision JSON or stream log.
- Do not overwrite the local C28E checkpoint/cache root.
- Create a new branch from `9caac01` for C28F work after confirming the branch name with the user if needed.

### Phase 1: Teacher-free retrieval forensics before long training

Add or run diagnostics on `calib_select` only. Report all of these separately:

1. Student pooled broad-top1000 GT-video recall and rank distribution.
2. Student late-reranked top200 GT-video recall and rank distribution.
3. GT promotion/demotion from broad rank to late rank.
4. Final VR R@1/R@5/R@10/R@100.
5. Teacher-vs-student candidate overlap by K.
6. Query-type and video-duration breakdowns.
7. Wrong-video top1 score margin versus the correct video.
8. Candidate recall before and after any GT insertion; never mix the two.

Decision routing:

- If broad-top1000 GT recall is low, repair pooled/student video retrieval first.
- If broad recall is high but late-top200 recall collapses, repair late reranking and score calibration.
- If late-top200 recall is high but VR R@1/R@100 is low, repair final score ordering/loss alignment.
- Only if video retrieval becomes adequate but VCMR remains weak should localization/proposal work become the primary focus.

### Phase 2: Required code repairs

Repair these before another full run:

1. Reachable teacher-free curriculum.
   - Make epoch boundaries explicit in config rather than hard-coded unreachable conditions.
   - For an eight-epoch run, a reasonable predeclared schedule is 120/120, 80/80, 40/40, 0/0 for epochs 0-7.
   - Assert before training that at least one final epoch is teacher-free.
   - Log actual teacher topK and teacher query rate every epoch.

2. Stage-wise retrieval audits.
   - Persist broad, late, and final retrieval metrics in train/eval reports.
   - Include teacher/student overlap and GT insertion rate.
   - Keep score-scale audits for pooled, late, token, and combined scores.

3. Metric-specific checkpoint retention.
   - Keep `latest`, `composite_best`, and `primary_r1_iou07_best` separately.
   - Prefer immutable per-epoch manifests or checkpoints if storage permits.
   - Never let a later composite winner erase the primary-metric peak.

4. Selection alignment.
   - Keep the composite score for broad stability.
   - Add a hard/lexicographic R1@0.7 condition so deeper recall cannot conceal front-rank regression.
   - Preserve and report both winners even when neither passes promotion gates.

5. Final metadata.
   - Write `status: complete` only after all epoch evaluations and final artifacts succeed.
   - Preserve a pending-finalization state that can be resumed after crashes.

Teacher candidate injection is not the same as teacher-to-student score learning. After the stage-wise audit, consider a full-graph rank-distillation loss that transfers teacher ordering into student scores while the candidate curriculum anneals to student-only. Do not add it blindly before locating whether broad retrieval or late reranking is the failing stage.

### Phase 3: Tests and code review gate

Before any formal training:

- Unit-test the exact eight-epoch teacher schedule and assert final teacher topK is zero.
- Unit-test broad/late recall accounting on synthetic candidate/rank examples.
- Unit-test duplicate GT videos and GT insertion accounting.
- Unit-test separate latest/composite/R1 checkpoint retention.
- Unit-test crash recovery during training and during pending epoch finalization.
- Run syntax and focused smoke tests.
- A smoke test may reduce query count, but must keep candidate topK, span count, target length, hidden size, loss graph, and score path unchanged.
- Update the issue matrix and checklist.
- Push code to GitHub and let the user review it.
- Do not start the next formal long run until the user accepts the code.

### Phase 4: Formal run only after approval

- Use a new output/cache root, experiment ID, report directory, checkpoint names, and append-only log.
- Start from the explicitly reviewed initialization; do not silently `--resume` C28E-3 into a changed curriculum.
- Recheck GPU ownership. Do not touch another user's GPU.
- Use detached tmux and 100-step recovery or stronger.
- Keep the full 200/1,000/64/64/384/32 scale.
- Record process ID, tmux name, physical GPU, config hash, commit hash, and artifact paths.
- If the server crashes, verify the recovery payload before restarting.

### Phase 5: Selection gate

Use only `calib_select` first. The existing minimum gates remain:

```text
VCMR R@1@0.7  >= 3.0
VCMR R@5@0.7  >= 8.0
VCMR R@10@0.7 >= 12.0
VR R@100       >= 80.0
wrong-video top1 <= 95.0
high-score false-positive <= 95.0
```

No `calib_holdout` or official validation unless all gates pass and the user explicitly authorizes the next stage.

## 9. Local Artifacts and Reproducibility

Result package committed to GitHub:

```text
blueprint_e2e_v2/reports/c28e_3_training/README.md
blueprint_e2e_v2/reports/c28e_3_training/C28E_3_FULL_E2E_TRAINING_DECISION.json
blueprint_e2e_v2/reports/c28e_3_training/C28C_FULL_medium_seed2026.training_log.json
blueprint_e2e_v2/reports/c28e_3_training/C28E3_formal_cuda0_20260709_stream.log
handoff/PROJECT_EXPERIMENT_ARCHIVE_C1_TO_C28E3_20260711.md
```

Local cache root, not committed:

```text
/home/a/yybwork/yybcounter/c28c_score_cache/CONQUER-RLEM-c2c3
```

Local checkpoints, not committed:

```text
/home/a/yybwork/yybcounter/c28c_score_cache/CONQUER-RLEM-c2c3/checkpoints/C28C_FULL_medium_seed2026.pt
/home/a/yybwork/yybcounter/c28c_score_cache/CONQUER-RLEM-c2c3/checkpoints/C28C_FULL_medium_seed2026.best.pt
```

Checkpoint facts:

```text
epoch = 7
size = 97,704,933 bytes
sha256 = fa0bafc6b5c1a0d049d462bcd0b77a073222715af0713ae291c05ea8631d13ab
latest and best are identical
```

## 10. Useful Read-Only Commands

Repository and PR:

```bash
git status --short --branch
git log -15 --oneline --decorate
gh pr view 9 --repo yuboyuan1231/yyb-to-project \
  --json url,state,baseRefName,headRefName,headRefOid,title
```

Final result summary:

```bash
jq '{status,stage,official_val_used,pseudo_official_holdout_used_for_selection,
     evaluator_modified,nms_modified,
     best: .full_training.best_select}' \
  blueprint_e2e_v2/reports/c28e_3_training/C28E_3_FULL_E2E_TRAINING_DECISION.json
```

Per-epoch key metrics:

```bash
jq -r '.training_log[] |
  [.epoch,.loss.L_total,.select_score,
   .select_summary."VCMR_R@1_IoU0.7",
   .select_summary."VCMR_R@100_IoU0.7",
   .select_summary."VR_R@100",
   .select_summary.wrong_video_top1_rate] | @tsv' \
  blueprint_e2e_v2/reports/c28e_3_training/C28C_FULL_medium_seed2026.training_log.json
```

Completion evidence:

```bash
rg -n 'training epoch 7 complete|eval calib_select: scored 8677/8677|C28E_FULL_E2E_TRAINING_COMPLETE' \
  blueprint_e2e_v2/reports/c28e_3_training/C28E3_formal_cuda0_20260709_stream.log
```

Process/GPU check before a future run:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,rss,cmd | \
  rg 'run_c28e|run_c28f|PID'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
tmux list-sessions
```

## 11. Known Warnings and Worktree Hazards

1. The repository contains many unrelated untracked historical directories/files. They belong to the user. Do not run `git clean`, do not reset them, and do not add them to a C28 commit.
2. The stream log contains a PyTorch warning about converting a non-writable NumPy view. The code immediately transfers it to the GPU and did not mutate the CPU view, so it did not invalidate C28E-3. Clean this warning in future code only if semantics and memory usage remain unchanged.
3. Root-level official authorization markers may exist from historical C7/C9 work. Their existence is not authorization for C28F. Never run official validation without current explicit user authorization and a branch-local guard.
4. `training_log.json status=running` is stale metadata; use the decision JSON and final log marker as authoritative evidence for C28E-3 completion.
5. Do not infer that epoch 7 is best for R1@0.7. It is only composite-best.
6. Do not compare C28E `calib_select` percentages directly with C7 official metrics as if they were the same evaluation regime.

## 12. Definition of Done for the Next Coding Window

The next coding window is complete only when:

1. Teacher-free stage diagnostics identify whether broad retrieval, late reranking, or final ordering is the first failing stage.
2. The eight-epoch curriculum reaches at least one predeclared teacher-free final epoch.
3. Broad/late/final retrieval and teacher/student overlap audits are persisted.
4. Latest, composite-best, and primary-R1-best checkpoints are separate.
5. Final training metadata writes a complete state and remains crash-resumable.
6. Focused tests and a mechanism-faithful smoke test pass without reducing formal candidate/proposal/model dimensions.
7. The issue matrix, checklist, and next-run config are updated.
8. Only relevant files are committed and pushed for user review.
9. No formal long run, holdout, or official validation has been started without the required user approval.

## 13. Bottom Line

C28E-3 answered the implementation question: the intended full E2E mechanism can run at full scale and recover from crashes. It did not answer the quality goal. The immediate blocker is student video retrieval under a teacher-free evaluation regime, compounded by an unreachable teacher-free training phase and checkpoint selection that is not primary-R1 aligned.

Continue with evidence-led C28F retrieval/curriculum repair. Do not rerun C28E-3, do not overwrite its artifacts, and do not spend protected evaluation data.
