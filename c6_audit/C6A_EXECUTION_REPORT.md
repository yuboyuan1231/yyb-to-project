# C6-A 执行报告：Adapter-only Trainable Mutualization

生成时间：2026-06-25  
阶段范围：Phase 0–7，仅 train_fit / train_calib  
最终状态：`C6A_NEGATIVE`

## 1. 结论先行

C6-A 已完成当前授权范围内的实现、cache 构建、smoke test、三组 adapter 训练与 train_calib freeze gate 审查。结论是：

```text
C6-A adapter-only trainable mutualization 当前不通过 freeze gate。
不得进入 official-val one-shot。
不得进入 C6-B / C6-C / candidate regeneration。
```

原因不是工程链路失败。相反，工程链路本身通过了：

- train_fit / train_calib cache 均 `PASS`；
- zero-adapter 可以严格复现 C4_final，`max_abs_diff = 0.0`；
- overfit smoke loss 明显下降；
- GPU forward/backward 跑通；
- movement safety 基本安全；
- official val 未读取、未用于选择。

真正失败点在算法结果：三组 C6-A adapter 在 train_calib 上都没有把 retrieval→localization prior 稳定转成 VCMR front-rank 增益。最佳配置 `c6a_regularized` 仍然是 negative：

```text
selection_score = -0.0631514
front_positive_count = 1
R@1 gate = false
localization_positive_count = 0
core_gate = false
```

## 2. 阶段边界与冻结约束

本阶段严格遵守以下边界：

```text
official_val_used = false
post_val_adjustment = false
backbone_finetuned = false
candidate_generation_modified = false
NMS_modified = false
evaluator_modified = false
C6_B_used = false
C6_C_used = false
```

C6-A 不是继续修 C5 fixed-candidate scorer，也不是 C6 full integration。它是一个新协议：

```text
train_fit 训练 adapter/head
train_calib 选择 epoch/config
冻结 CONQUER backbone
冻结 C4_final 作为 primary baseline
不读 official val
```

Primary baseline：

```text
C4_final = C4-r2-cal-v2.1 v21_00444
```

## 3. 新增代码与实现修正

新增 C6-A 代码位于：

```text
rlem_c6/
```

核心文件：

```text
rlem_c6/c6a_utils.py
rlem_c6/build_c6_cache.py
rlem_c6/validate_c6_cache.py
rlem_c6/train_c6a.py
rlem_c6/freeze_c6a.py
rlem_c6/write_start_state.py
rlem_c6/configs/c6a_adapter_default.yaml
rlem_c6/configs/c6a_adapter_large.yaml
rlem_c6/configs/c6a_adapter_regularized.yaml
```

执行中发现并修复了两个工程问题：

1. 直接运行 `rlem_c6/*.py` 时 repo root 未进入 `sys.path`，已修正；
2. 压缩 NPZ 随机访问会导致训练 hot path 反复解压，速度不可接受。已改为 Dataset 初始化时预载必要 arrays，训练 hot path 只做连续 numpy slice + GPU transfer。

这两个修复不改变 scoring/eval 口径，只解决可执行性与效率问题。

## 4. Cache 构建结果

### 4.1 train_fit cache

文件：

```text
results/rlem_c6a/cache/train_fit_c6_cache.npz
c6_audit/C6A_TRAIN_FIT_CACHE_MANIFEST.json
```

结果：

```text
status = PASS
queries = 78045
rows = 15609000
groups = 779142
bad_alignment_count = 0
missing_filtered_queries = 0
nonfinite_values = 0
temporal_bad_values = 0
official_val_used = false
elapsed_sec = 330.699
```

### 4.2 train_calib cache

文件：

```text
results/rlem_c6a/cache/train_calib_c6_cache.npz
c6_audit/C6A_TRAIN_CALIB_CACHE_MANIFEST.json
```

结果：

```text
status = PASS
queries = 8739
rows = 1747800
groups = 87230
bad_alignment_count = 0
missing_filtered_queries = 0
nonfinite_values = 0
temporal_bad_values = 0
official_val_used = false
elapsed_sec = 246.018
```

### 4.3 cache validation

文件：

```text
c6_audit/C6A_CACHE_AUDIT.md
c6_audit/C6A_CACHE_AUDIT.json
```

关键检查：

```text
train_fit p_b/p_e/p_ctx finite = true
train_fit p_b/p_e/p_ctx nonnegative = true
train_calib p_b/p_e/p_ctx finite = true
train_calib p_b/p_e/p_ctx nonnegative = true
max distribution sum error <= 2.98e-7
zero_adapter_max_abs_diff = 0.0
gpu_forward_device = cuda
official_val_used = false
```

`zero_adapter_max_abs_diff = 0.0` 很关键：它说明 C6-A 的数据管线与 scoring 框架在不启用 adapter 时严格复现 C4_final，没有引入隐性排序漂移。

## 5. Smoke tests

### 5.1 zero-adapter smoke

结果：

```text
PASS
max_abs_diff = 0.0
```

说明：C6-A score 在 adapter disabled 时等于 frozen C4_final。

### 5.2 overfit smoke

文件：

```text
c6_audit/C6A_OVERFIT_SMOKE.md
c6_audit/C6A_OVERFIT_SMOKE.json
```

100 query-video groups 上训练 4 个 epoch，loss：

```text
epoch 1: 2.154649
epoch 2: 2.194281
epoch 3: 1.414182
epoch 4: 1.483290
```

结果：

```text
status = PASS
```

说明：模型不是完全学不动，优化链路可用。

## 6. 模型配置

共训练三组非线性 adapter/head 配置，均不属于“退化线性 scorer”：

### 6.1 c6a_default

```text
hidden_dim = 256
temporal_layers = 4
heads = 8
ffn_dim = 512
span_head_hidden = 384
span_head_layers = 3
dropout = 0.10
epochs = 8
```

### 6.2 c6a_large

```text
hidden_dim = 384
temporal_layers = 4
heads = 8
ffn_dim = 768
span_head_hidden = 512
span_head_layers = 3
dropout = 0.10
epochs = 8
```

### 6.3 c6a_regularized

```text
hidden_dim = 256
temporal_layers = 3
heads = 8
ffn_dim = 512
span_head_hidden = 384
span_head_layers = 3
dropout = 0.15
epochs = 8
```

## 7. Train_calib 结果

### 7.1 总览

| config | best epoch | selection score | freeze gate |
|---|---:|---:|---|
| c6a_default | 1 | -0.091783 | FAIL |
| c6a_large | 1 | -0.114581 | FAIL |
| c6a_regularized | 1 | -0.063151 | FAIL |

最佳配置按 selection score 为：

```text
selected_config = c6a_regularized
selected_epoch = 1
```

但它仍未通过 C6-A freeze gate。

### 7.2 c6a_default vs C4_final

八项 delta：

```text
0.5-r1   +0.011443
0.5-r5   +0.000000
0.5-r10  -0.034329
0.5-r100 +0.022886
0.7-r1   -0.034329
0.7-r5   -0.022886
0.7-r10  -0.045772
0.7-r100 +0.034329
```

Movement：

```text
top1_changed_ratio = 0.005607
hard_positive_top100_exits = 30
hard_positive_top100_entries = 55
hard_positive_top100_exit_ratio = 0.000478
Pearson(C4_final, C6-A) = 0.999980
mean within-query Spearman = 0.999864
```

Localization diagnostics：

```text
oracle_video_r1_05_delta = -0.056794
oracle_video_r1_07_delta = -0.042595
selected_span_miou_delta = -0.000381
best_iou_span_rank_delta_mean = +0.003124
```

### 7.3 c6a_large vs C4_final

八项 delta：

```text
0.5-r1   +0.000000
0.5-r5   +0.011443
0.5-r10  -0.045772
0.5-r100 +0.011443
0.7-r1   +0.000000
0.7-r5   -0.057215
0.7-r10  -0.091544
0.7-r100 +0.000000
```

Movement：

```text
top1_changed_ratio = 0.005149
hard_positive_top100_exits = 19
hard_positive_top100_entries = 34
hard_positive_top100_exit_ratio = 0.000303
Pearson(C4_final, C6-A) = 0.999991
mean within-query Spearman = 0.999916
```

Localization diagnostics：

```text
oracle_video_r1_05_delta = -0.028397
oracle_video_r1_07_delta = +0.000000
selected_span_miou_delta = -0.000416
best_iou_span_rank_delta_mean = +0.004260
```

### 7.4 c6a_regularized vs C4_final

八项 delta：

```text
0.5-r1   -0.011443
0.5-r5   +0.045772
0.5-r10  +0.000000
0.5-r100 +0.045772
0.7-r1   -0.011443
0.7-r5   -0.045772
0.7-r10  -0.045772
0.7-r100 +0.011443
```

Movement：

```text
top1_changed_ratio = 0.003776
hard_positive_top100_exits = 27
hard_positive_top100_entries = 32
hard_positive_top100_exit_ratio = 0.000430
Pearson(C4_final, C6-A) = 0.999986
mean within-query Spearman = 0.999897
```

Localization diagnostics：

```text
oracle_video_r1_05_delta = -0.014198
oracle_video_r1_07_delta = +0.000000
selected_span_miou_delta = -0.000102
best_iou_span_rank_delta_mean = +0.004118
```

## 8. Freeze gate 审查

最佳配置 `c6a_regularized` 的 gate：

```text
core_gate = false
strong_gate = false
front_positive_count = 1
r1_gate = false
r100_gate = true
movement_gate = true
localization_positive_count = 0
```

按照 C6-A gate：

```text
至少 4/6 front-rank metrics positive
两个 R@1 至少一个 positive，另一个 nonnegative
R@100 deltas >= -0.05
hard-positive exit ratio <= 0.5%
至少 2 个 localization diagnostics positive
Pearson >= 0.98
Spearman >= 0.95
```

当前只满足 movement / R@100 safety，不满足 front-rank 与 localization 条件。

因此冻结审查结论为：

```text
C6A_NEGATIVE
```

## 9. 解释：为什么 C6-A negative

这次 negative 不是因为 C6-A 扰动过强破坏 retrieval 排序。相反，movement 很小，Pearson/Spearman 都非常接近 1，hard-positive top100 exit ratio 也远低于阈值。

问题在于：

```text
adapter 的有效扰动非常安全，但没有把 temporal prior 转化为稳定 localization / front-rank 增益。
```

具体表现：

- R@100 safety 多数为正或非负；
- top1_changed_ratio 很低；
- 但 R@1 没有稳定提升；
- 0.7 front-rank 指标普遍下降；
- localization diagnostics 没有正向；
- best_iou_span_rank_delta_mean 为正，表示 best-IoU span 平均排名反而略变差。

这说明在当前 fixed candidates + frozen backbone + adapter-only 框架下，C6-A 仍没有突破 C5 fixed-candidate 后处理暴露出的核心问题：

```text
retrieval→localization signal 存在，但在 fixed candidate reranking 上很难稳定转成 VCMR front-rank 收益。
```

## 10. 与前阶段关系

当前阶段不推翻 C4_final。

```text
C4_final = C4-r2-cal-v2.1 v21_00444
```

C4_final 仍是当前主结果与 primary baseline。

C5 fixed-candidate 系列已经 official-val negative / not promoted；C6-A 进一步验证了即使把 C5 prior 放进 trainable adapter，在 fixed candidates + frozen backbone 的约束下，也未能在 train_calib 上达到 promotion gate。

## 11. 推荐决策

建议：

```text
1. 不授权 C6-A official-val one-shot；
2. 不继续 C6-A 当前配置族的 epoch/config 微调；
3. 不回头继续 C5 fixed-candidate scorer；
4. 保留 C4_final 作为当前稳定主系统；
5. 若后续继续，应另开新协议，而不是在当前 C6-A 上补小 grid。
```

如果要继续探索，真正有意义的方向可能是：

```text
C6-B candidate regeneration / candidate proposal protocol
或更深的 backbone-integrated retrieval-localization mutualization
```

但这已经超出当前授权，本阶段没有启动。

## 12. 审计文件索引

Start / protocol：

```text
c6_audit/C6A_START_STATE.md
c6_audit/C6A_START_STATE.json
c6_audit/C6A_PROTOCOL.md
```

Cache：

```text
c6_audit/C6A_TRAIN_FIT_CACHE_MANIFEST.json
c6_audit/C6A_TRAIN_CALIB_CACHE_MANIFEST.json
c6_audit/C6A_CACHE_AUDIT.md
c6_audit/C6A_CACHE_AUDIT.json
```

Smoke / training：

```text
c6_audit/C6A_OVERFIT_SMOKE.md
c6_audit/C6A_OVERFIT_SMOKE.json
c6_audit/C6A_TRAINING_AUDIT.md
c6_audit/C6A_TRAINING_AUDIT.json
c6_audit/C6A_TRAINING_AUDIT_HASHES.json
```

Final negative：

```text
c6_audit/C6A_NEGATIVE_AUDIT.md
c6_audit/C6A_NEGATIVE_MANIFEST.json
c6_audit/C6A_NEGATIVE_HASHES.json
```

Model artifacts：

```text
results/rlem_c6a/train_logs/c6a_default/
results/rlem_c6a/train_logs/c6a_large/
results/rlem_c6a/train_logs/c6a_regularized/
```

## 13. Final status

```text
C6A_PHASE_0_TO_7_COMPLETE = true
C6A_FREEZE_GATE_PASS = false
C6A_STATUS = C6A_NEGATIVE
official_val_used = false
post_val_adjustment = false
C6_B_used = false
C6_C_used = false
```

