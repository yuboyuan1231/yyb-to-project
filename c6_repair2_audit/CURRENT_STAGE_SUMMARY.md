# CONQUER-RLEM 当前阶段结果总结

生成时间：2026-06-25  
总结范围：C4 final、C5 fixed-candidate、C6-A、C6-A-R、C6-A-R2  
当前主线结论：保留 `C4_final = C4-r2-cal-v2.1 v21_00444`，后续 C5/C6 fixed-candidate / adapter-only 系列均未 promotion。

## 1. 当前总状态

```text
C4_stage_status = COMPLETE
C4_final_system = C4-r2-cal-v2.1
C4_final_config = v21_00444

C5 fixed-candidate family = official-val negative / not promoted
C6-A adapter-only trainable mutualization = negative
C6-A-R teacher-guided repair = negative
C6-A-R2 boundary-oracle-initialized ranking-aware adapter = negative
```

当前不应进入：

```text
official val
C6-B
candidate regeneration
C6-C
full backbone fine-tuning
C4-main / VS-R2 head
```

除非后续另开新协议并明确授权。

## 2. C4：当前最终主结果

C4 已冻结完成：

```text
C4_final_system = C4-r2-cal-v2.1
C4_final_config = v21_00444
C4_core_module = C4-lite
primary_baseline = frozen C4-lite c4_00394
secondary_baseline = frozen C3.1
```

最终 scorer：

```text
S_C4_final =
  S_C4_lite
  + 0.075 * z(rel_logit)
  + 0.050 * z(iou07_logit)
  - 0.100 * z(quality_logit)
```

C4 阶段结论：

- C4-lite 是主要增益来源；
- C4-r2-cal-v2.1 是最终冻结输出；
- C4-main / VS-R2 head 未进入；
- `model/conquer.py` 未修改；
- candidate generation / NMS / evaluator 未修改；
- official-val one-shot 已完成；
- post-val adjustment=false。

因此当前主线基准应继续使用：

```text
C4_final = C4-r2-cal-v2.1 v21_00444
```

## 3. C5：fixed-candidate 后处理未 promotion

C5 系列包括：

```text
C5-lite-prior
C5-main-inference-A
C5-main-inference-A2
C5-main-inference-A3 / A3b-wide
```

最终 official-val one-shot：

```text
config = c5ma_0214
status = OFFICIAL_C5_NEGATIVE
```

关键判断：

- A3b-wide 在 train_calib 上有 localization signal；
- official val 上没有确认 promotion；
- fixed-candidate 后处理不能稳定把 prior 转成 front-rank VCMR 增益；
- 不应继续 C5 fixed-candidate scorer 小修小补。

保留的有价值信息：

```text
C5/A3b 暴露出 boundary/localization prior 确实存在；
但 fixed-candidate reranking 消费这个 prior 的方式不稳定。
```

## 4. C6-A：adapter-only mutualization negative

C6-A 目标：

```text
冻结 CONQUER backbone；
保留 C4_final；
训练 adapter/head；
尝试把 retrieval→localization prior 变成 front-rank VCMR 增益。
```

工程链路结果：

```text
train_fit cache PASS
train_calib cache PASS
zero-adapter max_abs_diff = 0.0
overfit smoke PASS
GPU forward/backward PASS
official_val_used = false
```

模型配置：

```text
c6a_default
c6a_large
c6a_regularized
```

最佳 train_calib 配置：

```text
selected_config = c6a_regularized
selected_epoch = 1
status = C6A_NEGATIVE
```

Best deltas vs C4_final：

```text
0.5-r1   -0.01144
0.5-r5   +0.04577
0.5-r10  +0.00000
0.5-r100 +0.04577
0.7-r1   -0.01144
0.7-r5   -0.04577
0.7-r10  -0.04577
0.7-r100 +0.01144
```

Freeze gate：

```text
core_gate = false
front_positive_count = 1
localization_positive_count = 0
movement_gate = true
r100_gate = true
r1_gate = false
```

解释：

```text
C6-A movement 很安全，但没有学到有效 localization；
adapter 没能把 temporal prior 转化为 front-rank 增益。
```

## 5. C6-A-R：诊断 + teacher-guided repair

C6-A-R 目的：

```text
先诊断 C6-A 为什么没学到 localization；
再测试 boundary-only oracle；
最后尝试 A3b teacher distillation。
```

### 5.1 Phase 1 diagnostics

诊断结果：

- BoundaryAdapter 梯度不是 0；
- endpoint_delta 幅度存在；
- 但 C6-A endpoint_delta 与 A3b teacher delta 相关性弱或反向；
- 说明原 C6-A 没学到 A3b 的归纳偏置。

### 5.2 Phase 2 boundary oracle

Boundary-only oracle 通过：

```text
status = PASS
```

关键 delta vs frozen：

```text
start_ce_delta          = -0.38601
end_ce_delta            = -0.37567
oracle_video_r1_05_delta = +0.001846
oracle_video_r1_07_delta = +0.002840
selected_span_iou_delta = +0.001152
best_iou_rank_delta     = -0.478915
```

这非常重要：它说明边界本身是能学的，C6-A negative 不是因为 prior 完全无用，也不是因为 adapter 架构完全失效。

### 5.3 Phase 3 A3b teacher distillation

结果：

```text
status = FAIL
stop_phase = Phase 3 A3b teacher distillation
stop_reason = teacher_adapter_failed_to_inherit_a3b_endpoint_delta_direction
```

endpoint_delta 与 A3b raw delta 相关性：

```text
Pearson  = -0.47737
Spearman = -0.40911
```

解释：

```text
A3b teacher 不是可靠 boundary teacher；
teacher distillation 没有继承 A3b endpoint residual 方向；
因此不进入 C6-A-R final ranking/fusion。
```

## 6. C6-A-R2：boundary-oracle init + ranking-aware adapter

C6-A-R2 的修正判断：

```text
A3b teacher 不再作为主监督；
使用 BoundaryOracleAdapter 初始化；
保留 boundary supervision；
训练 ranking-aware adapter；
尝试把 boundary improvement 转成 fixed-candidate front-rank gain。
```

执行配置：

```text
c6a_r2_default
c6a_r2_large
c6a_r2_safe
```

其中 `c6a_r2_safe` 是在 default/large 出现 R@1 正向但 R@5/R@10 大幅塌缩后追加的 conservative 配置，用于验证是否只是 residual α 过强。

### 6.1 Boundary init 结果

Boundary init 通过：

```text
C6A_R2_BOUNDARY_INIT_AUDIT = PASS
```

说明：

- default boundary init 可复现 C6-A-R Phase 2 boundary oracle；
- large boundary init 也完成；
- Stage 1 boundary-retention warmup 能保持 boundary 能力。

### 6.2 Ranking-aware continuation 结果

default / large 的典型现象：

```text
R@1 明显正向；
R@5 / R@10 大幅负向；
front-rank distribution 发生不安全 tradeoff。
```

safe 配置减小 residual α 后：

```text
塌缩幅度减轻；
但 front-rank 仍未转正；
selection score 仍为负。
```

最终 selected config：

```text
selected_config = c6a_r2_safe
selected_epoch = 3
selected_selection_score = -0.6180705
status = C6A_R2_NEGATIVE
```

Selected deltas vs C4_final：

```text
0.5-r1   -0.01144
0.5-r5   -1.00698
0.5-r10  -1.02987
0.5-r100 -0.05721
0.7-r1   -0.08010
0.7-r5   -0.30896
0.7-r10  -0.17164
0.7-r100 +0.22886
```

Freeze gates：

```text
front_positive_count = 0
r1_gate = false
r100_gate = false
localization_positive_count = 3
movement_gate = true
boundary_retention_gate = true
best_iou_rank_gate = true
core_gate = false
strong_gate = false
```

解释：

```text
Boundary/localization 改善是成立的；
但是 fixed-candidate ranking 转换失败；
边界增强会改变 top spans，但无法稳定提升 R@5/R@10/front-rank。
```

## 7. 当前综合判断

截至目前，证据链比较清楚：

```text
1. C4_final 是当前唯一 promotion 后的主线系统；
2. C5 fixed-candidate prior 有 signal，但 official val negative；
3. C6-A 普通 adapter 没学到 localization；
4. C6-A-R boundary oracle 证明 boundary 可以学；
5. A3b 不适合作为 boundary teacher；
6. C6-A-R2 证明 boundary init 可保持 localization，但 fixed-candidate front-rank 仍无法安全获益。
```

所以当前最准确的阶段结论是：

```text
Boundary distribution learning works.
Fixed-candidate reranking remains the bottleneck.
C4_final should remain the main frozen result.
Do not run official val for C6-A / C6-A-R / C6-A-R2.
Do not auto-enter C6-B.
```

## 8. 是否建议 C6-B？

当前不自动建议进入 C6-B。

但如果未来人工考虑 C6-B，理由只能是：

```text
boundary/localization 已经有正向；
best-IoU rank 可以被 boundary oracle 改善；
fixed candidates / NMS / candidate pool 可能限制了 front-rank 转化。
```

不过 C6-A-R2 的 ranking-aware fixed-candidate 失败说明：

```text
即使边界改善成立，也不能保证 candidate regeneration 会自然转成 VCMR 增益；
C6-B 需要单独协议、强审计、严格 train_fit/train_calib gate。
```

因此当前推荐：

```text
停止 C6 fixed-candidate / adapter-only 当前线；
保留 C4_final；
如要继续，另开新协议，而不是继续小幅调参。
```

## 9. 文件索引

C4 final：

```text
c4_audit/C4_STAGE_FINAL_SUMMARY.md
c4_audit/C4_STAGE_FINAL_MANIFEST.json
c4_audit/C4_STAGE_FINAL_HASHES.json
```

C5 official negative：

```text
c5_main_a3_video_slot_official_val/C5_MAIN_A3_OFFICIAL_VAL_ONE_SHOT_AUDIT.md
c5_main_a3_video_slot_official_val/C5_MAIN_A3_OFFICIAL_VAL_ONE_SHOT_MANIFEST.json
c5_main_a3_video_slot_official_val/C5_MAIN_A3_OFFICIAL_VAL_ONE_SHOT_HASHES.json
```

C6-A：

```text
c6_audit/C6A_EXECUTION_REPORT.md
c6_audit/C6A_NEGATIVE_AUDIT.md
c6_audit/C6A_NEGATIVE_MANIFEST.json
c6_audit/C6A_NEGATIVE_HASHES.json
```

C6-A-R：

```text
c6_repair_audit/C6A_R_DIAGNOSTIC_AUDIT.md
c6_repair_audit/C6A_R_BOUNDARY_ORACLE_AUDIT.md
c6_repair_audit/C6A_R_TEACHER_AUDIT.md
c6_repair_audit/C6A_R_NEGATIVE_AUDIT.md
c6_repair_audit/C6A_R_NEGATIVE_MANIFEST.json
c6_repair_audit/C6A_R_NEGATIVE_HASHES.json
```

C6-A-R2：

```text
c6_repair2_audit/C6A_R2_A3B_TEACHER_DIAGNOSTIC.md
c6_repair2_audit/C6A_R2_BOUNDARY_INIT_AUDIT.md
c6_repair2_audit/C6A_R2_TRAINING_AUDIT.md
c6_repair2_audit/C6A_R2_NEGATIVE_AUDIT.md
c6_repair2_audit/C6A_R2_NEGATIVE_MANIFEST.json
c6_repair2_audit/C6A_R2_NEGATIVE_HASHES.json
```

## 10. Final recommendation

```text
Current frozen main result:
  C4_final = C4-r2-cal-v2.1 v21_00444

Current C5/C6 status:
  C5 fixed-candidate = negative
  C6-A = negative
  C6-A-R = negative
  C6-A-R2 = negative

Recommended action:
  Stop current fixed-candidate / adapter-only line.
  Do not run official val.
  Do not continue small grid tuning.
  Keep C4_final as the stable result.
```

