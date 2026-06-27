# C5 failure decomposition summary

## Final classification

`B. C5_MAIN_CANDIDATE`

The frozen temporal priors contain localization signal, the fixed candidate set has high oracle coverage, and the prior is not simply redundant with C4_final. The failure is concentrated in the external span-mass/residual interface: it converts sharp boundary evidence into a weak, duration-sensitive span score and barely moves best-IoU candidates.

This classification is a recommendation for human review only. C5-main was not run or authorized.

## 1. Prior–GT alignment

Among the 7,043 train_calib queries whose GT video appears in the fixed candidates:

- `P_ctx` peak inside GT: `19.35%`
- `P_bd` peak inside GT: `87.11%`
- `P_retloc` peak inside GT: `79.71%`
- Mean GT-window `P_retloc` mass: `0.3859`
- Mean maximum hard-negative span mass: `0.4509`
- GT-window mass exceeds the best hard negative in only `37.13%` of queries
- Prior-only oracle-video R@1@0.5 / R@1@0.7: `30.20% / 11.56%`
- Prior-only selected-span mIoU: `0.3879`

Interpretation: the boundary distributions point to the correct temporal region, but summing a mixed prior over a candidate span rewards broad or mismatched spans. `P_ctx` is weak; the useful signal is mainly in frozen begin/end boundary evidence.

## 2. Redundancy with C4_final

- Mean same-video Spearman between `P_retloc` span mass and C4_final: `0.0611`
- GT-video-only mean Spearman: `0.0584`
- Fraction of groups with Spearman ≥ 0.7: `3.06%`
- `P_retloc` mass vs `Q_bd`: Pearson/Spearman `0.3103 / 0.3387`
- `P_retloc` mass vs `Q_joint`: Pearson/Spearman `0.0165 / 0.1171`
- `P_retloc` peak lies inside the C4-selected span in `88.04%` of queries
- Best-IoU span minus C4-top prior mass is negative on average: `-0.0896`

Interpretation: prior information is not rank-redundant with C4_final, although boundary-derived features partially overlap `Q_bd`. The high peak overlap alongside low rank correlation again indicates that span mass/length, rather than peak location, is the problematic interface.

## 3. Candidate and NMS bottleneck

- GT video contains an IoU≥0.5 candidate in `96.27%` of covered queries
- GT video contains an IoU≥0.7 candidate in `85.62%`
- Raw-200 oracle mIoU: `0.8322`
- Post-NMS top-100 oracle R@1@0.5 / R@1@0.7: `94.58% / 80.93%`
- Best-IoU candidate is in the pre-NMS top 100 in `94.11%`
- Best-IoU candidate survives NMS in `56.64%`
- Better-than-C4-selected evidence is removed by NMS in `23.04%`, but post-NMS oracle remains high
- Best-IoU rank under C4_final / prior-only: mean `5.72 / 12.50`
- Rejected `c5p_00138` improves best-IoU rank in only `2.09%`, worsens it in `1.25%`, and changes mean rank by only `-0.008`

Interpretation: NMS suppresses many individual best-IoU spans, but usually retains another high-IoU alternative. Candidate/NMS capacity is therefore not the dominant explanation for the failed C5-lite residual.

## Recommendation

Close C5-lite-prior permanently under the current fixed-candidate external residual protocol. Do not extend its grid, lower its gate, freeze a nonzero scorer, or run official val.

If a later phase is authorized after human review, the evidence supports testing a narrowly specified C5-main design that injects frozen retrieval-conditioned boundary information at the begin/end-logit level, where start/end roles and duration constraints remain explicit. It does not support reusing raw span mass as another final-score residual, and weak `P_ctx` should not dominate the injection.

## Safety and provenance

- `official_val_used=false`
- `post_val_adjustment=false`
- `C5-main_used=false`
- `C6_used=false`
- C4_final, CONQUER, begin/end logits, candidate generation, NMS and evaluator were not modified.
- `c5p_00138` was used only as a rejected diagnostic reference, never promoted.

Execution stops here pending human review.
