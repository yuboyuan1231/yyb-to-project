# C5-main-inference-A2 comparison

## Final classification

`B. A2_LOCALIZATION_ONLY`

Both A2 variants preserve genuine localization improvements, but neither satisfies the frozen VCMR safety gate. No A2 scorer should be frozen or taken to official val.

## A2a: video-neutral residual

Best rejected nonzero: `c5ma_0273` (`alpha=0.25`, `beta=0.10`, `tau=2.0`, `lambda=0.025`, `gamma=0.025`).

- Oracle-video R@1@0.5 / R@1@0.7 deltas: `+0.05679 / +0.32657`
- Selected-span mIoU delta: `+0.000755`
- Mean best-IoU rank delta: `-0.01491`
- Hard-positive exits/entries: `237 / 27`
- Hard-positive exit ratio: `0.3777%`
- R@100 deltas at IoU 0.5 / 0.7: `+0.03433 / -0.01144`
- Video anchor mean/max drift: `0.04374 / 0.27535`
- Group residual mean max absolute value: `1.59e-7`

A2a substantially reduces the original A trade-off: original A had 1,030 hard-positive exits and `0.7 R@100=-0.24030`. Plain video centering lowers exits by 77% and nearly removes the R@100 loss. However, several primary metrics remain negative, so only zero control passes six-primary.

## A2b: strict video-anchor preservation

Best rejected nonzero: `c5ma_0098` (`alpha=0.0`, `beta=0.10`, `tau=1.0`, `lambda=0.025`, `gamma=0.05`).

- Oracle-video R@1@0.5 / R@1@0.7 deltas: `+0.01420 / +0.31237`
- Selected-span mIoU delta: `+0.000645`
- Mean best-IoU rank delta: `-0.02826`
- Hard-positive exits/entries: `1107 / 99`
- Hard-positive exit ratio: `1.7642%`
- R@100 deltas at IoU 0.5 / 0.7: `+0.01144 / -0.21742`
- Video anchor mean/max drift: exactly `0 / 0`
- Video top-10 anchor membership change: `0`

A2b successfully enforces its structural invariant: every query-video maximum is exactly the frozen C4_final anchor. Nevertheless, within-video span replacement changes which candidates survive global top-100/NMS, so hard positives still exit. Protecting only the group maximum is therefore insufficient to protect candidate-level VCMR recall.

## Direct answers

1. **Did A2a reduce R@100 loss and hard-positive exits?** Yes, strongly: exits `1030→237`; `0.7 R@100 -0.24030→-0.01144` relative to original A.
2. **Did A2b better protect video-level anchors?** Yes. Drift and top-10 video-anchor membership change are exactly zero.
3. **Which has stronger localization?** A2a is stronger on both oracle R@1 deltas and mIoU; A2b has the larger mean best-IoU rank improvement.
4. **Which has stronger VCMR safety?** A2a. It has much lower hard-positive exits and nearly neutral R@100.
5. **Did either pass the freeze gate?** No. In each grid, only zero control satisfies all six primary constraints; promotion-candidate count is zero.
6. **Recommend official-val one-shot?** No.
7. **Close C5-main?** Close A, A2a and A2b as fixed-candidate inference scorers. Do not continue their grids.
8. **Recommend C5-main-B or candidate regeneration?** Do not automatically start either. If human review authorizes one follow-up, C5-main-B is better supported than candidate regeneration because candidate oracle coverage is already high. A new protocol must explicitly protect candidate-level top-100/NMS behavior, not only video anchors.
9. **Recommend C6?** No.

## Safety

- `official_val_used=false`
- `post_val_adjustment=false`
- `C5-main-B_used=false`
- `candidate_regeneration_used=false`
- `C6_used=false`
- C4_final, CONQUER checkpoint/source, candidates, NMS, effective_top_n and evaluator remain unchanged.

Execution stops pending human review.
