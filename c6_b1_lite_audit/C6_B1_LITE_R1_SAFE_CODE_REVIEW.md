# C6-B1-lite R1-safe selector code review

## Current design assessment

The current implementation is a reasonable first positive version:

- medium-scale selector, not a tiny heuristic:
  - 4 hidden blocks;
  - hidden dimension 384;
  - LayerNorm + GELU + Dropout;
  - 3.12M train_fit candidate examples.
- fixed C4_final video slots;
- sparse, gated span replacement;
- R@1-focused selection with R@5 safety gate;
- train_fit trains the selector, train_calib selects the replacement policy;
- no official-val access.

This structure directly addresses the C6-B0 failure: B0 replaced about 88% of top slots and collapsed R@5; B1-lite replaces only about 1.64% of top slots and improves all eight pseudo metrics.

## Code issue found during review

### Fixed: hard-positive movement reference

The first movement diagnostic compared candidate post-NMS output against original pre-NMS top100 positives. This made the baseline itself show hard-positive exits. The metric was corrected to compare:

```text
candidate post-NMS top100
vs
pseudo C4_final post-NMS top100
```

After rerun:

- best config unchanged: `c6b1r1_0057`;
- eight metric deltas unchanged;
- corrected hard-positive exits: `0`;
- corrected hard-positive entries: `8`.

## Main bottleneck in the current code

The slowest component is not model training. Training is fast on GPU. The bottleneck is Python-loop policy evaluation:

```text
for each grid config
  for each query
    reconstruct top100 slots
    run temporal NMS
    compute labels
```

This is acceptable for the current 78-config review, but it will become painful if we do richer local search.

Recommended engineering improvement:

1. Precompute baseline post-NMS selected indices once.
2. Precompute per-query, per-policy replacement decisions in vectorized arrays.
3. Cache per-query candidate labels before NMS.
4. Restrict NMS recomputation to queries where a replacement actually happens.
5. Optionally multiprocessing over configs, since configs are independent.

This would preserve exactness while making local robustness sweeps much cheaper.

## Model structure improvements likely to help R@1

### 1. Slot-conditioned two-head selector

Current selector uses one shared MLP for all slots and distinguishes slot rank through features. A better structure would explicitly split:

```text
shared candidate encoder
  -> top1 head
  -> top2/top3 head
  -> quality head
```

Reason: the optimal risk profile for slot 1 is different from slots 2/3. R@1 wants aggressive precision at slot 1; R@5 wants conservative diversity preservation below it.

Expected effect:

- better R@1 calibration;
- fewer high-R1 / high-top1-change rejected configs.

### 2. Listwise alternative selector per slot

Current model scores each candidate independently. A more natural candidate-interface model is listwise:

```text
for each query-video-slot:
  original span + boundary alternatives
  -> small Transformer / DeepSets encoder
  -> softmax over alternatives
```

Loss:

```text
L = CE(best alternative by IoU0.5/0.7)
  + margin(original vs replacement)
  + smooth quality regression
```

Reason: replacement is a relative decision: “is this boundary alternative better than the original span?” Independent scoring throws away that comparison structure.

Expected effect:

- more reliable replacement margins;
- better threshold stability;
- less train_calib threshold sensitivity.

### 3. Pairwise original-vs-boundary margin loss

Add pairwise constraints only when the boundary alternative truly improves the label:

```text
if alt_y > original_y:
  score(alt) >= score(original) + margin
else:
  score(original) >= score(alt) + margin
```

For the R1-focused objective, weight slot 0 more heavily:

```text
slot_weight = 4.0 for rank 1
            = 1.5 for rank 2-5
            = 0.5 otherwise
```

Expected effect:

- better R@1 without broad replacement;
- better control of false replacement.

### 4. Top5 diversity / do-no-harm regularizer

The current gate protects R@5 after the fact. A next model can include the safety idea in training:

```text
L_safety = penalty if a replacement candidate would suppress an original top5 positive under NMS
```

This can be approximated cheaply with overlap and original-candidate IoU labels.

Expected effect:

- fewer R@5-negative configs;
- less reliance on grid gates.

### 5. Better candidate pool construction

Current alternatives are mainly top P_b/P_e combinations. Add local alternatives around both boundary peaks and the original span:

```text
original span
original ± 1 / ± 2 clips
boundary peak combinations
duration-prior-constrained boundary spans
```

Reason: R@1 may need small local repairs rather than jumps to global boundary peaks.

Expected effect:

- higher precision replacements;
- lower top1_changed_ratio for the same R@1 gain.

### 6. Train-fit inner split for threshold calibration

Right now train_calib chooses policy thresholds. For a cleaner freeze path, split train_fit internally:

```text
train_fit_inner_train -> train selector
train_fit_inner_gate  -> choose thresholds
train_calib           -> read-only confirmation
```

This would make train_calib closer to a validation confirmation rather than the threshold-selection set.

Expected effect:

- stronger review posture before official val;
- lower risk of train_calib over-selection.

## Recommended next implementation order

If continuing this line, I would not broaden the current grid first. I would do:

1. Freeze/review current `c6b1r1_0057` as the first positive candidate.
2. Implement a listwise slot selector as `C6-B1-lite-v2`.
3. Add pairwise original-vs-boundary margin loss.
4. Keep replacement sparse and preserve the same R@5 movement gates.
5. Only after v2 train_calib review, decide whether one-shot official val is justified.

## Current freeze recommendation

Current code is good enough for train_calib freeze review after the movement fix.

Do not yet treat it as final C6. The result is promising because it solves the immediate B0 failure mode, but it still needs a formal freeze manifest and one-shot authorization before official-val use.
