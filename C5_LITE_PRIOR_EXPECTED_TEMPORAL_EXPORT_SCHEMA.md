# Expected C5-0 Temporal Prior Export Schema

C5-lite-prior requires a read-only export from CONQUER/QAL/ML outputs.  Do not use row-level candidate scores as a substitute.

## Required arrays

### Dense format

```text
group_id: int64 [G]
p_ctx: float32 [G, T]
p_b: float32 [G, T]
p_e: float32 [G, T]
```

### Ragged format

```text
group_id: int64 [G]
temporal_offsets: int64 [G+1]
p_ctx: float32 [sum_T]
p_b: float32 [sum_T]
p_e: float32 [sum_T]
```

`group_id` must exactly match the C4 cache `group_ids_sorted_unique` order.

## Optional arrays

```text
clip_start_time
clip_end_time
```

Dense shape: `[G, T]`.
Ragged shape: `[sum_T]`.

These should come from the actual video clip timeline.  If unavailable, C5 aggregation falls back to `--clip_length`, default `1.5`.

## Meaning of each prior

```text
p_b(t): begin/start probability from CONQUER ML head
p_e(t): end probability from CONQUER ML head
p_ctx(t): QAL-derived query-context temporal evidence
```

`p_ctx(t)` must be a temporal distribution or nonnegative temporal evidence over clips.  Suggested sources:

```text
1. QAL Query2Video / Video2Query attention-derived clip relevance;
2. clip-query similarity from QAL hidden states;
3. query pooled embedding cosine with QAL clip hidden states;
4. another read-only QAL temporal evidence path.
```

It must not be derived from labels or official-val outcomes.

## JSONL adapter

If Codex exports JSONL first, use:

```bash
python3 rlem/export_c5_temporal_prior_from_jsonl.py \
  --temporal_jsonl <split_temporal_prior.jsonl.gz> \
  --cache_npz <split_c4_cache.npz> \
  --output_npz <split_c5_temporal_prior.npz> \
  --output_audit_json <audit.json> \
  --split_role train_fit
```

Each JSONL row should contain:

```json
{
  "group_id": 123,
  "desc_id": 456,
  "video_idx": 789,
  "p_ctx": [0.1, 0.2, 0.7],
  "p_b": [0.2, 0.5, 0.3],
  "p_e": [0.1, 0.3, 0.6],
  "clip_start_time": [0.0, 1.5, 3.0],
  "clip_end_time": [1.5, 3.0, 4.5]
}
```
