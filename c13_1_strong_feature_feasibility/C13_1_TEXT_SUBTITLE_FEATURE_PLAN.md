# C13-1 Text Subtitle Feature Plan

Decision: strong text/subtitle features are worth a pilot.

Plan:

1. Re-encode queries and subtitle sentences with a stronger text encoder.
2. Build sentence-level and token-level query-subtitle similarity features.
3. Preserve subtitle timestamps, sentence ids, and clip-center alignment.
4. Evaluate only on train_fit/calib_select/calib_holdout sample buckets first.

Expected value: highest for t/vt queries and for failures where current score assigns high confidence to wrong boundary spans.

Risk level: low_to_medium.

official_val_used = false
