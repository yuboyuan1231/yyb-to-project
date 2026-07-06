# C13-1 Multiscale Event Feature Plan

Decision: multiscale/event features are the most promising pivot target.

Plan:

1. Pool frame/clip tokens into 8-32 event tokens per video.
2. Build span inside/outside context features over event tokens.
3. Keep both clip-level and event-level representations so boundary models can attend across scales.
4. Use C12 M1000 failure cases to test whether event tokens move high-IoU spans upward.

Expected value: directly targets the C12 failure mode where oracle span exists in M1000 but rank/calibration does not find it.

Risk level: medium.

official_val_used = false
