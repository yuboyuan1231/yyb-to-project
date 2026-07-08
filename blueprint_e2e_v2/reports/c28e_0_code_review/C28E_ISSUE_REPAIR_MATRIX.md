# C28E Issue Repair Matrix

This branch is code-review first. It fixes the blueprint-mechanism gaps before any new full experiment is run.

| Issue | C28D Problem | C28E Code Fix | Review Target |
|---|---|---|---|
| PR scope polluted | PR #8 was based on `main`, so historical/C28C files appeared in the diff | Branch from `c28c-cleanroom-e2e-blueprint-v2`; open PR against that base | GitHub PR changed-files scope |
| Candidate eval truncation | `hard_negative_k + 1` truncated eval candidates | `candidate_topk_train`, `candidate_topk_eval`, and `eval_candidate_k` are separate | `engine/train.py`, `engine/evaluate.py` |
| Candidate duration bug | Wrong candidate videos reused query/GT video duration | Candidate proposals use `video_id -> duration` and per-candidate span masks | `data/proposal_dataset.py`, `data/collate.py` |
| Fake moment-aware label | `use_moment_aware_bank` existed but final weight was `0.0` | C28E uses real clip visual/subtitle/joint late interaction | `engine/c28e_late_interaction.py` |
| No clip-level retrieval | Retriever was global query embedding to global video mean embedding | Two-stage current-student pooled broad topK then clip-level rerank topK | `two_stage_retrieve` |
| Video encoder bypass | Pre-encoded banks in training would prevent video encoder gradients | Training scores raw candidate clips through `VideoSubtitleEncoder` | `late_scores_for_raw_candidates` |
| Train metric selection | Best checkpoint used train `VR@100` | Best checkpoint uses `calib_select` selection score only | `train_c28e_retriever` |
| Holdout leakage risk | Holdout was repeatedly used for gate/sweeps | Holdout final stage requires `--allow_holdout_final` | `run_c28e_clip_late_interaction_retriever.py` |
| Weak teacher distillation | Teacher score schema was not audited | First-stage reference exposes score audit and item scores when present | `data/first_stage_reference.py` |
| Resume report overwrite | Resume overwrote prior logs | Training log is loaded and appended on resume | `train_c28e_retriever` |
| Naming/report confusion | C28D reports reused C28D-1 names under later stages | C28E stages and report names are C28E-specific | `run_c28e_clip_late_interaction_retriever.py` |
| Premature full model | Full mutual model could be tempting despite retriever failure | Full model remains absent and gated until retriever passes | C28E runner stages |
