# C28E Issue Repair Matrix

This branch is code-review first. It fixes the blueprint-mechanism gaps before any new full experiment is run.

| Issue | C28D Problem | C28E Code Fix | Review Target |
|---|---|---|---|
| PR scope polluted | PR #8 was based on `main`, so historical/C28C files appeared in the diff | Branch from `c28c-cleanroom-e2e-blueprint-v2`; open PR against that base | GitHub PR changed-files scope |
| Candidate eval truncation | `hard_negative_k + 1` truncated eval candidates | `candidate_topk_train`, `candidate_topk_eval`, and `eval_candidate_k` are separate | `engine/train.py`, `engine/evaluate.py` |
| Candidate duration bug | Wrong candidate videos reused query/GT video duration | Candidate proposals use `video_id -> duration` and per-candidate span masks | `data/proposal_dataset.py`, `data/collate.py` |
| Fake moment-aware label | `use_moment_aware_bank` existed but final weight was `0.0` | C28E uses real clip visual/subtitle/joint late interaction | `engine/c28e_late_interaction.py` |
| No clip-level retrieval | Retriever was global query embedding to global video mean embedding | Two-stage current-student pooled broad topK then clip-level rerank topK | `two_stage_retrieve` |
| Padding pollution | Sequence banks padded/truncated to 64 clips but late interaction had no valid-clip mask | Visual/subtitle/joint clip masks are cached and used by pooled, PREM, late softTopK, and token MaxSim paths | `data/*_bank.py`, `models/full_model.py`, `engine/c28e_late_interaction.py` |
| Video encoder bypass | Pre-encoded banks in training would prevent video encoder gradients | Training scores raw candidate clips through `VideoSubtitleEncoder` | `late_scores_for_raw_candidates` |
| Monolithic GPU candidate tensor | Raw scoring used one `B x C x T x D` tensor, risking OOM and tempting candidate reduction | Candidate scoring is chunked with `candidate_encode_chunk` while preserving `candidate_topk_train=200` | `late_scores_for_raw_candidates` |
| Retriever-only branch | C28E late score existed outside the full VCMR graph | `C28CFullModel` can use late retriever score before localizer, feedback, and joint VCMR scorer | `models/full_model.py` |
| Broad-stage blind spot | Late rerank could not reveal whether GT was already absent from pooled broad topK | Replay reports broad pooled rank metrics separately from late rerank metrics | `two_stage_retrieve` |
| Missing curriculum | Candidate mix was fixed teacher/student/random, not the requested staged curriculum | Epoch curriculum moves from teacher-heavy to student-dynamic and audits source counts | `train_c28e_retriever` |
| Query padding in miner | Full candidate refresh encoded padded query tokens without mask | `refresh_candidates` now passes query masks to the query encoder | `engine/refresh_hard_negatives.py` |
| Train metric selection | Best checkpoint used train `VR@100` | Best checkpoint uses `calib_select` selection score only | `train_c28e_retriever` |
| Holdout leakage risk | Holdout was repeatedly used for gate/sweeps | Holdout final stage requires `--allow_holdout_final` | `run_c28e_clip_late_interaction_retriever.py` |
| Weak teacher distillation | Teacher score schema was not audited | First-stage reference exposes score audit and item scores when present | `data/first_stage_reference.py` |
| Resume report overwrite | Resume overwrote prior logs | Training log is loaded and appended on resume | `train_c28e_retriever` |
| Naming/report confusion | C28D reports reused C28D-1 names under later stages | C28E stages and report names are C28E-specific | `run_c28e_clip_late_interaction_retriever.py` |
| Premature full experiment | Full mutual experiment could be tempting despite retriever failure | Full-model code path exists for review, but full experiments remain gated until retriever passes | C28E runner stages |
