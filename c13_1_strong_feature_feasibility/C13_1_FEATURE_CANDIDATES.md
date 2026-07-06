# C13-1 Feature Candidates

Status: `C13_STRONG_MULTISCALE_FEATURE_PROMISING`

Candidate feature families:

| family | candidates | feature dim | timestamp alignment | risk |
|---|---|---:|---|---|
| Text / subtitle | sentence-transformer style embedding, DeBERTa/RoBERTa/BERT re-encoding, token-level query-subtitle similarity | 768 typical | sentence offsets and clip centers must be preserved | low_to_medium |
| Visual | CLIP / EVA-CLIP / SigLIP image-video clip features, DINOv2 frame features | 768-1152 typical | feasible through sampled frame timestamps | medium_to_high |
| Motion | VideoMAE / InternVideo style motion tokens | 768-1408 typical | feasible with fixed clip windows | high |
| Multi-scale / event | clip tokens, event pooled tokens, inside/outside span context tokens | N x 768/1024 | best fit for boundary/localizer redesign | medium |

Small-sample audit requirements are represented in `C13_1_SAMPLE_COVERAGE_AUDIT.json`. The only uncovered natural bucket is `positive_not_in_b6_top100`, because the C12 CONQUER/B6 replay cache has 100% train-only GT-video top100 coverage; use native C12 missed-top100 aggregate as a hard-case proxy.

official_val_used = false
