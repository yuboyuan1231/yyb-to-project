# C6-B2 start state

- stage: `C6-B2 goal-mode multi-arm train_fit/train_calib`
- current_promoted_system: `C6-B1-lite c6b1r1_0057`
- previous_promoted_system: `C4_final v21_00444`
- official_val_used: `false`
- official_val_authorized: `false`
- post_val_adjustment: `false`
- C6_C_used: `false`
- C4_final_modified / NMS_modified / evaluator_modified: `false / false / false`
- primary metric: `0.7 R@1`
- secondary metric: `0.5 R@1`
- R@5/R@10: guardrails; R@100 reported only.

This starts train_fit/train_calib-only C6-B2. No official-val artifact may be read for selection or training.
