# C6-A protocol

C6-A is an adapter-only trainable stage. It keeps `C4_final = C4-r2-cal-v2.1 v21_00444` frozen, does not modify the CONQUER backbone, does not regenerate candidates, and does not modify NMS or the evaluator.

Primary baseline: frozen C4_final. Secondary references: C4-lite, C3.1, and C5-main-A3b-wide official-val negative.

This phase trains only a lightweight-but-nontrivial adapter/head on train_fit and selects on train_calib. It validates whether the retrieval→localization temporal-prior signal that fixed-candidate C5 could not stably consume can be learned by a trainable branch.

Official val is not authorized in this phase. Phase 7 stops at freeze review or negative audit. Any official-val one-shot requires separate user authorization.
