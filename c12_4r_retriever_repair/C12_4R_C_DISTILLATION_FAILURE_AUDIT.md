# C12-4R-C Distillation Failure Audit

Original C12-4 teacher distillation reached only R@100 6.75 because it learned
from sampled negative batches without preserving CONQUER's corpus-level ranking
as the base score. C12-4R fixes this by keeping CONQUER first-stage score as
`S_conquer_init` and training only a small delta.

```json
{
  "teacher_score_coverage_full_corpus": "No; C12-4 student saw sampled negatives and sparse teacher topK.",
  "student_only_saw_sampled_negatives": true,
  "teacher_topK_entered_training_batch": "partially in C12-4; explicitly top128 in C12-4R",
  "gt_video_always_positive": true,
  "hard_negatives_hard_enough": "improved in C12-4R via teacher topK + group negatives",
  "kl_weight_issue": "C12-4 KL was not enough to preserve teacher ordering over corpus",
  "v_query_visual_bridge_weak": true,
  "subtitle_branch_near_invalid": true
}
```
