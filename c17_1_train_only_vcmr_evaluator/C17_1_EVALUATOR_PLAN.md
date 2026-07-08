# C17-1 Evaluator Plan

Build final video-span rankings from train-only candidate rows, apply historical
temporal suppression per query/video, then compute VCMR R@K@IoU, VR recall,
GT-video-only span recall, and error breakdowns.

Medium mode replays retriever top100 videos, so VR@100 and VCMR@100 are
computed from an actual multi-video candidate pool, not GT-video localizer-only
coverage.
