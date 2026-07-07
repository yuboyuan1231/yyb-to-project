# C26-7 VCMR R@100 Sanity Audit

Status: `C26_R100_SANITY_PASS_SHARED_SINGLE_SPAN_LIMITATION`.

- C24H selected B on same subset VCMR R@100@0.5: `8.555555555555555`
- C26 final F on same subset VCMR R@100@0.5: `8.555555555555555`
- Span unit: `seconds_consistent`
- One span per query-video: `True`

C24H on the same C26 medium subset is also low at VCMR R@100, while VR R@100 is near-saturated. The gap is a shared one-span/localizer proposal limitation, not a C26-only scoring regression.
