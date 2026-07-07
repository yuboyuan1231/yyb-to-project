# C26R Final Decision

Status: `C26R_MULTI_PROPOSAL_REPAIR_PROMISING`.

Best inference method by calib_select: `R3b1_C26_PREM_guarded_localizer`.

Holdout VCMR R@100@0.5: `34.0` versus C26 single-span `8.555555555555555`.

Highest inference holdout R@100@0.5: `R1c_zero_delta_c26_final_video_multi_span` = `36.666666666666664`.

Diagnostic-only oracle headroom R@100@0.5: `71.66666666666667`; excluded from selection because it uses GT IoU.

C26R does not use official validation or pseudo-official selection and does not promote a system.
