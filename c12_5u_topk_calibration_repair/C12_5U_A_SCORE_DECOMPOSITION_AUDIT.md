# C12-5U-A Score Decomposition Audit

status = C12_5U_SCORE_DECOMPOSITION_COMPLETE_FULL

T2 global Spearman = -0.1930954945834624
T2 top100 Spearman = 0.033238826444936434
T2 top500 Spearman = -0.1090988223901011

Duration Spearman = {'long': -0.3229349567006851, 'medium': -0.198415537441486, 'short': -0.21534637388511318}
Query-type Spearman = {'t': -0.1487189531946739, 'v': -0.217565957559689, 'vt': -0.1738461015800601}

Conclusion: T2 is an effective topK selector. Spearman is negative globally and top500, but top100 is only weakly positive near zero, so calibration remains unusable even though topK promotion works.

official_val_used = false
