# C5 failure decomposition: prior redundancy with C4_final

Correlations and rank agreement are descriptive only; no coefficient is fitted.

## Prior/head correlations

| Diagnostic | Value |
|---|---:|
| `ctx_mass vs q_joint Pearson` | -0.03921923 |
| `ctx_mass vs q_joint Spearman` | 0.00117994 |
| `ctx_mass vs q_bd Pearson` | 0.08311657 |
| `ctx_mass vs q_bd Spearman` | 0.12035735 |
| `ctx_mass vs e_fp Pearson` | -0.04063607 |
| `ctx_mass vs e_fp Spearman` | -0.07714011 |
| `bd_mass vs q_joint Pearson` | 0.20278399 |
| `bd_mass vs q_joint Spearman` | 0.28816286 |
| `bd_mass vs q_bd Pearson` | 0.43359511 |
| `bd_mass vs q_bd Spearman` | 0.43256819 |
| `bd_mass vs e_fp Pearson` | 0.18645950 |
| `bd_mass vs e_fp Spearman` | 0.20755535 |
| `retloc_mass vs q_joint Pearson` | 0.01645839 |
| `retloc_mass vs q_joint Spearman` | 0.11712638 |
| `retloc_mass vs q_bd Pearson` | 0.31025335 |
| `retloc_mass vs q_bd Spearman` | 0.33874419 |
| `retloc_mass vs e_fp Pearson` | 0.08427397 |
| `retloc_mass vs e_fp Spearman` | 0.08682470 |
| `ctx_boundary_agree vs q_joint Pearson` | -0.02175801 |
| `ctx_boundary_agree vs q_joint Spearman` | -0.04067892 |
| `ctx_boundary_agree vs q_bd Pearson` | -0.03593203 |
| `ctx_boundary_agree vs q_bd Spearman` | -0.03620028 |
| `ctx_boundary_agree vs e_fp Pearson` | 0.04987762 |
| `ctx_boundary_agree vs e_fp Spearman` | -0.01902857 |
| `bd_boundary_agree vs q_joint Pearson` | 0.19148148 |
| `bd_boundary_agree vs q_joint Spearman` | 0.21840520 |
| `bd_boundary_agree vs q_bd Pearson` | 0.32097660 |
| `bd_boundary_agree vs q_bd Spearman` | 0.31050007 |
| `bd_boundary_agree vs e_fp Pearson` | 0.51224634 |
| `bd_boundary_agree vs e_fp Spearman` | 0.55449648 |

## Within-video agreement

| Diagnostic | Value |
|---|---:|
| `all-group mean Spearman` | 0.06111901 |
| `GT-group mean Spearman` | 0.05844397 |
| `groups with Spearman >= 0.7` | 0.03064598 |
| `P_retloc peak inside C4 top span` | 0.88042110 |

## Prior-mass comparison

| Diagnostic | Value |
|---|---:|
| `C4 selected mass mean` | 0.39590207 |
| `best-IoU minus C4-top mass mean` | -0.08962482 |
| `best-IoU minus C4-top mass median` | -0.09289142 |
