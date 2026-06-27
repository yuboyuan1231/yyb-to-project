# C6-B2 training audit

Official val used: `false`

| arm | epochs | final loss | feasible | 0.7 R@1 Δ | note |
|---|---:|---:|---:|---:|---|
| listwise_set | 4 | 1.757949903685385 | False | -1.1214 | initial arm |
| listwise_set_retry1 | 5 | 1.4828434942280435 | False | -0.9612 | retry after listwise underperformance |
| listwise_set_retry2 | 5 | 0.9416360357356945 | False | -0.3318 | retry after listwise underperformance |
| pairwise_main | 4 | 3.784226127727303 | True | +0.3891 | selected feasible |

Listwise was audited after poor initial metrics and retried twice (`sqrt_cap`, then `none` class weighting). Both retries failed the C6-B2 freeze gate, so no further listwise tuning is performed in this run.
