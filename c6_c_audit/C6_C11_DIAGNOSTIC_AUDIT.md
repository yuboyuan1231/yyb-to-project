# C6-C1.1 diagnostic audit

- Status: `C6_C11_DIAGNOSTICS_COMPLETE`
- Official val used: `false`
- C6-C0 proposal_confidence identical 0.7 recalls are an oracle-definition artifact, not a promotion signal.

## Query Counts

```json
{
  "conservative": {
    "enabled_queries": 0,
    "top1_gain_queries": 86,
    "top1_loss_queries": 52,
    "top5_gain_queries": 30,
    "top5_loss_queries": 26,
    "top10_gain_queries": 24,
    "top10_loss_queries": 19,
    "lost_R5_but_gain_R1_queries": 0,
    "lost_R10_but_gain_R1_queries": 0
  },
  "balanced": {
    "enabled_queries": 0,
    "top1_gain_queries": 86,
    "top1_loss_queries": 52,
    "top5_gain_queries": 30,
    "top5_loss_queries": 26,
    "top10_gain_queries": 24,
    "top10_loss_queries": 19,
    "lost_R5_but_gain_R1_queries": 0,
    "lost_R10_but_gain_R1_queries": 0
  },
  "R1-only": {
    "enabled_queries": 0,
    "top1_gain_queries": 86,
    "top1_loss_queries": 52,
    "top5_gain_queries": 30,
    "top5_loss_queries": 26,
    "top10_gain_queries": 24,
    "top10_loss_queries": 19,
    "lost_R5_but_gain_R1_queries": 0,
    "lost_R10_but_gain_R1_queries": 0
  }
}
```
