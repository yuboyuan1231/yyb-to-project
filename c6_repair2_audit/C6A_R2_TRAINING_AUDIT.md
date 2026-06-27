# C6-A-R2 training audit

- Status: `C6A_R2_TRAINING_COMPLETE_NO_GATE_PASS`
- Official val used: `false`
- A3b teacher loss used: `false`

## c6a_r2_default
- Best epoch: `7`
- Best selection score: `-6.700320578952598`
- Gates: `{'front_positive_count': 2, 'r1_gate': True, 'r100_gate': True, 'localization_positive_count': 4, 'movement_gate': False, 'boundary_retention_gate': True, 'best_iou_rank_gate': True, 'core_gate': False, 'strong_gate': False}`
### Deltas vs C4_final
- `0.5-r1`: `0.8811076782240548`
- `0.5-r10`: `-4.85181370866232`
- `0.5-r100`: `0.0`
- `0.5-r5`: `-6.694129763130796`
- `0.7-r1`: `1.3845977800663682`
- `0.7-r10`: `-1.4074836937864745`
- `0.7-r100`: `0.8467788076438865`
- `0.7-r5`: `-2.7119807758324725`

## c6a_r2_large
- Best epoch: `7`
- Best selection score: `-5.981350582606503`
- Gates: `{'front_positive_count': 2, 'r1_gate': True, 'r100_gate': True, 'localization_positive_count': 4, 'movement_gate': False, 'boundary_retention_gate': True, 'best_iou_rank_gate': True, 'core_gate': False, 'strong_gate': False}`
### Deltas vs C4_final
- `0.5-r1`: `0.37761757638173776`
- `0.5-r10`: `-4.153793340199115`
- `0.5-r100`: `-0.0343288705801541`
- `0.5-r5`: `-5.80157912804669`
- `0.7-r1`: `1.270168211465844`
- `0.7-r10`: `-1.1900675134454701`
- `0.7-r100`: `0.7781210664835783`
- `0.7-r5`: `-2.128389975969789`

## c6a_r2_safe
- Best epoch: `3`
- Best selection score: `-0.6180705429905908`
- Gates: `{'front_positive_count': 0, 'r1_gate': False, 'r100_gate': False, 'localization_positive_count': 3, 'movement_gate': True, 'boundary_retention_gate': True, 'best_iou_rank_gate': True, 'core_gate': False, 'strong_gate': False}`
### Deltas vs C4_final
- `0.5-r1`: `-0.011442956860051368`
- `0.5-r10`: `-1.0298661174047368`
- `0.5-r100`: `-0.05721478430025684`
- `0.5-r5`: `-1.006980203684634`
- `0.7-r1`: `-0.08010069802037023`
- `0.7-r10`: `-0.17164435290078472`
- `0.7-r100`: `0.22885913720104156`
- `0.7-r5`: `-0.30895983522142245`
