# C5-main-inference-A2A negative audit

- Status: `NO_PROMOTION`
- Mode: `video_neutral`
- Selected: `c5ma_0000`
- Best nonzero: `c5ma_0273`
- Official val used: `false`

## Best nonzero deltas vs C4_final
- `0.5-r1`: `-0.10298661174047652`
- `0.5-r5`: `-0.06865774116031531`
- `0.5-r10`: `0.0343288705801541`
- `0.7-r1`: `-0.011442956860051368`
- `0.7-r5`: `-0.0343288705801541`
- `0.7-r10`: `-0.16020139604073336`
- `0.5-r100`: `0.034328870580168314`
- `0.7-r100`: `-0.011442956860065578`

## Localization
- oracle R1@0.5 delta: `0.05679397983813317`
- oracle R1@0.7 delta: `0.3265653840692835`
- selected mIoU delta: `0.0007545251717169243`
- best-IoU mean-rank delta: `-0.014908419707511004`

## Safety
- hard-positive exit ratio: `0.0037769526207588966`
- anchor drift max abs: `0.27535057067871094`
- video top-10 membership changed ratio: `0.0`

Execution stopped before official val.
