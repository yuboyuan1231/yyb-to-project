# C12-1 Split Audit

Status: `C12_SPLIT_GATE_PASS`

This split is generated only from train data. It groups connected components of
duplicate query text and video/show group, then deterministically assigns the
components into `train_fit`, `calib_select`, `calib_holdout`, and
`pseudo_official_holdout`.

```json
{
  "stage": "C12-1",
  "split_version": "c12_connected_text_video_group_split_v1",
  "official_val_used": false,
  "pseudo_official_holdout_not_used_for_selection": true,
  "split_overlap_query": 0,
  "duplicate_text_overlap": 0,
  "video_group_overlap": 0,
  "schema_audit_pass": true,
  "gate_pass": true,
  "split_salt": "C12_SPLIT_CONNECTED_TEXT_VIDEO_GROUP_V1",
  "component_count": 719,
  "query_overlap": 0,
  "counts": {
    "train_fit": 69428,
    "calib_select": 8677,
    "calib_holdout": 4340,
    "pseudo_official_holdout": 4339
  },
  "target_counts": {
    "train_fit": 69427,
    "calib_select": 8678,
    "calib_holdout": 4339,
    "pseudo_official_holdout": 4339
  },
  "split_files": {
    "train_fit": "c12_1_schema_and_split/splits/train_fit_desc_ids.txt",
    "calib_select": "c12_1_schema_and_split/splits/calib_select_desc_ids.txt",
    "calib_holdout": "c12_1_schema_and_split/splits/calib_holdout_desc_ids.txt",
    "pseudo_official_holdout": "c12_1_schema_and_split/splits/pseudo_official_holdout_desc_ids.txt"
  }
}
```

`pseudo_official_holdout` is reserved for one-shot train-only branch validation
and must not be used for model/config/feature/safety selection.
