# C7-B3 split overlap audit

```json
{
  "status": "PASS",
  "official_val_used": false,
  "train_core_calib_A_calib_B_calib_C_mutually_exclusive": true,
  "stress_split_diagnostic_only": true,
  "train_calib_final_review_relation": "superset/full train_calib review split",
  "split_hashes": {
    "train_core": "f2ca830763802b974113afc29efea42a31ba7f619137978c0bb0c51acaaeca7f",
    "calib_A": "e29ae5beaa7352a610c381693dd814ead3058443f7b58048f045e5ff3c60ff8a",
    "calib_B": "3497de880b934b51cb70d5a183fc5fb5f283d51899264da9cea3aea3bb29cb3d",
    "calib_C": "24ff013688413f4533bc9bc50d766d524b3847d069d2c8191fd860ec01b1d193",
    "stress_split": "1726a24e91f2536e43ae3da5c17ffe30e643ae97fcc7210bf5ebe75b085e352d",
    "train_calib_final_review": "1ba9829704c299bcd5e24917a03709d24f124106d9aee4edfbddd625fda3efb5"
  },
  "archived_split_hashes": {
    "train_core": "f2ca830763802b974113afc29efea42a31ba7f619137978c0bb0c51acaaeca7f",
    "calib_A": "e29ae5beaa7352a610c381693dd814ead3058443f7b58048f045e5ff3c60ff8a",
    "calib_B": "3497de880b934b51cb70d5a183fc5fb5f283d51899264da9cea3aea3bb29cb3d",
    "calib_C": "24ff013688413f4533bc9bc50d766d524b3847d069d2c8191fd860ec01b1d193",
    "stress_split": "1726a24e91f2536e43ae3da5c17ffe30e643ae97fcc7210bf5ebe75b085e352d",
    "train_calib_final_review": "1ba9829704c299bcd5e24917a03709d24f124106d9aee4edfbddd625fda3efb5"
  },
  "split_hash_match": {
    "train_core": true,
    "calib_A": true,
    "calib_B": true,
    "calib_C": true,
    "stress_split": true,
    "train_calib_final_review": true
  },
  "overlap_matrix_note": "train_core/calib_A/calib_B/calib_C are mutually exclusive sha256(desc_id) buckets; stress_split is diagnostic-only and may overlap; train_calib_final_review is the full train_calib set.",
  "overlap_matrix": {
    "train_core": {
      "train_core": 4268,
      "calib_A": 0,
      "calib_B": 0,
      "calib_C": 0,
      "stress_split": 876,
      "train_calib_final_review": 4268
    },
    "calib_A": {
      "train_core": 0,
      "calib_A": 1411,
      "calib_B": 0,
      "calib_C": 0,
      "stress_split": 279,
      "train_calib_final_review": 1411
    },
    "calib_B": {
      "train_core": 0,
      "calib_A": 0,
      "calib_B": 1344,
      "calib_C": 0,
      "stress_split": 255,
      "train_calib_final_review": 1344
    },
    "calib_C": {
      "train_core": 0,
      "calib_A": 0,
      "calib_B": 0,
      "calib_C": 1246,
      "stress_split": 244,
      "train_calib_final_review": 1246
    },
    "stress_split": {
      "train_core": 876,
      "calib_A": 279,
      "calib_B": 255,
      "calib_C": 244,
      "stress_split": 1748,
      "train_calib_final_review": 1748
    },
    "train_calib_final_review": {
      "train_core": 4268,
      "calib_A": 1411,
      "calib_B": 1344,
      "calib_C": 1246,
      "stress_split": 1748,
      "train_calib_final_review": 8739
    }
  }
}
```
