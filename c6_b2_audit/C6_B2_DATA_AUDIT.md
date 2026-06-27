# C6-B2 data audit

{
  "status": "PASS",
  "official_val_used": false,
  "source_train_pool": {
    "path": "results/rlem_c6_b1_lite_r1_safe/train_fit_candidate_pool_top_slots.npz",
    "exists": true,
    "size": 112176367,
    "sha256": "6167fc888fc53a1f9004588d225661f9b5487746bb00b1b594ffc06e2c226fab"
  },
  "source_calib_pool": {
    "path": "results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz",
    "exists": true,
    "size": 12624246,
    "sha256": "08029943dece218fa5c1cbf016fe3bb9b4330dd83a05d759e2dea2d85ecceb15"
  },
  "train_pair_cache": {
    "split": "train_fit",
    "queries": 78045,
    "slots": 5,
    "alts": 8,
    "pair_examples": 2731575,
    "feature_dim": 153,
    "gain07_positive_rate": 0.043866634368896484,
    "gain05_positive_rate": 0.03810109570622444,
    "risk07_rate": 0.1706535667181015,
    "risk05_rate": 0.15736928582191467,
    "replace_positive_rate": 0.11676816642284393,
    "elapsed_sec": 24.986939668655396,
    "source_pool": {
      "path": "results/rlem_c6_b1_lite_r1_safe/train_fit_candidate_pool_top_slots.npz",
      "exists": true,
      "size": 112176367,
      "sha256": "6167fc888fc53a1f9004588d225661f9b5487746bb00b1b594ffc06e2c226fab"
    },
    "x": {
      "path": "results/rlem_c6_b2/caches/train_fit_pair_x.npy",
      "exists": true,
      "size": 1671724028,
      "sha256": "db72a81b3b05487cdfeb2deb34d09761f1cbb4f07cd18d2bf7621ff621b1b7de"
    },
    "labels": {
      "path": "results/rlem_c6_b2/caches/train_fit_pair_labels.npz",
      "exists": true,
      "size": 4562095,
      "sha256": "2bf16925557a5718780ea86059b3c45fa05268c2a774cc7eec1d709bef04c9ad"
    },
    "official_val_used": false
  },
  "calib_pair_cache": {
    "split": "train_calib",
    "queries": 8739,
    "slots": 5,
    "alts": 8,
    "pair_examples": 305865,
    "feature_dim": 153,
    "gain07_positive_rate": 0.03854968771338463,
    "gain05_positive_rate": 0.03807888925075531,
    "risk07_rate": 0.1673189103603363,
    "risk05_rate": 0.1706308275461197,
    "replace_positive_rate": 0.11738511919975281,
    "elapsed_sec": 2.6321070194244385,
    "source_pool": {
      "path": "results/rlem_c6_b1_lite_r1_safe/train_calib_candidate_pool_top_slots.npz",
      "exists": true,
      "size": 12624246,
      "sha256": "08029943dece218fa5c1cbf016fe3bb9b4330dd83a05d759e2dea2d85ecceb15"
    },
    "x": {
      "path": "results/rlem_c6_b2/caches/train_calib_pair_x.npy",
      "exists": true,
      "size": 187189508,
      "sha256": "c419125c18776a9ba630d216d1619b5018998c94d1abaf3a6488613b4055fa5c"
    },
    "labels": {
      "path": "results/rlem_c6_b2/caches/train_calib_pair_labels.npz",
      "exists": true,
      "size": 506718,
      "sha256": "36f299116693b7c152a74cb107f3f8a11e586e678901ea2dcc739691e068d9d1"
    },
    "official_val_used": false
  },
  "listwise_source": "directly from C6-B1 pool reshape, no official val"
}
