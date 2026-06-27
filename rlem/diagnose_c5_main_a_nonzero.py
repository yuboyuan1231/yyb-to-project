#!/usr/bin/env python
"""Materialize localization diagnostics for the best rejected nonzero A config."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))

from rlem.c5_main_a_utils import (EndpointPriorEngine, exact_candidate_indices, group_max,
    load_c4_final_scores, load_npz_arrays, retrieval_gate, retrieval_z, setting_id,
    standardized_delta)
from rlem.c5_prior_utils import attach_eval_labels, load_cache, localization_diagnostics, movement_diagnostics, write_json
from rlem.grid_search_c5_main_a import _extra_rank_diagnostics


def main():
    p=argparse.ArgumentParser()
    for name in ("cache_npz","c4_final_scores_npz","temporal_prior_npz","stats_json","gt_jsonl","grid_json","output_json"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    a=p.parse_args()
    if Path(a.output_json).exists(): raise FileExistsError(a.output_json)
    grid=json.loads(Path(a.grid_json).read_text())['records']
    nonzero=[r for r in grid if r['family']!='zero_control']
    rec=max(nonzero,key=lambda r:r['selection_score_delta_vs_c4_final']); cfg={k:rec[k] for k in ('config_id','family','alpha','beta','tau','lambda','gamma')}
    stats=json.loads(Path(a.stats_json).read_text()); cache=load_cache(a.cache_npz); attach_eval_labels(cache,a.gt_jsonl)
    gid=cache['row_group_id'].astype(np.int64); base=load_c4_final_scores(a.c4_final_scores_npz,len(gid))
    tl=load_npz_arrays(a.temporal_prior_npz,('temporal_length',))['temporal_length'].astype(np.int64)
    si,ei,index_audit=exact_candidate_indices(cache['start_time'],cache['end_time'],gid,tl)
    gs=group_max(base,gid,len(tl)); gate=retrieval_gate(retrieval_z(gs,stats['retrieval_score_stats']),cfg['tau'])
    eng=EndpointPriorEngine.load(a.temporal_prior_npz,a.device); ls,le,_=eng.role_log_priors(cfg['alpha'],cfg['beta'])
    delta=eng.injected_endpoint_delta(ls,le,np.float32(cfg['lambda'])*gate,gid,si,ei,chunk_rows=500000)
    key=setting_id(cfg['alpha'],cfg['beta'],cfg['tau'],cfg['lambda']); z=standardized_delta(delta,stats['delta_stats'][key])
    score=(base+np.float32(cfg['gamma'])*z).astype(np.float32)
    loc=localization_diagnostics(cache,base,score); loc.update(_extra_rank_diagnostics(cache,base,score))
    out={"status":"PASS","stage":"C5-main-A best rejected nonzero diagnostic","scope":"train_calib_only",
         "official_val_used":False,"post_val_adjustment":False,"promoted":False,"config":cfg,
         "selection_score_delta_vs_c4_final":rec['selection_score_delta_vs_c4_final'],
         "metrics":{k[7:]:v for k,v in rec.items() if k.startswith('metric_')},
         "deltas_vs_c4_final":{k[6:]:v for k,v in rec.items() if k.startswith('delta_')},
         "movement":movement_diagnostics(cache,base,score),
         "localization":loc,"endpoint_index_audit":index_audit,
         "note":"Diagnostic only; this rejected config is not frozen or authorized for official val."}
    write_json(a.output_json,out); print(json.dumps({"status":"PASS","config":cfg['config_id'],"output":a.output_json},indent=2))


if __name__=="__main__": main()
