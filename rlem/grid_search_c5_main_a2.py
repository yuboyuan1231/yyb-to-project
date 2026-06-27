#!/usr/bin/env python
"""C5-main-inference-A2a/A2b train_fit stats + train_calib search.

A2a centers the endpoint residual within every query-video group.
A2b additionally re-anchors every updated group to the exact frozen C4_final
group maximum, so video-level anchors cannot drift.
"""

from __future__ import annotations

import argparse, csv, gc, json, math, multiprocessing as mp, os, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0,str(_ROOT))

from rlem.c4_lite_utils import PRIMARY_METRICS, selection_score  # noqa: E402
from rlem.c4_video_targets import frozen_c4_lite_row_scores  # noqa: E402
from rlem.c5_main_a_utils import (ALPHAS,BETAS,GAMMAS,LAMBDAS,TAUS,EndpointPriorEngine,
    artifact,exact_candidate_indices,fit_delta_stats,fit_retrieval_stats,group_max,
    iter_effective_configs,load_c4_final_scores,load_npz_arrays,retrieval_gate,retrieval_z,
    setting_id,standardized_delta)
from rlem.c5_prior_utils import (attach_eval_labels,fast_metrics,load_cache,localization_diagnostics,
    metric_deltas,movement_diagnostics,write_json)
from rlem.grid_search_c5_main_a import _adjacent,_boundary,_extra_rank_diagnostics,_movement_safe


def args_parse():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['video_neutral','video_anchor'],required=True)
    for n in ('train_fit_cache','train_calib_cache','train_fit_temporal_prior','train_calib_temporal_prior',
              'train_fit_c4_final_scores','train_calib_c4_final_scores','train_calib_gt','dataset_config','output_dir','audit_dir'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--workers',type=int,default=max(1,min(12,os.cpu_count() or 1)))
    p.add_argument('--chunk_rows',type=int,default=1_000_000)
    p.add_argument('--effective_top_n',type=int,default=100)
    p.add_argument('--max_after_nms',type=int,default=100)
    p.add_argument('--nms_thd',type=float,default=.7)
    return p.parse_args()


def center_by_group(values:np.ndarray,gid:np.ndarray,groups:int):
    sums=np.bincount(gid,weights=values.astype(np.float64),minlength=groups)
    counts=np.bincount(gid,minlength=groups)
    means=sums/np.maximum(counts,1)
    centered=(values.astype(np.float32)-means[gid].astype(np.float32)).astype(np.float32)
    check=np.bincount(gid,weights=centered.astype(np.float64),minlength=groups)/np.maximum(counts,1)
    return centered,{"group_mean_max_abs":float(np.max(np.abs(check))),"group_mean_rms":float(np.sqrt(np.mean(check*check))),"groups":int(groups)}


def mode_prefix(mode): return 'A2A' if mode=='video_neutral' else 'A2B'


def score_mode(base,z,gid,base_anchor,gamma,mode):
    raw=(base+np.float32(gamma)*z).astype(np.float32)
    if mode=='video_neutral': return raw
    new_anchor=group_max(raw,gid,len(base_anchor))
    return (raw-new_anchor[gid]+base_anchor[gid]).astype(np.float32)


def score_diagnostics(base,score,gid,groups,group_query):
    base_anchor=group_max(base,gid,groups); new_anchor=group_max(score,gid,groups)
    drift=new_anchor.astype(np.float64)-base_anchor.astype(np.float64)
    residual=score.astype(np.float64)-base.astype(np.float64)
    sums=np.bincount(gid,weights=residual,minlength=groups); counts=np.bincount(gid,minlength=groups)
    means=sums/np.maximum(counts,1)
    # Compare per-query top-10 video-anchor membership.
    changed=[]
    for q in np.unique(group_query):
        ix=np.flatnonzero(group_query==q); k=min(10,len(ix))
        a=set(ix[np.argsort(-base_anchor[ix],kind='stable')[:k]].tolist())
        b=set(ix[np.argsort(-new_anchor[ix],kind='stable')[:k]].tolist())
        changed.append(float(a!=b))
    return {
        "video_anchor_drift_max_abs":float(np.max(np.abs(drift))),
        "video_anchor_drift_mean_abs":float(np.mean(np.abs(drift))),
        "video_anchor_drift_nonzero_ratio":float(np.mean(np.abs(drift)>1e-6)),
        "group_residual_mean_max_abs":float(np.max(np.abs(means))),
        "group_residual_mean_rms":float(np.sqrt(np.mean(means*means))),
        "row_residual_std":float(np.std(residual)),
        "video_top10_membership_changed_query_ratio":float(np.mean(changed)),
    }


_C=_B=_Z=_BM=_GID=_ANCHOR=_MODE=None
def fast_rec(cfg):
    if cfg['family']=='zero_control': score=_B
    else:
        z=_Z[setting_id(cfg['alpha'],cfg['beta'],cfg['tau'],cfg['lambda'])]
        score=score_mode(_B,z,_GID,_ANCHOR,cfg['gamma'],_MODE)
    metrics,_,_=fast_metrics(_C,score,100,100,.7); delta=metric_deltas(metrics,_BM)
    move=movement_diagnostics(_C,_B,score)
    six=all(delta.get(k,-1e9)>=-1e-9 for k in PRIMARY_METRICS)
    r100=all(delta.get(k,-1e9)>=-1e-9 for k in ('0.5-r100','0.7-r100'))
    safe=_movement_safe(move)
    return {"config":cfg,"metrics":metrics,"deltas_vs_c4_final":delta,"movement":move,
            "six_primary_nonnegative":bool(six),"r100_both_nonnegative":bool(r100),
            "movement_safe":bool(safe),"vcmr_movement_gate":bool(six and r100 and safe),
            "selection_score_delta_vs_c4_final":float(selection_score(metrics,_BM))}


def main():
    a=args_parse(); mode=a.mode; prefix=mode_prefix(mode)
    if (a.effective_top_n,a.max_after_nms,a.nms_thd)!=(100,100,.7): raise ValueError('Frozen retrieval constants must remain 100/100/0.7')
    out,ad=Path(a.output_dir),Path(a.audit_dir); out.mkdir(parents=True,exist_ok=True); ad.mkdir(parents=True,exist_ok=True)
    stats_path=ad/f'C5_MAIN_{prefix}_STATS_FREEZE.json'
    paths={"grid_csv":out/'grid_results.csv',"grid_json":out/'grid_results.json',"best":out/'best_config.json',
           "loc":out/'localization_diagnostics.json',"move":out/'movement_diagnostics.json',
           "nonzero":out/'best_nonzero_diagnostics.json',"audit":out/'train_calib_search_audit.json'}
    if any(p.exists() for p in [stats_path,*paths.values()]): raise FileExistsError('A2 output exists')

    # ---- train_fit-only statistics ----
    fc=load_npz_arrays(a.train_fit_cache,('row_group_id','start_time','end_time','group_ids_sorted_unique'))
    fgid=fc['row_group_id'].astype(np.int64); fg=len(fc['group_ids_sorted_unique'])
    fbase=load_c4_final_scores(a.train_fit_c4_final_scores,len(fgid))
    ftl=load_npz_arrays(a.train_fit_temporal_prior,('temporal_length',))['temporal_length'].astype(np.int64)
    fsi,fei,fidx=exact_candidate_indices(fc['start_time'],fc['end_time'],fgid,ftl)
    frscore=group_max(fbase,fgid,fg); rstats=fit_retrieval_stats(frscore); frz=retrieval_z(frscore,rstats)
    fgates={t:retrieval_gate(frz,t) for t in TAUS}
    feng=EndpointPriorEngine.load(a.train_fit_temporal_prior,a.device)
    dstats={}; sanity={}; center_checks={}
    for alpha in ALPHAS:
      for beta in BETAS:
        ls,le,san=feng.role_log_priors(alpha,beta); sanity[f'a{alpha:.2f}_b{beta:.2f}']=san
        for tau in TAUS:
          for lam in LAMBDAS:
            key=setting_id(alpha,beta,tau,lam)
            raw=feng.injected_endpoint_delta(ls,le,np.float32(lam)*fgates[tau],fgid,fsi,fei,chunk_rows=a.chunk_rows)
            centered,chk=center_by_group(raw,fgid,fg); center_checks[key]=chk
            dstats[key]={"alpha":alpha,"beta":beta,"tau":tau,"lambda":lam,**fit_delta_stats(centered)}
            del raw,centered
        del ls,le
    stats={"status":"PASS","stage":f'C5-main-inference-{prefix} train_fit stats freeze',"mode":mode,
           "stats_source":"train_fit_only","official_val_used":False,"post_val_adjustment":False,
           "candidate_rows":int(len(fgid)),"video_groups":int(fg),"endpoint_index_audit":fidx,
           "retrieval_score_stats":rstats,"delta_center_definition":"Delta_raw - group_mean(Delta_raw)",
           "delta_stats":dstats,"group_center_sanity":center_checks,"prior_sanity":sanity,
           "A2b_anchor_formula":("S=V_C4 + L_raw - group_max(L_raw)" if mode=='video_anchor' else None),
           "artifacts":{"cache":artifact(a.train_fit_cache),"temporal":artifact(a.train_fit_temporal_prior),"scores":artifact(a.train_fit_c4_final_scores)},
           "runtime":{"device":str(feng.device),"cuda_visible_devices":os.environ.get('CUDA_VISIBLE_DEVICES'),"command_line":" ".join(sys.argv)},
           "retrieval_constants":{"effective_top_n":100,"nms_thd":.7,"max_after_nms":100}}
    write_json(str(stats_path),stats)
    del fc,fgid,fbase,ftl,fsi,fei,frscore,frz,fgates,feng,dstats,center_checks; gc.collect(); torch.cuda.empty_cache()

    # ---- train_calib-only selection ----
    cache=load_cache(a.train_calib_cache); attach_eval_labels(cache,a.train_calib_gt)
    gid=cache['row_group_id'].astype(np.int64); groups=len(cache['group_ids_sorted_unique'])
    base=load_c4_final_scores(a.train_calib_c4_final_scores,len(gid)); base_anchor=group_max(base,gid,groups)
    tl=load_npz_arrays(a.train_calib_temporal_prior,('temporal_length',))['temporal_length'].astype(np.int64)
    si,ei,idxaudit=exact_candidate_indices(cache['start_time'],cache['end_time'],gid,tl)
    rscore=group_max(base,gid,groups); rz=retrieval_z(rscore,rstats); gates={t:retrieval_gate(rz,t) for t in TAUS}
    eng=EndpointPriorEngine.load(a.train_calib_temporal_prior,a.device); zmap={}
    for alpha in ALPHAS:
      for beta in BETAS:
        ls,le,_=eng.role_log_priors(alpha,beta)
        for tau in TAUS:
          for lam in LAMBDAS:
            key=setting_id(alpha,beta,tau,lam)
            raw=eng.injected_endpoint_delta(ls,le,np.float32(lam)*gates[tau],gid,si,ei,chunk_rows=min(a.chunk_rows,500000))
            centered,_=center_by_group(raw,gid,groups); zmap[key]=standardized_delta(centered,stats['delta_stats'][key]); del raw,centered
        del ls,le
    bm,_,_=fast_metrics(cache,base,100,100,.7)
    c4l,_,_,_=frozen_c4_lite_row_scores(cache); c4lm,_,_=fast_metrics(cache,c4l,100,100,.7)
    c31m,_,_=fast_metrics(cache,cache['s_c31'].astype(np.float32),100,100,.7)
    global _C,_B,_Z,_BM,_GID,_ANCHOR,_MODE
    _C,_B,_Z,_BM,_GID,_ANCHOR,_MODE=cache,base,zmap,bm,gid,base_anchor,mode
    cfgs=list(iter_effective_configs())
    if a.workers<=1: records=[fast_rec(c) for c in cfgs]
    else:
      with mp.get_context('fork').Pool(a.workers) as pool: records=list(pool.imap_unordered(fast_rec,cfgs,chunksize=1))
    records.sort(key=lambda r:r['config']['config_id'])
    strict=[r for r in records if r['vcmr_movement_gate']]
    best_nonzero=max((r for r in records if r['config']['family']!='zero_control'),key=lambda r:r['selection_score_delta_vs_c4_final'])
    detailed=list(strict)
    if best_nonzero not in detailed: detailed.append(best_nonzero)
    for r in detailed:
      c=r['config']; score=base if c['family']=='zero_control' else score_mode(base,zmap[setting_id(c['alpha'],c['beta'],c['tau'],c['lambda'])],gid,base_anchor,c['gamma'],mode)
      loc=localization_diagnostics(cache,base,score); loc.update(_extra_rank_diagnostics(cache,base,score)); r['localization']=loc
      positives=sum([loc['oracle_video_r1_05_delta']>0,loc['oracle_video_r1_07_delta']>0,loc['selected_span_miou_delta']>0,loc['best_iou_span_rank_delta_mean']<0])
      r['localization_positive_count']=int(positives); r['localization_gate']=bool(positives>=2)
      r['core_promotion_gate']=bool(r['vcmr_movement_gate'] and r['localization_gate'])
      r['score_diagnostics']=score_diagnostics(base,score,gid,groups,cache['video_group_keys'][:,0].astype(np.int64))
    core=[r for r in detailed if r.get('core_promotion_gate') and r['config']['family']!='zero_control']
    for r in records:
      c=r['config']; neigh=[] if c['family']=='zero_control' else [x['config']['config_id'] for x in core if x is not r and _adjacent(c,x['config'])]
      r['passing_neighbor_ids']=neigh;r['passing_neighbor_count']=len(neigh);r['active_boundary']=bool(c['family']!='zero_control' and _boundary(c));r['isolated_boundary']=bool(r['active_boundary'] and not neigh)
      anchor_ok=True
      if mode=='video_anchor' and r.get('score_diagnostics'): anchor_ok=r['score_diagnostics']['video_anchor_drift_max_abs']<=1e-6
      r['promotion_candidate']=bool(r.get('core_promotion_gate',False) and not r['isolated_boundary'] and anchor_ok)
    promotable=[r for r in records if r['promotion_candidate']]
    if promotable: best=max(promotable,key=lambda r:(r['passing_neighbor_count'],r['selection_score_delta_vs_c4_final'])); status='PASS'
    else: best=max(strict,key=lambda r:r['selection_score_delta_vs_c4_final']) if strict else max(records,key=lambda r:r['selection_score_delta_vs_c4_final']); status='NO_PROMOTION'
    best['deltas_vs_c4_lite']=metric_deltas(best['metrics'],c4lm);best['deltas_vs_c31']=metric_deltas(best['metrics'],c31m)
    # Ensure selected and best nonzero both have detailed diagnostics.
    for r in (best,best_nonzero):
      if 'localization' not in r:
        c=r['config'];score=base if c['family']=='zero_control' else score_mode(base,zmap[setting_id(c['alpha'],c['beta'],c['tau'],c['lambda'])],gid,base_anchor,c['gamma'],mode)
        loc=localization_diagnostics(cache,base,score);loc.update(_extra_rank_diagnostics(cache,base,score));r['localization']=loc;r['score_diagnostics']=score_diagnostics(base,score,gid,groups,cache['video_group_keys'][:,0].astype(np.int64))
    rows=[]
    for r in records:
      c,mv=r['config'],r['movement']; row={**c,"selection_score_delta_vs_c4_final":r['selection_score_delta_vs_c4_final'],"six_primary_nonnegative":r['six_primary_nonnegative'],"r100_both_nonnegative":r['r100_both_nonnegative'],"movement_safe":r['movement_safe'],"localization_gate":r.get('localization_gate'),"promotion_candidate":r['promotion_candidate'],"passing_neighbor_count":r['passing_neighbor_count'],"isolated_boundary":r['isolated_boundary'],"top1_changed_ratio":mv['top1_changed_ratio'],"hard_positive_top100_exits":mv['hard_positive_top100_exits'],"hard_positive_top100_entries":mv['hard_positive_top100_entries'],"hard_positive_exit_ratio":mv['hard_positive_top100_exit_ratio'],"pearson":mv['pearson_base_candidate'],"mean_within_query_spearman":mv['mean_within_query_spearman'],**{f'delta_{k}':v for k,v in r['deltas_vs_c4_final'].items()},**{f'metric_{k}':v for k,v in r['metrics'].items()}}
      for k,v in (r.get('localization') or {}).items(): row['loc_'+k]=v
      rows.append(row)
    fields=[]
    for row in rows:
      for k in row:
        if k not in fields:fields.append(k)
    with open(paths['grid_csv'],'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write_json(str(paths['grid_json']),{"status":status,"mode":mode,"grid_size":len(records),"records":rows})
    write_json(str(paths['best']),best['config']);write_json(str(paths['loc']),best['localization']);write_json(str(paths['move']),best['movement'])
    write_json(str(paths['nonzero']),{"promoted":False,"config":best_nonzero['config'],"metrics":best_nonzero['metrics'],"deltas_vs_c4_final":best_nonzero['deltas_vs_c4_final'],"movement":best_nonzero['movement'],"localization":best_nonzero['localization'],"score_diagnostics":best_nonzero['score_diagnostics']})
    audit={"status":status,"stage":f'C5-main-inference-{prefix} train_calib search',"mode":mode,"scope":"train_fit/train_calib_only","official_val_used":False,"post_val_adjustment":False,"training_used":False,"primary_baseline":"C4_final = C4-r2-cal-v2.1 v21_00444","baseline_metrics":bm,"secondary_baselines":{"C4_lite":c4lm,"C3.1":c31m},"best_config":best['config'],"best_metrics":best['metrics'],"best_deltas_vs_c4_final":best['deltas_vs_c4_final'],"best_deltas_vs_c4_lite":best['deltas_vs_c4_lite'],"best_deltas_vs_c31":best['deltas_vs_c31'],"best_movement":best['movement'],"best_localization":best['localization'],"best_score_diagnostics":best['score_diagnostics'],"best_nonzero":{"config":best_nonzero['config'],"metrics":best_nonzero['metrics'],"deltas_vs_c4_final":best_nonzero['deltas_vs_c4_final'],"movement":best_nonzero['movement'],"localization":best_nonzero['localization'],"score_diagnostics":best_nonzero['score_diagnostics']},"neighbor_robustness":{"passing_neighbor_count":best['passing_neighbor_count'],"passing_neighbor_ids":best['passing_neighbor_ids'],"isolated_boundary":best['isolated_boundary']},"counts":{"six_primary_nonnegative":sum(r['six_primary_nonnegative'] for r in records),"both_r100_nonnegative":sum(r['r100_both_nonnegative'] for r in records),"movement_safe":sum(r['movement_safe'] for r in records),"core_promotion":sum(bool(r.get('core_promotion_gate')) for r in records),"promotion_candidate":sum(r['promotion_candidate'] for r in records)},"grid_size":len(records),"workers":a.workers,"stats_freeze":artifact(str(stats_path)),"cuda_visible_devices":os.environ.get('CUDA_VISIBLE_DEVICES'),"command_line":" ".join(sys.argv),"endpoint_index_audit":idxaudit,"retrieval_constants":{"effective_top_n":100,"nms_thd":.7,"max_after_nms":100},"used":{"C5_main_A2":True,"C5_main_B":False,"C6":False,"candidate_regeneration":False}}
    write_json(str(paths['audit']),audit)
    finalize(ad,prefix,mode,status,audit,paths,stats_path)
    print(json.dumps({"status":status,"mode":mode,"best":best['config']['config_id'],"audit":str(paths['audit'])},indent=2))


def finalize(ad,prefix,mode,status,audit,paths,stats_path):
    pass_status=status=='PASS'; stem='FREEZE' if pass_status else 'NEGATIVE'
    md=ad/('C5_MAIN_A2_FREEZE_REVIEW.md' if pass_status else 'C5_MAIN_A2_NEGATIVE_AUDIT.md')
    manifest=ad/f'C5_MAIN_A2_{stem}_MANIFEST.json'; hashes=ad/f'C5_MAIN_A2_{stem}_HASHES.json'
    b=audit['best_config']; bn=audit['best_nonzero']; loc=bn['localization']; mov=bn['movement']; diag=bn['score_diagnostics']
    m={"status":("C5_MAIN_A2_FREEZE_REVIEW_PASS" if pass_status else "NO_PROMOTION"),"mode":mode,"classification":f'C5-main-inference-{prefix} '+('positive' if pass_status else 'train_calib negative'),"selected_config":b['config_id'],"best_nonzero_config":bn['config']['config_id'],"official_val_used":False,"official_val_authorized":False,"post_val_adjustment":False,"training_used":False,"C5_main_B_used":False,"C6_used":False,"candidate_regeneration_used":False,"model_conquer_modified":False,"NMS_modified":False,"evaluator_modified":False,"promotion_candidate_count":audit['counts']['promotion_candidate'],"best_nonzero_deltas_vs_c4_final":bn['deltas_vs_c4_final'],"best_nonzero_localization":loc,"best_nonzero_movement":mov,"best_nonzero_score_diagnostics":diag}
    write_json(str(manifest),m)
    lines=[f'# C5-main-inference-{prefix} {"freeze" if pass_status else "negative"} audit\n',f'\n- Status: `{m["status"]}`\n- Mode: `{mode}`\n- Selected: `{b["config_id"]}`\n- Best nonzero: `{bn["config"]["config_id"]}`\n- Official val used: `false`\n', '\n## Best nonzero deltas vs C4_final\n']
    for k,v in bn['deltas_vs_c4_final'].items():lines.append(f'- `{k}`: `{v}`\n')
    lines+=['\n## Localization\n',f'- oracle R1@0.5 delta: `{loc["oracle_video_r1_05_delta"]}`\n',f'- oracle R1@0.7 delta: `{loc["oracle_video_r1_07_delta"]}`\n',f'- selected mIoU delta: `{loc["selected_span_miou_delta"]}`\n',f'- best-IoU mean-rank delta: `{loc["best_iou_span_rank_delta_mean"]}`\n','\n## Safety\n',f'- hard-positive exit ratio: `{mov["hard_positive_top100_exit_ratio"]}`\n',f'- anchor drift max abs: `{diag["video_anchor_drift_max_abs"]}`\n',f'- video top-10 membership changed ratio: `{diag["video_top10_membership_changed_query_ratio"]}`\n','\nExecution stopped before official val.\n']
    md.write_text(''.join(lines),encoding='utf-8')
    h={"status":"HASHES_MATERIALIZED","official_val_used":False,"artifacts":{}}
    for name,p in {"stats":stats_path,"grid_csv":paths['grid_csv'],"grid_json":paths['grid_json'],"best":paths['best'],"search_audit":paths['audit'],"localization":paths['loc'],"movement":paths['move'],"best_nonzero":paths['nonzero'],"review":md,"manifest":manifest,"code":Path(__file__)}.items():h['artifacts'][name]=artifact(str(p))
    write_json(str(hashes),h)


if __name__=='__main__':main()
