#!/usr/bin/env python
"""C5-main-inference-A3 constrained candidate reranking.

A3a locks the exact C4_final pre-NMS top-100 row set.
A3b preserves the exact C4_final top-100 video slot sequence/quota while its
wide mode may replace spans with other fixed candidates from the same video.
"""
from __future__ import annotations
import argparse,csv,json,math,multiprocessing as mp,os,sys
from collections import Counter,defaultdict
from pathlib import Path
from typing import Dict
import numpy as np

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:sys.path.insert(0,str(_ROOT))
from rlem.c4_lite_utils import PRIMARY_METRICS,selection_score
from rlem.c4_video_targets import frozen_c4_lite_row_scores
from rlem.c5_main_a_utils import (ALPHAS,BETAS,GAMMAS,LAMBDAS,TAUS,EndpointPriorEngine,artifact,
 exact_candidate_indices,group_max,iter_effective_configs,load_c4_final_scores,load_npz_arrays,
 retrieval_gate,retrieval_z,setting_id,standardized_delta)
from rlem.c5_prior_utils import (attach_eval_labels,fast_metrics,load_cache,localization_diagnostics,
 metric_deltas,movement_diagnostics,write_json)
from rlem.grid_search_c5_main_a import _adjacent,_boundary,_extra_rank_diagnostics


def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['top100_locked','video_slot'],required=True)
 for n in ('train_calib_cache','train_calib_temporal_prior','train_calib_c4_final_scores','train_calib_gt','dataset_config','stats_json','output_dir','audit_dir'):p.add_argument('--'+n,required=True)
 p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--workers',type=int,default=max(1,min(12,os.cpu_count() or 1)))
 p.add_argument('--effective_top_n',type=int,default=100);p.add_argument('--max_after_nms',type=int,default=100);p.add_argument('--nms_thd',type=float,default=.7)
 return p.parse_args()


def center(values,gid,groups):
 sums=np.bincount(gid,weights=values.astype(np.float64),minlength=groups);cnt=np.bincount(gid,minlength=groups);return (values-sums[gid]/np.maximum(cnt[gid],1)).astype(np.float32)


def rank_score(order):
 inv=np.empty_like(order,dtype=np.int16);np.put_along_axis(inv,order,np.broadcast_to(np.arange(order.shape[1],dtype=np.int16),order.shape),axis=1)
 return (-inv.astype(np.float32)).reshape(-1)


def a3a_order(sloc):
 sm=sloc.reshape(_Q,200); top=_BASE_ORDER[:,:100]; vals=np.take_along_axis(sm,top,axis=1)
 within=np.argsort(-vals,axis=1,kind='stable');locked=np.take_along_axis(top,within,axis=1)
 return np.concatenate([locked,_BASE_ORDER[:,100:]],axis=1)


def a3b_order(sloc,wide=True):
 sm=sloc.reshape(_Q,200);out=np.empty_like(_BASE_ORDER)
 for q in range(_Q):
  top=_BASE_ORDER[q,:100].tolist();seq=[int(_VID[q,i]) for i in top];quota=Counter(seq);queues={}
  for v,m in quota.items():
   pool=np.flatnonzero(_VID[q]==v) if wide else np.asarray([i for i in top if int(_VID[q,i])==v],dtype=np.int64)
   queues[v]=pool[np.argsort(-sm[q,pool],kind='stable')[:m]].tolist()
  used=set();head=defaultdict(int);chosen=[]
  for v in seq:
   i=queues[v][head[v]];head[v]+=1;chosen.append(i);used.add(i)
  rest=[i for i in _BASE_ORDER[q].tolist() if i not in used]
  out[q]=np.asarray(chosen+rest,dtype=np.int64)
 return out


def constrained_order(sloc,mode,strict=False):return a3a_order(sloc) if mode=='top100_locked' else a3b_order(sloc,wide=not strict)


def order_movement(base,sloc,new_order,cache):
 bo=_BASE_ORDER;bi=np.empty_like(bo,dtype=np.int16);ni=np.empty_like(new_order,dtype=np.int16);r=np.broadcast_to(np.arange(200,dtype=np.int16),bo.shape)
 np.put_along_axis(bi,bo,r,axis=1);np.put_along_axis(ni,new_order,r,axis=1)
 hard=cache['eval_y05'].astype(bool);diff=bi.astype(np.float64)-ni.astype(np.float64);sp=1-6*np.sum(diff*diff,axis=1)/(200*(200*200-1))
 exits=int(np.sum(hard&(bi<100)&(ni>=100)));entries=int(np.sum(hard&(bi>=100)&(ni<100)))
 return {"queries":_Q,"top1_changed_ratio":float(np.mean(bo[:,0]!=new_order[:,0])),"hard_positive_top100_exits":exits,"hard_positive_top100_entries":entries,"hard_positive_top100_exit_ratio":float(exits/max(int(hard.sum()),1)),"pearson_base_localization_score":float(np.corrcoef(base.astype(np.float64),sloc.astype(np.float64))[0,1]),"mean_within_query_spearman":float(np.mean(sp))}


def structure(order,mode):
 base_top=_BASE_ORDER[:,:100];new_top=order[:,:100];mem=[];seq=[];multi=[];quota=0;dup=0
 for q in range(_Q):
  b=base_top[q].tolist();n=new_top[q].tolist();mem.append(set(b)!=set(n));dup+=100-len(set(n))
  bs=[int(_VID[q,i]) for i in b];ns=[int(_VID[q,i]) for i in n];seq.append(bs!=ns);multi.append(Counter(bs)!=Counter(ns));quota+=sum((Counter(bs)-Counter(ns)).values())+sum((Counter(ns)-Counter(bs)).values())
 return {"top100_membership_drift_query_ratio":float(np.mean(mem)),"top100_membership_symmetric_difference_rows":int(sum(len(set(_BASE_ORDER[q,:100])^set(order[q,:100])) for q in range(_Q))),"video_multiset_drift_query_ratio":float(np.mean(multi)),"video_slot_sequence_drift_query_ratio":float(np.mean(seq)),"quota_violation_count":int(quota),"duplicate_span_count":int(dup),"constraint_satisfied":bool((not any(mem)) if mode=='top100_locked' else (not any(multi) and quota==0 and dup==0 and not any(seq)))}


_CACHE=_BASE=_Z=_BM=_MODE=_Q=_BASE_ORDER=_VID=None
def eval_cfg(cfg):
 sloc=_BASE if cfg['family']=='zero_control' else (_BASE+np.float32(cfg['gamma'])*_Z[setting_id(cfg['alpha'],cfg['beta'],cfg['tau'],cfg['lambda'])]).astype(np.float32)
 order=_BASE_ORDER if cfg['family']=='zero_control' else constrained_order(sloc,_MODE)
 score=_BASE if cfg['family']=='zero_control' else rank_score(order)
 metrics,_,_=fast_metrics(_CACHE,score,100,100,.7);delta=metric_deltas(metrics,_BM);move=order_movement(_BASE,sloc,order,_CACHE)
 six=all(delta.get(k,-1e9)>=-1e-9 for k in PRIMARY_METRICS);r100=all(delta.get(k,-1e9)>=-1e-9 for k in ('0.5-r100','0.7-r100'))
 safe=move['top1_changed_ratio']<=.08 and move['mean_within_query_spearman']>=.985 and move['hard_positive_top100_exit_ratio']<=.005
 st=structure(order,_MODE);return {"config":cfg,"metrics":metrics,"deltas":delta,"movement":move,"structure":st,"six_primary_nonnegative":bool(six),"r100_both_nonnegative":bool(r100),"movement_safe":bool(safe),"vcmr_gate":bool(six and r100 and safe and st['constraint_satisfied']),"selection_delta":float(selection_score(metrics,_BM))}


def score_and_order(rec,strict=False):
 c=rec['config'];sloc=_BASE if c['family']=='zero_control' else (_BASE+np.float32(c['gamma'])*_Z[setting_id(c['alpha'],c['beta'],c['tau'],c['lambda'])]).astype(np.float32)
 order=_BASE_ORDER if c['family']=='zero_control' else constrained_order(sloc,_MODE,strict=strict);score=_BASE if c['family']=='zero_control' else rank_score(order);return sloc,order,score


def detail(rec,strict=False):
 sloc,order,score=score_and_order(rec,strict);loc=localization_diagnostics(_CACHE,_BASE,score);loc.update(_extra_rank_diagnostics(_CACHE,_BASE,score));pos=sum([loc['oracle_video_r1_05_delta']>0,loc['oracle_video_r1_07_delta']>0,loc['selected_span_miou_delta']>0,loc['best_iou_span_rank_delta_mean']<0]);return {"metrics":fast_metrics(_CACHE,score,100,100,.7)[0],"movement":order_movement(_BASE,sloc,order,_CACHE),"structure":structure(order,_MODE),"localization":loc,"localization_positive_count":int(pos),"localization_gate":bool(pos>=2)}


def detail_rec_worker(rec):
 d=detail(rec);rec.update({"localization":d['localization'],"localization_positive_count":d['localization_positive_count'],"localization_gate":d['localization_gate']});rec['core_gate']=bool(rec['vcmr_gate'] and rec['localization_gate']);return rec


def main():
 a=parse_args();mode=a.mode
 if (a.effective_top_n,a.max_after_nms,a.nms_thd)!=(100,100,.7):raise ValueError('Frozen constants must remain 100/100/0.7')
 out,ad=Path(a.output_dir),Path(a.audit_dir);out.mkdir(parents=True,exist_ok=True);ad.mkdir(parents=True,exist_ok=True)
 paths={k:out/v for k,v in {'grid_csv':'grid_results.csv','grid_json':'grid_results.json','best':'best_config.json','loc':'localization_diagnostics.json','move':'movement_diagnostics.json','nonzero':'best_nonzero_diagnostics.json','strict':'strict_diagnostics.json','audit':'train_calib_search_audit.json'}.items()}
 if any(p.exists() for p in paths.values()):raise FileExistsError('A3 outputs exist')
 stats=json.load(open(a.stats_json));assert stats['stats_source']=='train_fit_only' and stats['official_val_used'] is False
 cache=load_cache(a.train_calib_cache);attach_eval_labels(cache,a.train_calib_gt);gid=cache['row_group_id'].astype(np.int64);groups=len(cache['group_ids_sorted_unique']);base=load_c4_final_scores(a.train_calib_c4_final_scores,len(gid))
 tl=load_npz_arrays(a.train_calib_temporal_prior,('temporal_length',))['temporal_length'].astype(np.int64);si,ei,idxaudit=exact_candidate_indices(cache['start_time'],cache['end_time'],gid,tl)
 gs=group_max(base,gid,groups);rz=retrieval_z(gs,stats['retrieval_score_stats']);gates={t:retrieval_gate(rz,t) for t in TAUS};eng=EndpointPriorEngine.load(a.train_calib_temporal_prior,a.device);zmap={}
 for alpha in ALPHAS:
  for beta in BETAS:
   ls,le,_=eng.role_log_priors(alpha,beta)
   for tau in TAUS:
    for lam in LAMBDAS:
     key=setting_id(alpha,beta,tau,lam);raw=eng.injected_endpoint_delta(ls,le,np.float32(lam)*gates[tau],gid,si,ei,chunk_rows=500000);zmap[key]=standardized_delta(center(raw,gid,groups),stats['delta_stats'][key]);del raw
   del ls,le
 bm,_,_=fast_metrics(cache,base,100,100,.7);c4l,_,_,_=frozen_c4_lite_row_scores(cache);c4lm=fast_metrics(cache,c4l,100,100,.7)[0];c31m=fast_metrics(cache,cache['s_c31'].astype(np.float32),100,100,.7)[0]
 global _CACHE,_BASE,_Z,_BM,_MODE,_Q,_BASE_ORDER,_VID
 _CACHE,_BASE,_Z,_BM,_MODE=cache,base,zmap,bm,mode;_Q=len(cache['desc_ids']);_BASE_ORDER=np.argsort(-base.reshape(_Q,200),axis=1,kind='stable');_VID=cache['video_idx'].reshape(_Q,200)
 cfgs=list(iter_effective_configs())
 with mp.get_context('fork').Pool(a.workers) as pool:records=list(pool.imap_unordered(eval_cfg,cfgs,chunksize=1))
 records.sort(key=lambda r:r['config']['config_id']);strict_ok=[r for r in records if r['vcmr_gate']];bn=max((r for r in records if r['config']['family']!='zero_control'),key=lambda r:r['selection_delta'])
 targets=list(strict_ok)
 if bn not in targets:targets.append(bn)
 if len(targets)>1 and a.workers>1:
  with mp.get_context('fork').Pool(a.workers) as pool:detailed=list(pool.imap_unordered(detail_rec_worker,targets,chunksize=1))
 else:detailed=[detail_rec_worker(r) for r in targets]
 detailed_by_id={r['config']['config_id']:r for r in detailed}
 records=[detailed_by_id.get(r['config']['config_id'],r) for r in records]
 strict_ok=[r for r in records if r['vcmr_gate']];bn=max((r for r in records if r['config']['family']!='zero_control'),key=lambda r:r['selection_delta'])
 core=[r for r in detailed if r.get('core_gate') and r['config']['family']!='zero_control']
 for r in records:
  c=r['config'];neigh=[] if c['family']=='zero_control' else [x['config']['config_id'] for x in core if x is not r and _adjacent(c,x['config'])];r['passing_neighbors']=neigh;r['isolated_boundary']=bool(c['family']!='zero_control' and _boundary(c) and not neigh);r['promotion_candidate']=bool(r.get('core_gate',False) and not r['isolated_boundary'])
 promo=[r for r in records if r['promotion_candidate']]
 if promo:best=max(promo,key=lambda r:(len(r['passing_neighbors']),r['selection_delta']));status='PASS'
 else:best=max(strict_ok,key=lambda r:r['selection_delta']) if strict_ok else max(records,key=lambda r:r['selection_delta']);status='NO_PROMOTION'
 for r in (best,bn):
  if 'localization' not in r:
   d=detail(r);r['localization']=d['localization'];r['localization_positive_count']=d['localization_positive_count'];r['localization_gate']=d['localization_gate']
 bnd=detail(bn);strictdiag=detail(bn,strict=True) if mode=='video_slot' else bnd
 rows=[]
 for r in records:
  row={**r['config'],"selection_score_delta_vs_c4_final":r['selection_delta'],"six_primary_nonnegative":r['six_primary_nonnegative'],"r100_both_nonnegative":r['r100_both_nonnegative'],"movement_safe":r['movement_safe'],"structure_satisfied":r['structure']['constraint_satisfied'],"promotion_candidate":r['promotion_candidate'],"passing_neighbor_count":len(r['passing_neighbors']),"isolated_boundary":r['isolated_boundary'],**r['movement'],**{f'delta_{k}':v for k,v in r['deltas'].items()},**{f'metric_{k}':v for k,v in r['metrics'].items()}}
  for k,v in (r.get('localization') or {}).items():row['loc_'+k]=v
  rows.append(row)
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with open(paths['grid_csv'],'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 write_json(str(paths['grid_json']),{"status":status,"mode":mode,"grid_size":len(rows),"records":rows});write_json(str(paths['best']),best['config']);write_json(str(paths['loc']),best['localization']);write_json(str(paths['move']),best['movement']);write_json(str(paths['nonzero']),{"config":bn['config'],**bnd});write_json(str(paths['strict']),{"config":bn['config'],**strictdiag})
 refs={}
 for n,p in [('A','results/rlem_c5_main_inference_a/best_nonzero_diagnostics.json'),('A2a','results/rlem_c5_main_a2_video_neutral/best_nonzero_diagnostics.json'),('A2b','results/rlem_c5_main_a2_video_anchor/best_nonzero_diagnostics.json')]:
  if Path(p).exists():refs[n]=json.load(open(p))
 best_d4=best['deltas'];audit={"status":status,"stage":f'C5-main-inference-A3 {mode}',"mode":mode,"scope":"train_calib_only_with_trainfit_frozen_stats","official_val_used":False,"post_val_adjustment":False,"training_used":False,"primary_baseline":"C4_final = C4-r2-cal-v2.1 v21_00444","baseline_metrics":bm,"secondary_baselines":{"C4_lite":c4lm,"C3.1":c31m},"prior_stage_references":refs,"best_config":best['config'],"best_metrics":best['metrics'],"best_deltas_vs_c4_final":best_d4,"best_deltas_vs_c4_lite":metric_deltas(best['metrics'],c4lm),"best_deltas_vs_c31":metric_deltas(best['metrics'],c31m),"best_movement":best['movement'],"best_localization":best['localization'],"best_structure":best['structure'],"best_nonzero":{"config":bn['config'],**bnd},"strict_diagnostic":{"config":bn['config'],**strictdiag},"counts":{"six_primary_nonnegative":sum(r['six_primary_nonnegative'] for r in records),"both_r100_nonnegative":sum(r['r100_both_nonnegative'] for r in records),"structure_satisfied":sum(r['structure']['constraint_satisfied'] for r in records),"core_promotion":sum(bool(r.get('core_gate')) for r in records),"promotion_candidate":sum(r['promotion_candidate'] for r in records)},"neighbor_robustness":{"passing_neighbor_count":len(best['passing_neighbors']),"passing_neighbor_ids":best['passing_neighbors'],"isolated_boundary":best['isolated_boundary']},"grid_size":len(records),"stats_reused":artifact(a.stats_json),"command_line":" ".join(sys.argv),"cuda_visible_devices":os.environ.get('CUDA_VISIBLE_DEVICES'),"endpoint_index_audit":idxaudit,"retrieval_constants":{"effective_top_n":100,"nms_thd":.7,"max_after_nms":100},"used":{"C5_main_A3":True,"C5_main_B":False,"candidate_regeneration":False,"C6":False}}
 write_json(str(paths['audit']),audit);finalize(ad,status,mode,audit,paths);print(json.dumps({"status":status,"mode":mode,"best":best['config']['config_id']},indent=2))


def finalize(ad,status,mode,audit,paths):
 passed=status=='PASS';stem='FREEZE' if passed else 'NEGATIVE';md=ad/('C5_MAIN_A3_FREEZE_REVIEW.md' if passed else 'C5_MAIN_A3_NEGATIVE_AUDIT.md');man=ad/f'C5_MAIN_A3_{stem}_MANIFEST.json';hs=ad/f'C5_MAIN_A3_{stem}_HASHES.json';bn=audit['best_nonzero']
 m={"status":("C5_MAIN_A3_FREEZE_REVIEW_PASS" if passed else "NO_PROMOTION"),"mode":mode,"selected_config":audit['best_config'],"promotion_candidate_count":audit['counts']['promotion_candidate'],"official_val_used":False,"official_val_authorized":False,"post_val_adjustment":False,"training_used":False,"C5_main_B_used":False,"candidate_regeneration_used":False,"C6_used":False,"selected_metrics":audit['best_metrics'],"selected_deltas_vs_c4_final":audit['best_deltas_vs_c4_final'],"selected_movement":audit['best_movement'],"selected_localization":audit['best_localization'],"selected_structure":audit['best_structure'],"selected_neighbor_robustness":audit['neighbor_robustness'],"highest_selection_nonzero_config":bn['config']};write_json(str(man),m)
 md.write_text(f'# C5-main-inference-A3 {mode} {"freeze" if passed else "negative"} audit\n\n- Status: `{m["status"]}`\n- Selected config: `{audit["best_config"]["config_id"]}`\n- Promotion candidates: `{audit["counts"]["promotion_candidate"]}`\n- Passing one-step neighbors: `{audit["neighbor_robustness"]["passing_neighbor_count"]}`\n- Official val used/authorized: `false / false`\n\n## Selected deltas vs C4_final\n' + ''.join(f'- `{k}`: `{v}`\n' for k,v in audit['best_deltas_vs_c4_final'].items()) + '\n## Structure\n' + ''.join(f'- `{k}`: `{v}`\n' for k,v in audit['best_structure'].items()) + f'\nSee `{paths["audit"]}` for complete movement/localization and strict/wide diagnostics.\n',encoding='utf-8')
 h={"status":"HASHES_MATERIALIZED","official_val_used":False,"artifacts":{}}
 for n,p in {**paths,"review":md,"manifest":man,"code":Path(__file__)}.items():h['artifacts'][n]=artifact(str(p))
 write_json(str(hs),h)

if __name__=='__main__':main()
