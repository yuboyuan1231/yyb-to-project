# C7 CONQUER Source Audit

- Scope: `source read only`
- Training started: `false`
- Official val used: `false`
- Evaluator/NMS modified: `false`

## Recommendation

C7-B should be implemented as an integrated wrapper/subclass that freezes CONQUER and attaches RLEM heads to exported intermediates from `get_pred_from_raw_query(..., return_intermediates=True)`. The score should enter before VCMR flatten/sort and before unchanged NMS/top100.

## Module Map

### data_loading_entry

- File: `data_loader/second_stage_start_end_dataset.py`
- Class/function: `StartEndDataset.__getitem__`
- Lines: `226-363`
- Input shape: `query feat (Lq<=30,768); visual feat (Nv,Lv<=100,4352); sub feat optional (Nv,Lv<=100,768)`
- Output shape: `model_inputs dict; train Nv=1+neg_video_num, eval Nv=1+max_vcmr_video`
- Training usage: samples one GT video plus hard negatives from first-stage VR ranklist; emits st_ed_indices
- Inference usage: uses first-stage VR top-k; stores inference_vr_scores and sample_vid_name_list
- Safe to modify: `False`
- Frozen in C7-B: `True`
- C7 note: Do not modify data loader in C7-B; add optional sidecar tensors only after an audit if needed.

### train_entry

- File: `train.py`
- Class/function: `start_training/train/train_epoch`
- Lines: `38-365`
- Input shape: `batched model_inputs from start_end_collate`
- Output shape: `loss scalar plus loss_dict`
- Training usage: calls model(model_inputs), optimizer, eval_epoch
- Inference usage: not used except train-time validation
- Safe to modify: `True`
- Frozen in C7-B: `False`
- C7 note: C7-B can add a separate train script/wrapper; avoid changing original train.py until freeze review.

### config_args

- File: `config/config.py; config/model_config.json`
- Class/function: `BaseOptions/TestOptions; model_config`
- Lines: `config.py 37-206; model_config.json 1-68`
- Input shape: `N/A`
- Output shape: `opt namespace; CONQUER config object`
- Training usage: sets ctx_mode, max_vcmr_video, losses, similarity_measure, lr schedule
- Inference usage: loads saved opt then allows eval overrides
- Safe to modify: `True`
- Frozen in C7-B: `False`
- C7 note: Prefer C7-specific config file/args rather than changing defaults.

### conquer_model_class

- File: `model/conquer.py`
- Class/function: `CONQUER`
- Lines: `13-226`
- Input shape: `batch with query/video/sub tensors`
- Output shape: `video_similarity_score (B,Nv) or None; begin/end logits (B,Nv,Lv)`
- Training usage: computes moment CE and optional video CE
- Inference usage: get_pred_from_raw_query provides logits used by VCMR/SVMR/VR
- Safe to modify: `C7 wrapper preferred; direct edit only after design approval`
- Frozen in C7-B: `True`
- C7 note: Existing return_intermediates path exposes qdf/qal/contextual/G/video_mask needed by new heads.

### qdf_module

- File: `model/conquer.py; model/backbone/encoder.py`
- Class/function: `CONQUER.compute_final_score; QueryWeightEncoder`
- Lines: `conquer.py 92-107; encoder.py 196-231`
- Input shape: `per-modality video features dict each (B*Nv,Lv,768); moe weights (B*Nv)`
- Output shape: `QDF_feature (B*Nv,Lv,768)`
- Training usage: query dependent fusion of visual/sub modalities
- Inference usage: same path
- Safe to modify: `False`
- Frozen in C7-B: `True`
- C7 note: C7-B should consume QDF/QAL outputs, not alter QDF fusion.

### qal_module

- File: `model/qal/query_aware_learning_module.py`
- Class/function: `BiDirectionalAttention.forward`
- Lines: `16-94`
- Input shape: `QDF_emb (B*Nv,Lv,768), query_emb (B*Nv,Lq,768), masks`
- Output shape: `QAL (B*Nv,Lv,3072); optional q2v attention (B*Nv,Lv)`
- Training usage: core query-aware feature learning before localization/video heads
- Inference usage: same path
- Safe to modify: `False`
- Frozen in C7-B: `True`
- C7 note: Late QAL unfreeze belongs only to C7-C if C7-B passes.

### shared_contextual_qal

- File: `model/conquer.py; model/layers.py`
- Class/function: `FCPlusTransformer`
- Lines: `conquer.py 53-54,148-153; layers.py 88-115`
- Input shape: `QAL_feature (B*Nv,Lv,3072)`
- Output shape: `Contextual_QAL (B*Nv,Lv,768); G concat (B*Nv,Lv,3840)`
- Training usage: shared contextual layer for ML and optional VS heads
- Inference usage: same path
- Safe to modify: `False`
- Frozen in C7-B: `True`
- C7 note: C7 heads should attach after G/Contextual_QAL.

### moment_localization_head

- File: `model/head/ml_head.py`
- Class/function: `MomentLocalizationHead.forward`
- Lines: `10-59`
- Input shape: `G (B*Nv,Lv,3840), Contextual_QAL (B*Nv,Lv,768), video_mask (B*Nv,Lv)`
- Output shape: `begin logits (B*Nv,Lv), end logits (B*Nv,Lv), reshaped to (B,Nv,Lv)`
- Training usage: moment CE via shared-normalized flattened start/end labels
- Inference usage: softmaxed start/end probabilities produce SVMR/VCMR tuples
- Safe to modify: `additive heads only in C7-B`
- Frozen in C7-B: `True`
- C7 note: Proposal confidence can reuse begin/end features/logits without changing this head.

### video_retrieval_head

- File: `model/head/vs_head.py`
- Class/function: `VideoScoringHead.forward`
- Lines: `9-41`
- Input shape: `G (B*Nv,Lv,3840), video_mask (B*Nv,Lv)`
- Output shape: `video_similarity_score (B*Nv,1), reshaped to (B,Nv)`
- Training usage: only when similarity_measure='exclusive'
- Inference usage: if internal VR scores are enabled; otherwise external first-stage scores are used
- Safe to modify: `C7-C only`
- Frozen in C7-B: `True`
- C7 note: C7-B should learn a residual moment-aware VR score rather than overwrite this head.

### vcmr_score_generation

- File: `inference.py`
- Class/function: `compute_query2ctx_info / compute_query2ctx_info_disjoint`
- Lines: `163-368; 371-563`
- Input shape: `begin/end logits (B,Nv,Lv), video scores (B,Nv)`
- Output shape: `flat top max_before_nms VCMR tuples per query`
- Training usage: train-time validation
- Inference usage: S_video * p_start * p_end for general/exclusive; start+end for disjoint
- Safe to modify: `C7 wrapper preferred`
- Frozen in C7-B: `True`
- C7 note: C7 integrated scorer should be implemented as an alternate inference function, leaving original intact.

### nms_top100_logic

- File: `utils/inference_utils.py; utils/temporal_nms.py`
- Class/function: `post_processing_vcmr_nms/filter_vcmr_by_nms/temporal_non_maximum_suppression`
- Lines: `inference_utils.py 21-76; temporal_nms.py 25-74`
- Input shape: `prediction lists [video_idx, st, ed, score]`
- Output shape: `top max_after_nms predictions after per-video temporal NMS`
- Training usage: train-time validation when nms_thd != -1
- Inference usage: official/eval post-processing
- Safe to modify: `False`
- Frozen in C7-B: `True`
- C7 note: Do not alter NMS/top100 logic. C7 must produce better scores before this boundary.

### loss_computation

- File: `model/conquer.py`
- Class/function: `CONQUER.forward/get_moment_loss_share_norm`
- Lines: `188-226`
- Input shape: `begin/end logits (B,Nv,Lv), st_ed_indices (B,2), optional video_similarity_score (B,Nv)`
- Output shape: `scalar loss and loss_dict`
- Training usage: moment CE plus optional video CE
- Inference usage: not used
- Safe to modify: `C7 wrapper preferred`
- Frozen in C7-B: `True`
- C7 note: C7-B should compute additional losses outside original forward or in a subclass without changing baseline loss.

### checkpoint_loading

- File: `train.py; inference.py`
- Class/function: `encoder_pretrain load; setup_model`
- Lines: `train.py 345-359; inference.py 650-677`
- Input shape: `checkpoint state_dict`
- Output shape: `CONQUER model with loaded weights`
- Training usage: can load encoder/query_weight pretrain
- Inference usage: loads full checkpoint['model']
- Safe to modify: `C7 wrapper preferred`
- Frozen in C7-B: `True`
- C7 note: C7-B should load C6/B2-compatible CONQUER checkpoint and freeze original parameters.
