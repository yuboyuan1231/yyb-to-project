# C7 Tensor Interface Map

| tensor | shape | producer | consumer | C7 use |
|---|---|---|---|---|
| `query.feat` | `(B,Lq<=30,768)` | StartEndDataset.get_query_feat_by_desc_id | VideoQueryEncoder.forward_repr_query | query encoder input; frozen in C7-B |
| `visual.feat` | `(B,Nv,Lv<=100,4352)` | StartEndDataset.__getitem__ | VideoQueryEncoder.forward_repr_video | video encoder input; frozen in C7-B |
| `sub.feat` | `(B,Nv,Lv<=100,768)` | StartEndDataset.__getitem__ | TwoModalEncoder | subtitle branch input; frozen in C7-B |
| `_query_feature` | `(B,Lq,768)` | CONQUER.encoder(task='repr_query') | repeat_interleave for shared videos | available in intermediates; can condition RLEM heads |
| `video_feature_dict` | `dict modality -> (B*Nv,Lv,768)` | CONQUER.encoder(task='repr_video') | QDF compute_final_score | diagnostic export only in C7-B |
| `moe_weights_dict` | `dict modality -> (B*Nv)` | QueryWeightEncoder | compute_final_score | modality agreement/conflict features |
| `QDF_feature` | `(B*Nv,Lv,768)` | compute_final_score | BiDirectionalAttention | frozen feature source |
| `QAL_feature` | `(B*Nv,Lv,3072)` | BiDirectionalAttention | contextual_QAL_feature_learning | primary C7 head input |
| `Contextual_QAL` | `(B*Nv,Lv,768)` | FCPlusTransformer | MomentLocalizationHead and G concat | proposal confidence and MIL input |
| `G` | `(B*Nv,Lv,3840)` | cat(QAL_feature, Contextual_QAL) | ML head and VS head | primary integrated head input |
| `begin_score_distribution` | `(B,Nv,Lv)` | MomentLocalizationHead | loss/inference | boundary evidence and proposal confidence |
| `end_score_distribution` | `(B,Nv,Lv)` | MomentLocalizationHead | loss/inference | boundary evidence and proposal confidence |
| `video_similarity_score` | `(B,Nv) or None` | VideoScoringHead when exclusive | video CE and inference | residual video score anchor; remain frozen in C7-B |
| `inference_vr_scores` | `(B,max_vcmr_video)` | StartEndDataset eval first-stage ranklist | compute_query2ctx_info when use_interal_vr_scores | external VR score anchor |
| `VCMR score tensor` | `(B,max_vcmr_video,Lv,Lv)` | inference score composition | top max_before_nms flatten/sort | C7 score enters before flatten/sort, not after NMS |
