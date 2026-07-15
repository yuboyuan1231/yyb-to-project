# C28F v5.0-R2 F0/F1 实现—方案一致性报告

评估日期：2026-07-14  
评估类型：编码完成后的唯一一次整包静态评估  
评估结论：**PASS**  
评估边界：本报告形成前未执行 Python import、compile、test、模型加载或 raw `.pt` 读取。

启动事实保持为 `transition_seq=1113`、`F0_A_COMPLETED / F0_A_ANALYZE`、state SHA `4ebfe727d45dbde7e30594d8100f14399907166c7ced62fedb63d2bca92dc177`；F0-A 仍为 generation 4、eval `F0A-EVAL-8b80aa50b2b215a1235e8ab2ed3ac49b`、run `F0A-RUNTIME-71a02539b3dab25d8fe3ff0bea1a71ec`。

用户在本轮完成了 G4 universe 的事实核验，并明确覆盖 Recovery Goal 文档中的旧数字：`train_fit_desc_ids.txt` 为 69,428 个唯一、已排序 ID，文件 SHA256 为 `24007d3f6f1159f55e68a8ee64c779436a34db4bff601f9d9cd74ee6e102bf0b`；它与 8,677 个 `calib_select` ID 的交集为 0。因此 active G4 只使用全部 69,428 个 `train_fit` ID，不做减法、不采样，也不采用旧 oracle 中写死的 60,751。

## 1. 六个 action 是否从当前 state 到终态可达，runner choice 与 dispatch 是否一致

**PASS。** `runtime_runner.RUNTIME_ACTIONS` 与 dispatch 都严格为：

1. `F0_A_ANALYZE`
2. `G4_ROLE_POLICY_LOCK`
3. `RUN_F0_B`
4. `F0_B_ANALYZE`
5. `G6_VERIFY_F1`
6. `G7_FINALIZE`

对应状态后像依次为 `F0_A_ANALYZED`、`ROLE_POLICY_LOCKED`、`F0_B_RUNNING`、`F0_B_COMPLETED`、`F0_F1_ROUTED`、`F1_COMPLETE`、`PENDING_FINALIZATION` 和 `F0_F1_WINDOW_COMPLETE_STOPPED / NONE`。`PENDING_FINALIZATION` 是 G7 内部同一 logical finalization 的必需提交点，不是额外 runner action。

## 2. G1/G2/TRAIN/F0-A 是否只通过已有 bridge/receipt/hash 接入，old goal_control 是否保持只读

**PASS。** active handlers 只读并验证 supersession bridge `ea22ea92671a2c5aa05c92f4f19db2efa8ad013f0d80249f629e984a867afbac`、G1 receipt `85d1152b3b834bde65d2f96074af77dd5f883ac1a05598748b742434ee9a4fa2`、G2 receipt `4aaec261cbf2c2ee29f2e6ce49731dfe4a48f1edac0dbda5dd5c9e3674c423b7`、TRAIN receipt `0771cac8da99eb99c4e92d567d78128f147ad95d5bab9d9ab0ccf9ae89d3542a`、F0-A material `c756b9dfcc4a9f877dc9890eba2b0e953ddad0bc93585ed78f072762ec0655ac` 与 completion receipt `c2a26e76bf45f4b8974e8536bb6f49fabd1b45a3bacce282a9569d660f82ba55`。active 文件中没有以 `CONTROL_ROOT` 为目标的写调用；所有新增正式写入均位于 `goal_runtime`。静态评估时 old `goal_control` 元数据树指纹为 `03bfdbf8d0a3fae07173b7717312daed1fb9d37d81948a0c0f2723657a01e123`，供终态复核。

## 3. G3、G4、F0-B、G6、G7 每项功能和必要产物是否有唯一实现映射

**PASS。** 映射唯一且无 shadow builder：

- G3：`run_f0_a_analyze`，一次 completed-forward derivation，提交规定的 9 项产物与唯一 `g3/STAGE_RECEIPT.json`。
- G4：`run_g4_role_policy_lock`，一次 protected ID projection 与一次完整 69,428-ID train-fit projection，调用 `a4_roles` 的 cluster-atomic/power-adaptive pure functions，提交五角色、lock/policy/power/protected audit、canonical projection/source binding，共 15 项与唯一 g4 receipt。
- F0-B：`run_f0_b` 在任何 query/feature 内容打开前提交确定性 eval/run ID，随后复用 `a4_forward` 的唯一 cursor/chunk forward；`run_f0_b_analyze` 提交规定 7 项与唯一 g5 receipt。completed-state/material 窄窗口只复用既有 RUNNING proof 类型和已提交 chunks，不重发 token、不重开 active source、不重跑模型。
- G6：`run_g6_verify_f1` 复用 `a4_f1` 的 schedule、temporal、checkpoint、recovery、cost、superset、shape 与 synthetic fixtures，提交规定 8 项与唯一 g6 receipt；F2–F5 均为 `NOT_APPLICABLE_SCHEMA_ONLY`。
- G7：`run_g7_finalize` 以内存 read-only gate view 调用 `a4_finalize.build_final_decision`，提交规定 9 项与唯一 g7 receipt，并将 authority close 与 pending record 绑定到 terminal CAS。

## 4. metric、NMS、threshold、query/corpus universe、role policy、routing 和 finalization 是否与旧方案一致

**PASS。** Metric aggregation、bucket/gate/routing 直接调用 `a4_stages` pure functions；runtime NMS 保持 score-desc/video/start/end/source-ordinal 排序、per-video `IoU > 0.7`、最多 100，joint false-positive mass 仍采用旧 oracle 的 `[-1e-12, 1+1e-12]` 校验后 clamp。阈值不允许 observation 后修改。F0-A 使用已封存 8,677-query identity 与原 17,435 corpus；F0-B 从 committed `train_fit_route_dev` 取 query identity，并继承同 checkpoint、corpus、evaluator、NMS 与 threshold。角色策略复用 `a4_roles`，`STRICT_CORE_ONLY / U_formal=17360` 不变；唯一口径修正是按用户确认使用完整 69,428 train-fit universe。Routing 只产生 `ROUTING_CONSISTENT` 或 `ROUTING_AMBIGUOUS`；finalizer 仍要求精确 G0–G6 gate 集、两个不同且各一次的 eval ID、零 real-data optimizer update、零 protected/official/F2+ execution。

## 5. F0-A 是否不可重跑，F0-B 是否最多一个 eval ID，训练/F2+/protected eval 是否无可达入口

**PASS。** Runner 没有 `RUN_F0_A`、issue/mark/reserve、training、protected eval、official workflow 或 F2+ action。F0-A 只作为 completed material 输入。F0-B identity 由 goal/attempt、route manifest、route IDs/count 与 input hashes 确定；已存在 RUNNING generation 时逐字段比对并复用同一 eval/run ID，输入漂移 fail-closed，successful budget 上限仍为 1。恢复路径只接受同 cursor/chunks。唯一出现的 F2 字样位于 routing recommendation 和 `NEXT_AUTHORIZATION_REQUEST`，两者均明确不执行、不激活权限。

## 6. active import graph 是否排除旧 stateful a4_control，新增代码是否仅为主路所需

**PASS。** Active graph 为 `runtime_runner -> runtime_stages -> runtime_control/runtime_data/runtime_store`；执行边界按需导入 `a4_forward`，科学 pure modules 为 `a4_stages/a4_roles/a4_f1/a4_finalize`。这些 active 模块没有导入旧 stateful `a4_control` 或 `runtime_workflow`。`a4_forward._a4_control_module()` 明确解析到 compact `runtime_control`。旧 monolith 文件及 runtime_control 中不可达的历史 F0-A 参考函数按 Goal 要求保留物理文件，runner 无入口；清理留给独立 cleanup Goal。

## 7. 是否存在重复 transaction/store、兼容层、双计算、双静态审核、多级 fallback 或无需求安全对象

**PASS。** 唯一 store 是 `RuntimeStore`；每个 stage 只有一份 postimage、一个 stage receipt 和一次 stage CAS。Prepared receipt 是唯一可恢复提交边界；只有 output 而无 receipt 的候选 fail-closed。G7 单独允许规定的 pending intent 文件先存在，若无 receipt 却出现其他 G7 输出则 fail-closed。F0-B forward 使用既有 cursor/journal，不新增 capability/claim/lease 类型；completed-material 窄窗口复用既有 capability 类型与官方 recovery function。没有第二状态根、兼容 dispatch、自动降级、generation 5、备用 universe、旧控制器 fallback、第二 raw shadow scan或第二静态审核流程。

## 静态评估源码指纹

- `runtime_runner.py`: `2d290f34c93cda73670adad76c6cb2899d337b3a7a8509d570739f11cf7e0de7`
- `runtime_store.py`: `e755ec94d0729679847b65b1c2b260137ead3227ef7ad816eb08dc1b237a88a5`
- `runtime_control.py`: `6268ab85e9e9a90de328d46043c2e09f91d4ba716ed70134166625f1e0877eaf`
- `runtime_data.py`: `4dd89fa225a1f3a9abc7abade3651de85d747dd4e21f9ff9410941efcd113ad8`
- `runtime_stages.py`: `7097bd84dfaaa4ff6079b99c79b233abf92f06dcc7f37f91223edf177674dfff`
- `test_runtime_recovery.py`: `384169eb82aefaf69687123cf32bb54b891f55319b20cb39ad5b135d1e18d142`
- `test_a4_control.py`: `c34f2215c7b5c974e35ee340d7f137fbbb7b68086c6bbf3ef5ed8bf47aa4ab5c`

七项均为 PASS，允许进入 import/compile/targeted tests/full suite/temporary no-data dry-run。若测试暴露实现缺陷，只复查并追加记录 changed patch 及受影响映射，不生成第二份一致性报告。

## Changed-patch 静态复审：TRAIN EOF 末行 framing

复审日期：2026-07-14  
触发事实：G3 已以 `transition_seq=1114 / F0_A_ANALYZED` 原子提交；首次 G4 在任何 g4 stage receipt、stage output 或 state CAS 形成前，以 `TRAIN indexed pread framing drift` 失败关闭。失败后的 state SHA 仍为 `c5fc39fde0c3215cf5016cd5cf02c6d9417e8e2e1d4c7c7f14835ea599da39ec`。

根因是已提交 TRAIN 索引构建协议 `_scan_train_jsonl_descriptor_index_once` 明确允许且验证唯一的无换行 EOF 末行，而 active `ImportedTrainIndex.project` 错误要求每个 `pread` 结果都以 `LF` 结尾。绑定源文件的物理末行是 desc_id `62633`，offset `16627440`、length `248`，恰好抵达 descriptor-bound source size `16627688`；该 ID 位于完整 69,428-ID train-fit universe 中。

Changed patch 只新增 `_frame_indexed_jsonl_record` 并在 active indexed projection 调用它：已有 `LF` 的记录原样传递；无 `LF` 的记录仅在 `offset + length == source_size_bytes` 且补入分隔符后仍满足 1 MiB parser cap 时，向无 I/O 权限的纯解析器提供一个虚拟 `LF`。物理源文件、已提交索引、`pread` 内容、`pread_byte_count`、source descriptor/hash 和 open count 均不改变；非 EOF 的缺失分隔符继续 fail-closed，纯 projector 的严格三参数接口及权限面不变。

受影响映射复审结论：

- 第 2 项仍为 **PASS**：只消费既有不可重放 TRAIN index/descriptor/hash，不重建索引、不写 old `goal_control`。
- 第 3 项仍为 **PASS**：只修复 G4 唯一 69,428-ID projection 的合法 EOF framing；没有新增 handler、builder、transaction 或 fallback。
- 第 4 项仍为 **PASS**：desc_id→video_id 的授权字段、完整 universe、role/power/routing 算法均未改变；虚拟分隔符不进入 retained record 或科学 hash。
- 第 5 项仍为 **PASS**：没有新增 F0-A/F0-B token、模型、训练、protected eval 或 F2+ 入口。
- 第 6 项仍为 **PASS**：import graph 不变；辅助函数位于既有 `runtime_data` 主路。
- 第 7 项仍为 **PASS**：失败尝试未产生 g4 receipt/output/state transition；重试仍使用同一唯一 G4 action 和 store。

Changed-patch 源码指纹：

- `runtime_data.py`: `19e47f60a7ec616abbdfbaf9c842186e6190f86408e1478eb28ca01f1e45c2a4`
- `test_runtime_recovery.py`: `84d931908d67ade844c55f777428777d4ae1057a263b073d1006179bd70442cf`

Changed patch 静态复审结论：**PASS**。允许执行该边界的 targeted test、真实 EOF 单记录只读投影、全量测试，然后从未改变的 G4 前状态重试。

## Changed-patch 静态复审：G6 temporal oracle 自哈希字段

复审日期：2026-07-14  
触发事实：G3、G4、F0-B 与 G5 已完成；状态为 `transition_seq=1383 / F0_F1_ROUTED / G6_VERIFY_F1`。首次 G6 在构造内存中的 `g6_control_binding` 时触发 `KeyError: contract_sha256`，未创建 g6 output、stage receipt、transaction 或 state CAS；失败后 state SHA 仍为 `92c277ec6b6cf136b45ba6778b5bd67877f4c081568049b966f281c752672229`。

根因是既有纯函数 `a4_f1.temporal_oracles()` 的自哈希字段按其公开产物契约命名为 `oracles_sha256`，而 active G6 绑定代码误引用 `contract_sha256`。同一绑定中的 `recovery["matrix_sha256"]`、`checkpoint["contract_sha256"]` 与 `cost["protocol_sha256"]` 均已逐项对照实际返回键，保持正确。

Changed patch 只将 `runtime_stages.run_g6_verify_f1` 的该单一引用改为 `temporal["oracles_sha256"]`，并增加一个回归测试固定实际 temporal 产物键和 active G6 引用。temporal oracle 内容、自哈希算法、F1 schedule/checkpoint/recovery/cost/superset/shape 逻辑、8 项 G6 输出、F2–F5 schema-only 口径与状态机均不改变。

受影响映射复审结论：

- 第 1 项仍为 **PASS**：action 与状态可达关系不变，只修复 G6 内部产物绑定。
- 第 3 项仍为 **PASS**：G6 仍只有 `run_g6_verify_f1` 一份实现和 8 项唯一输出，无替代 builder 或 fallback。
- 第 4 项仍为 **PASS**：修复引用的是既有 temporal oracle 自哈希，不改变任何 temporal case、schedule 或 routing/finalization 决策。
- 第 5 项仍为 **PASS**：没有模型前向、real-data optimizer、F2+ 或 protected eval 入口变化。
- 第 6 项仍为 **PASS**：import graph 不变，测试只调用既有 pure function。
- 第 7 项仍为 **PASS**：失败尝试未产生 G6 事务或半提交输出；重试仍走同一 action/store/CAS。

Changed-patch 源码指纹：

- `runtime_stages.py`: `6d05f110ccd7f5ff8a873ca15a46a1eb589e5701d875a1e6052c2bd758bd829f`
- `test_runtime_recovery.py`: `80ba8ffade6df39aba33571feafcdef757f11fe89554792a385c8da0d272fd72`

Changed patch 静态复审结论：**PASS**。允许执行 targeted/full regression 并从未改变的 G6 前状态重试。
