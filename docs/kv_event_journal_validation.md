# LMetric × vLLM-Ascend KV Cache journal：实现与真实单卡验证

日期：2026-10-25（北京时间）。原始日志保留主机 UTC 时间。接续 [指标审计](ascend_metric_audit.md)、[请求事件单卡验收](ascend_single_card_validation.md)，协议见 [kv_event_journal_design.md](kv_event_journal_design.md)。

## 1. 实际结果

**当前限定配置下的 KV 元数据 journal 已实现，并通过真实同步、异步及完整生产 Router 网关验证。** 没有只停留在设计或合成重放：本地 Qwen3-0.6B 在单张 Ascend 910B2 上运行，实际缓存映射、完整占用列表、释放、驱逐、物理 ID 复用与 idle reset 经原生 IPC/SSE 到达 Rust。

| 正式证据 | 真实推理请求数 | 连续 SSE envelope | 连续 mutation | 核验结果 |
| --- | --- | --- | --- | --- |
| `sync_final` | 8 | 87 | 53 | 87/87 步 Rust 权威物理表/占用表与真实池快照一致 |
| `async_release` | 8 | 95 | 53 | 最后 Rust 版本重跑，95/95 步一致 |
| `gateway_release` | 3 | 32 | 16 | 完整生产网关记录32步连续消费；文本/usage/命中/reset/指标通过 |

同步/异步矩阵均验证：首次冷请求0命中，第二次重复前缀真实命中256 token；有限不同前缀后原内容被驱逐，再发0命中；idle reset 先独立到达 Rust，再发0命中。每条结束后 `bs=waiting=prefill=all_tokens=0`，请求占用表为空。最终池内仍有2项缓存映射而7个非 null 块可释放，证明请求释放没有被错误上报为缓存驱逐。

本轮仍是受限元数据原型，**不代表完整昇腾适配完成，不是设备 KV 内容写入完成证明或性能结果**。没有多卡、图模式、无界压测、下载模型、替换系统推理依赖、提交或 push。

有一项历史保全例外：旧 CPU 回归测试的固定输出路径重写了 `results/ascend_minimal_events/synthetic_ipc_events.json`。原随机 epoch 无备份，不能逐字节恢复；用户确认无备份。这是一份 CPU 合成样例，不影响旧真实 NPU 验收或本轮独立 NPU 结果。保全审计明确记录例外，**没有宣称全部历史结果原样保留**。后续已用安全包装重定向该测试输出。

总结果：[summary.json](../results/kv_journal_20261025_j7bol44q/summary.json)。

## 2. 真实支持范围与环境

实际运行使用：

- `/models/Qwen3-0.6B`，现存本地权重；所有推理进程 offline 模式，无权重下载。
- 一张910B2，TP=PP=DP=1，一个前端，一个 FullAttention KV group，block/hash size 均为128。
- `max_num_seqs=1`、max_model_len=1024、max_num_batched_tokens=128；物理池 override=8，0为 null，非 null 池容量7。
- 同步明确 `--no-async-scheduling`；异步明确 `--async-scheduling`；均 `--enforce-eager`。运行后对象而非拟用配置确认这些值。
- speculation、KV/EC connector、LoRA、多模态均未启用。额外盐、抢占/resume/abort、多组、滑窗/混合 spec、非 eager 路径明确排除。
- KV 原型限制 max_num_seqs=1、max_model_len<=8192；Rust 首版要求 lmetric-q + radixtree、simulator关闭。不是用省略字段冒充其它模式支持。

实际同步 Scheduler：`vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler`，继承本地 vLLM Scheduler/Interface；实际方法 `EngineCore.step`，batch queue=None。

实际异步 Scheduler：`vllm.v1.core.sched.async_scheduler.AsyncScheduler`，MRO 包含 Ascend BalanceScheduler → 本地 Scheduler → Interface；方法 `step_with_batch_queue`，容量2、最终队列0。

两种模式实际 spec 为 `FullAttentionSpec(block_size=128,num_kv_heads=8,head_size=128,dtype=bfloat16,kv_quant_mode=NONE,head_size_v=128,sliding_window=None,attention_chunk_size=None)`。完整 MRO、实际 import 路径、parallel config、num_blocks、可用 memory、队列等保存于各会话 `VALIDATION_RUNTIME` 和 verified.json。没有拿预期配置文件填运行快照。

实际 vLLM/EngineCore 来自 `/root/lmetric-repro/engines/vllm/vllm/`；Ascend 插件继续来自 `/vllm-workspace/vllm-ascend/vllm_ascend/`。HEAD 仍分别为 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。Python3.12.13、torch2.10.0+cpu、torch-npu2.10.0.post4、vLLM/Ascend0.23.0 沿用现有安装；只建项目内无安装依赖的 `.venv --system-site-packages`，以进程 PYTHONPATH 加载源码，没有改 editable 安装。

## 3. 引擎采集和传输

[BlockPool](../engines/vllm/vllm/v1/core/block_pool.py) 在实际 `cached_block_hash_to_block.insert` 成功后记录 insert，在实际 `pop(old_hash,physical_id)` 成功后记录 remove；显式 `evict_blocks` 复用同一删除钩子。`free_blocks` 不产生缓存删除。成功 reset 在 map/hash 清理后记录；失败 reset 不伪造事件。

[KVCacheManager](../engines/vllm/vllm/v1/core/kv_cache_manager.py) 在成功分配的生命周期位置主动复制按 group 的完整 `get_block_ids`，变化时记录 request_blocks；coordinator free 完成后记录 request_release。完整占用表不等于 new_block_ids，也不等于已缓存映射。没有将 get_block_ids 改成自动回调或把多个 group flatten。

新 [KVJournal](../engines/vllm/vllm/v1/engine/lmetric_kv_journal.py) 属于本 EngineCore tracker，attach 前要求实际池为空。它独立于原生 `take_events`，没有额外 drain 或改变原生缓存策略。使用同一 engine_epoch，mutation_seq 从1连续递增；drain 只移交 pending records，不重置序号。

缓冲限制：8192条/16MiB、单条 finalized token proof最多8192 token，活动占用去重表上限1024；EventHub另有32MiB队列字节预算及原有条数上限。越界锁存失败，不能静默丢弃并继续使用增量状态。

新协议 `request-kv-v1` / `kv-journal-v1`；请求旧协议 `request-v1` 保留，默认 KV 开关关闭。启用需同时设置：

```bash
VLLM_LMETRIC_EVENTS=1
VLLM_LMETRIC_KV_EVENTS=1
```

journal 的观察含义明确为 `scheduler_cache_metadata`。cache_full_blocks 可在 schedule 内提前登记元数据；异步完成 envelope 可包含后续在途批次已经建立的真实元数据。本轮在 update 后按全局 mutation 顺序导出一致观察点，**不把 mutation_seq 解释成当前输出步的 NPU 写入完成序号**。

传输复用既有路径：`lmetric_event → EngineCoreOutputs → 原生 msgspec/ZMQ → core_client → AsyncLLM → EventHub → /v1/metrics SSE`。纯 KV/reset 的 outputs=[] 不会被过滤。idle reset 在 scheduler与batch queue都排空后生成独立 control envelope，经实际 output_queue 发送，不依赖下一条推理。

EventHub 先映射 journal 请求身份，再处理最终输出和身份注销。全局 insert/remove/reset没有虚构 request_id；insert 的 logical_index 来自真实循环下标，prefix_token_ids 来自真实 Request finalized token 序列。engine错误、未知身份、满队列或断流要求会话重启。

## 4. Rust 消费和独立回归

[kv_journal.rs](../blitz-router/router/src/engine/kv_journal.rs) 维护按物理 ID 的权威映射和独立请求占用表。后端 hash 作为 insert/remove 身份核验值；**不等同于 Rust xxh3**。Rust 使用真实 prefix_token_ids 和自己的 BlockHashState 重算本地前缀链。

SSE先校验协议、epoch、step与mutation连续性、group/粒度/容量/负载边界。消费循环在扣账前暂存验证整批变更，按 mutation_seq 串行验证旧删除、复用、新插入；同物理 ID 未先删除不得插入。不能粗暴把一批内全部 remove 移到全部 insert 之前，破坏多次复用顺序。验证成功后一次发布树和占用状态。

旧 radix 的父块删除会连带剪枝后代，原断言却要求删除数<=显式ID数，独立测试真实复现 panic，现按实际失效映射数核验，不靠 clamp。新权威表保留仍实际存活的孤儿子块；仅在所有前驱缓存存在时构建可命中路径，父前缀恢复后可重建。多 copy 只删指定物理 copy，不按 hash 全删。

另发现并先回归复现：同规模重建树的内部 epoch重复，使评分时128-token命中在真实内容替换后仍被接纳。修复为使用单调 mutation_seq 作为 KV 缓存版本，接纳锁内必要时重新 lookup。

失败时锁存无效并清树，DSL候选剔除该副本，评分到接纳之间再次检查；不会继续依赖过期缓存状态。该机制不是完整的客户端取消/恢复协议；无效会话应停止，不能重连恢复旧账本。

## 5. 正式 NPU 矩阵与对照依据

每个8请求矩阵输入均为真实文本经现有模型 tokenizer 得到的313个 token，每条 max_tokens=min_tokens=8、n=1，先由真实 VllmStepReceiver 完成 GET SSE订阅，再实际 HTTP提交。模式序列：A、A、B、C、D、E、A、reset、A。

| 请求位置 | 实际/预测命中 | 真实 prompt 工作量 | 验证目标 |
| --- | --- | --- | --- |
| 第1条A | 0 / 0 | 313 | 冷启动真实新映射 |
| 第2条A | 256 / 256 | 57 | 真正复用两个完整块 |
| B至E | 各0 / 0 | 各313 | 有限不同内容，形成真实淘汰压力 |
| 驱逐后A | 0 / 0 | 313 | 旧映射已真实删除、物理ID已复用 |
| reset后A | 0 / 0 | 313 | control在新请求前独立到达、清树后重算 |

同步和异步各有14个实际 insert、7个实际 remove、22个完整占用变化、8个release，以及2个reset标记（其中一个是attach时验证过的initial_empty基线，只有一个是实际reset操作）。不是把 initial_empty 算成第二次引擎 reset。

[analyze_npu.py](../scripts/kv_journal/analyze_npu.py) 对每个 envelope：重放严格序号 → 比较真实 Rust 应用后权威表 → 比较独立遍历真实 `BlockPool.cached_block_hash_to_block._cache` 的快照及实际 get_block_ids 占用。旧hash删除、不同内容复用、请求释放不驱逐均有真实证据。所有生成 token IDs 解码后与相应 HTTP 文本一致，usage=313+8=321。

正式证据：[同步 verified.json](../results/kv_journal_20261025_j7bol44q/sync_final/verified.json)、[异步 verified.json](../results/kv_journal_20261025_j7bol44q/async_release/verified.json)。完整 SSE、Rust日志、池快照、NPU进程表和启动命令分别在会话子目录。比较的是权威物理映射/占用；没有声称逐字节比较设备 KV tensor 内容。

同步KV矩阵在后续网关聚合修复前完成；该修复不改KV mutation或扣账。修复后又验证了完整同步网关3请求，并以最后Rust版本重跑全部异步8请求矩阵。

## 6. 完整生产网关

[gateway_release](../results/kv_journal_20261025_j7bol44q/gateway_release/verified.json) 使用实际构建的 `router` binary，不是私有测试循环替代入口。单个后端、kvcache_block_size=128，本地 tokenizer、模板模式none，最多3条 HTTP `/v1/chat/completions`，中间一次实际 idle reset。

真实前端路径：HTTP → tokenizer/validation → LmetricQ接纳 → work_event_loop → VllmClient → NPU → SSE → 生产completion_event_loop → HTTP响应。原有 LOG_LEVEL 开启 cache_tracking后，连续32个生产消费步骤与32个SSE和独立池快照对应。预测/真实命中为0、256、0；三条返回HTTP200，文本等于完整真实SSE token序列解码，prompt/completion/total=313/8/321。四项副本指标最终全0。

这一步发现并修复了之前未验收的网关响应缺陷：on_finish_request曾发送默认GeneratedText，Infer未传输入长度，最终token没有纳入Entry响应元数据。现在Prefill携真实输入长度，在扣账完成后计入最后token，最终文本用全部真实token IDs整体解码，usage由真实token数量生成。结束元数据计数错误先由回归复现为Some(0)而应为Some(8)。没有因此改变KV账本或缓存策略。

当前SSE仅有结束布尔值，未新增所有finish_reason/取消原因合同。网关本轮限定min=max=8、正常length结束，未据此验收EOS、stop strings、取消等完整产品语义。

证据：[gateway.log](../results/kv_journal_20261025_j7bol44q/gateway_release/gateway.log)、[responses.json](../results/kv_journal_20261025_j7bol44q/gateway_release/responses.json)、[metrics.txt](../results/kv_journal_20261025_j7bol44q/gateway_release/metrics.txt)。

## 7. 测试、失败与有界性

| 检查 | 结果与性质 |
| --- | --- |
| 先写Python测试 | python_red.log记录未实现journal的失败，之后才改生产逻辑 |
| 最终Python journal测试 | 14通过；真实BlockPool/manager/core方法，输入是CPU合成夹具 |
| 原request-v1 Python回归 | 安全重定向后9通过，含原生跨进程ZMQ；没有把合成结果称作NPU |
| 最终Router lib | 43通过、3明确忽略；新NPU用例常规测试中ignored，另由有界driver实际运行 |
| 独立radix | 17通过；父块删除修复前失败证据保留 |
| 观测脚本 | 1通过，修复emit参数冲突后再上卡 |
| 完整网关 | binary离线编译成功，正式3请求真实NPU通过 |
| git diff --check | 两个工作树均通过；未安装缺失的rustfmt组件 |

这些数量不能把Rust子集与全集重复相加。阶段测试日志保留在同一新产物根目录。

首次KV上卡被独立观测脚本 `emit(kind=...)` 参数冲突中断，已停止；修复并单测后用新目录继续。首次网关测试temperature=0被原校验拒绝，未推理；之后真实网关发现上述响应元数据错误，停止、增加回归、修复后继续。第一次逐步日志验证使用了RUST_LOG，而项目读取LOG_LEVEL，因此该次不声称具备32步消费日志证据，正式gateway_release已补齐。所有失败/预验目录保留，不覆盖成功证据。

每个矩阵最多8条推理，网关最多3条；启动240秒、驱动180秒或单请求30秒等硬界限。每次先npu-smi确认无其他进程，再创建自己独立的process group；只终止本轮进程组。所有实验已停止，最终 [npu_final.txt](../results/kv_journal_20261025_j7bol44q/npu_final.txt) 确认无NPU运行进程。没有长时间性能压测。

多个会话在主动关闭时仍出现先前已知的EngineDeadError/force-kill日志和1个semaphore清理警告；与推理期的错误分开记录，未声称优雅关闭已修复。

## 8. 工作保全及历史样例例外

开始时保存两个仓库全部现有源码副本、tracked/untracked SHA256、HEAD、分支和diff，并单独采集request-sim子模块；未reset、clean、stash、checkout、commit或push。原分支均为 `lmetric/ascend-minimal-events`，HEAD未变，已有修改与新修改共享工作树；本轮相对基线的差异单独保存。

最终逐一检查：没有原文件缺失；未改动的原文件哈希一致；原安装vLLM/Ascend和request-sim子模块的HEAD/status/已采集内容一致；旧真实单卡验收目录43文件哈希全部一致。见 [preservation.json](../results/kv_journal_20261025_j7bol44q/preservation.json)。

历史合成IPC样例例外已经明确披露，见 [history_integrity_issue.json](../results/kv_journal_20261025_j7bol44q/history_integrity_issue.json)。用户确认无备份，原epoch仍标未知，未伪造重建；当前样例副本封存在本轮目录，不作为旧实验原始输出。安全包装 [run_request_regression.py](../scripts/kv_journal/run_request_regression.py) 将旧测试的唯一产物ROOT重定向到新目录，后续9项回归成功并验证历史文件在该次安全运行前后SHA256相同。该例外不影响真实NPU结果，但不能将“所有历史文件原样保全”写成通过。

## 9. 阶段报告、复跑和剩余边界

阶段报告分别保存于产物根的stage1、stage2、stage3、stage4_sync、stage4_async、stage5_gateway；失败修复过程、真实/合成类别和最终汇总均可追溯。所有新实验只在 `results/kv_journal_20261025_j7bol44q/` 子目录启动，旧请求回归误写例外如上单列。

本机复跑（NPU空闲时，必须选一个未存在的新结果名称；脚本拒绝覆盖）：

```bash
cd /root/lmetric-repro
# CPU测试显式加载项目源码：
PYTHONPATH=/root/lmetric-repro/engines/vllm:$PYTHONPATH \
  .venv/bin/python scripts/kv_journal/test_journal.py
.venv/bin/python scripts/kv_journal/run_npu.py sync sync_new_name
.venv/bin/python scripts/kv_journal/analyze_npu.py sync_new_name
.venv/bin/python scripts/kv_journal/run_gateway.py gateway_new_name
.venv/bin/python scripts/kv_journal/verify_gateway.py gateway_new_name
```

本机运行器使用本轮已编译Rust驱动和固定已有模型/fixture，源码启用、超时和关闭逻辑均在脚本中。它不是通用生产部署入口。旧CPU回归必须经安全包装或另行指定新输出根，不能直接调用其固定历史输出路径。

尚未验收或明确不支持：多请求并行调度、取消/抢占/恢复、断流续传、任意EOS/stop边界、整块全命中尾token重算专项、跨引擎内容迁移、多group/滑窗/混合/CP/外部KV、LoRA/MM/salt、图模式、长时间运行与性能。每次提交目前重建有界前缀视图并重复传token proof；开销未测，不作生产性能承诺。保留当前限制和失败即失效语义，不自动扩展到多卡或压力矩阵。
