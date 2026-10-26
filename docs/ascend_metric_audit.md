# LMetric 在 vLLM-Ascend 上的内部指标可获取性审计

审计日期：2026-10-24，Asia/Shanghai。范围：本地固定 v0.23.0 源码、当前 Rust 工作树、已有固定 yaullm 归档；无 NPU 推理。

**结论：生成 Token、结束标志已有结构化输出；请求命中、调度 Token 数、完整物理块映射及抢占状态内部也已有，但尚未导出为 Router 所需的逐步 SSE。新缓存映射的请求/物理 ID 归属、真实驱逐 ID、可靠事件传递需要增加事件记录和 IPC 结构。实际 NPU 执行时序、异步/图模式一致性和采集开销仍须上卡验证。**

**不能直接搬运旧补丁。** 真实 Rust CPU 重放确认旧 yaullm Prefill 上报会产生负账本；还确认 Ascend 常见的 128 Token 块大小与 Rust 部分块哈希判定不兼容。最小下一步是先确定并测试跨端合同、处理这些 P0 风险，再在隔离引擎副本实现单实例、单 KV 组、同步路径的观测原型。本次没有修改生产源码、安装包、路由公式，没有提交代码或宣称兼容/性能复现成功。

## 1. 版本、环境与证据边界

### 1.1 实际检查结果

| 对象 | 本次实际结果 |
| -- | -- |
| 工作区 | `/root/lmetric-repro`；本身不是 Git 仓库 |
| Router | `/root/lmetric-repro/blitz-router`；分支 `lmetric/camera-ready`，跟踪 `origin/lmetric/camera-ready`；HEAD `a005ceb3a6689d30514bd27e45ede0db933d0fea` |
| vLLM | 已安装 `0.23.0+empty`；editable 来源 `file:///vllm-workspace/vllm`；源码 HEAD `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，detached HEAD，tag `v0.23.0`，工作树干净，浅历史 |
| vLLM-Ascend | 已安装 `0.23.0`；editable 来源 `file:///vllm-workspace/vllm-ascend`；源码 HEAD `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`，detached HEAD，tag `v0.23.0`，工作树干净 |
| 环境记录匹配 | 上述两个 commit 与既有 `results/npu_smoke/versions.json` 一致。本报告以这组本地版本为目标；未来部署换版本须重审 |
| CPU/Python | aarch64；Python 3.12.13；torch `2.10.0+cpu`；Rust 测试用 `cargo +1.88.0`；`cpu.max=800000 100000`，8 核等效 CPU 配额 |
| NPU 相关包 | 元数据可读，详见 [environment.json](../results/ascend_metric_audit/environment.json)；安装存在不证明运行可用 |
| 设备检查 | 未见 `/dev/davinci*`；`npu-smi info` 返回 187，`dcmi module initialize failed. ret is -8005`。本轮未加载模型、未导入启动 vLLM/Ascend、未执行 NPU 算子 |
| 本地指令 | 工作区未发现适用的 `AGENTS.md`；未使用并行子代理 |

Router 在本轮开始前已有未提交修改，不能将当前工作树等同于 HEAD。检查记录如下，本轮未覆盖或提交这些修改：

```text
 M .cargo/config.toml
 M CLAUDE.md
 M README.md
 M docs/dsl/policies.md
 ? request-sim
 M router/src/engine/colocation.rs
 M router/src/gateway/server.rs
 M router/src/scheduler/policies/llm_d/least_bs.rs
 M router/src/scheduler/policies/llm_d/least_waiting.rs
 M router/src/scheduler/policies/lmetric.rs
 M router/src/scheduler/policies/mod.rs
 M router/src/scheduler/policies/policy_runner.rs
 M router/src/scheduler/policies/vllm.rs
?? docs/cpu-testing.md
?? docs/paper-cpu-tests.md
?? router/src/scheduler/policies/paper_tests.rs
```

本轮固定源码清单覆盖50个文件。完整状态、editable 安装来源及 tracked diff 摘要保存在 [environment.json](../results/ascend_metric_audit/environment.json)。源码路径、函数起始行、SHA256 保存在 [sources.json](../results/ascend_metric_audit/sources.json)。Router 行号均指本轮工作树；引擎行号指上述固定 commit。未使用最新主分支代替目标版本，无需重复下载源码。

### 1.2 复用的已有产物

- [yaullm 静态字段合同](../notes/yaullm_rust_field_contract.md)、[适配审计](../notes/yaullm_adaptation_audit.md)、`scripts/audit_yaullm_field_contract.py`。
- yaullm 固定作者版本 `aa897f04a5a891c39e96ca2d554cf310d2933fa2`，`lmetric/step-reporter-v2`；基点 `6d8d0a24c02bfd84d46b3016b865a44f048ae84b`。复用 `results/yaullm_audit/` 的完整补丁、归档源码和 manifest，没有重新进行相同上游历史调查。
- 本次对 manifest 覆盖的 **25 个既有归档文件**重新核对 SHA256，全部匹配：[archive_integrity.json](../results/ascend_metric_audit/archive_integrity.json)。`data/yaullm-audit` 没有已检出的普通工作树，不依赖它进行本次追踪。
- 作者配套实验 Router 的精确提交仍未确定；本次能确认的是上述 yaullm 与当前本地 Router 的合同及不兼容，不推断论文实验曾运行同一组合。

证据层次：**源码确认**表示固定实现与条件可读；**CPU 重放**表示模拟输入下真实 Rust 消费函数或抽取的原始 Python 方法被执行；**NPU 待验**表示实际配置、硬件执行和异步时序尚未证明。后两者不能互相替代。

## 2. 当前架构、Ascend 可达性与采集时点

### 2.1 已有 yaullm → Rust 数据流

```text
Scheduler / KVCacheManager / BlockPool
  → SchedulerOutput
  → EngineCore.step 补 latency、step_id 和缓存增量
  → EngineCoreOutputs（msgspec / ZMQ）
  → core_client → AsyncLLM.metrics_queue
  → full JSON → GET /v1/metrics 的 data: JSON SSE
  → VllmMetric → EngineStepOutput
  → colocation::completion_event_loop
      请求账本 → 删除旧缓存 → 更新请求完整 bids → 插入新缓存 → 应用 LMetric
      → simulator（若启用）→ 客户端通知 → 异常终止清理
```

Rust 接纳时由 `apply_default_after` 自行增加 `bs`、`waiting_reqs`、`prefill_tokens=L-H_pred`、`all_tokens=L`，随后 SSE 扣账。LMetric 是 Router 的状态账本，并非引擎返回的聚合指标。`waiting_reqs` 直到协议的首次 `PREFILL` 输出才减少；不能用 Scheduler 原生 waiting/running 队列长度覆盖它。[R-ADMIT][R-CQ][R-METRIC]

### 2.2 目标版本现状

```text
NPUPlatform.pre_register_and_update → adapt_patch → Ascend 平台补丁
VllmConfig / SchedulerConfig → EngineCore.__init__ 选择 Scheduler
Scheduler.schedule → KVCacheManager → Coordinator → SingleTypeManager → BlockPool
  → SchedulerOutput → executor → NPUWorker → NPUModelRunner
  → ModelRunnerOutput → Scheduler.update_from_output
  → EngineCoreOutputs → EngineCoreProc.output_queue → Msgpack/ZMQ
  → core_client.get_output_async → AsyncLLM._run_output_handler → OutputProcessor
  → 通用请求输出/日志/Prometheus
```

固定源码没有 yaullm 的 `metrics_queue/report_metrics` 及 `/v1/metrics` 路由。原生 `/metrics` 是 Prometheus 文本聚合端点，原生 completion SSE 也不包含本协议的逐步物理块账本。[V-ASYNC][V-HTTP][V-PROM]

| 配置路径 | 源码可确认的到达关系 | 限制 |
| -- | -- | -- |
| 普通模型、无特殊调度器、同步 | `NPUPlatform.pre_register_and_update:184` → `adapt_patch:563` → platform patch 导入；`patch_balance_schedule.py:56,107,745` 用 `BalanceScheduler` 替换通用 Scheduler，其 balance 关闭且非 PD producer 时走 `super().schedule()`；继承通用输出处理；单 KV 组走原 coordinator | 能确认通用采集点在该条件下可达；未启动实际实例核实类名/补丁导入顺序 |
| 异步 | `SchedulerConfig.get_scheduler_cls:168` 选择 `AsyncScheduler`；其 `_update_after_schedule:20`、`_update_request_with_output:45` 维护 placeholders 并补 cache；Ascend 的 SWA patch 包装通用进度/输出/分配方法 | 不能把同步 `core.step` 补丁直接宣称支持异步；基类绑定也受导入时序影响 |
| 默认选择 | `VllmConfig` 在兼容条件下会把 `async_scheduling=None` 设为 True（957–997）；`core.py:192–220` 根据 `max_concurrent_batches` 选择 `step` 或 `step_with_batch_queue` | eager 不等于关闭异步；最小原型必须显式确定 async、PP、batch queue 状态 |
| balance / dynamic batch / profiling chunk | Ascend `BalanceScheduler.schedule` 有完整重写分支；`platform.py:712–725` 可选择 `SchedulerDynamicBatch.schedule:151`、`ProfilingChunkScheduler.schedule:240` | 不能只给通用 `schedule` 打点。DynamicBatch 的命中/输出构造与通用版本不同，本次未证明其合同一致 |
| PD recompute | `platform.py:693–710` 选择 `RecomputeScheduler` 或 `AsyncRecomputeScheduler`；重写 `schedule:171`、`update_from_output:831` | 有重算、跨实例 KV、延迟释放语义，单组本地缓存合同不覆盖 |
| NPU worker 回流 | `platform.py:669` 选择 `NPUWorker`；`worker.py:514–520` 构造 runner；固定 0.23.0 下即使请求 v2，`worker.py:160–163` 也回退 v1；`model_runner_v1.py:2404,2529` 返回 `ModelRunnerOutput`，包含 `sampled_token_ids` | NPU 真正执行、异步转 CPU Token、图模式耗时/顺序均未验证 |
| 多组缓存 | `patch_kv_cache_coordinator.py:456` 对单组/禁用缓存回退原实现；其他情况及 DeepSeek V4 可用 `AscendHybridKVCacheCoordinator`；压缩 manager 仍调用 `BlockPool.cache_full_blocks:184` | block size、group、CP、压缩比改变物理映射，不能 flatten |

上述证据详见 [A-PLATFORM][A-PATCH][A-BALANCE][A-HYBRID][A-SWA][A-RUNNER][V-CONFIG][V-CORE]；这是一份条件可达性证明，**没有宣称所有 Ascend 配置走同一实现**。

### 2.3 推荐采集生命周期（方案，未实现）

1. **请求提交与 ID 注册**：保存 Router u64 ↔ API 外部 ID ↔ EngineCore 内部 ID 双向映射。原生 completion 会加 `cmpl-`、子请求序号，InputProcessor 还默认添加随机后缀。不得简单 `int(request_id)` 或按最后一个连字符切分。[V-ID][V-COMPLETION]
2. **schedule 内、成功分配之后、推进进度之前**：复制每个已调度请求的 `prev_computed_tokens`、prompt 长度、scheduled Token 数、首次有效命中、完整分组物理 ID；复制成整数/不可变列表，不能持有随后会被复用的 Block 对象。失败的 lookup/allocate 尝试不能算已执行工作。[V-S]（609–658,843–848,997–1011）[V-KV]
3. **cache_full_blocks 每次实际插入映射处**：记录 request、group、逻辑块索引、物理 ID、hash 及本地 mutation 序号。跳过 null/mask；不是按“新分配 ID”推导新缓存。[V-POOL]（211–331）
4. **_maybe_evict_cached_block 真正 pop 成功之后、reset_hash 前**：保存旧 hash/group/physical ID；补全 `evict_blocks`、全量 reset、connector 无效块和取消路径。避免重复调用 `take_events()` 抢走原生发布器的事件。[V-POOL]（365–400,443–495）
5. **执行结果处理**：用属于该 `SchedulerOutput` 的冻结快照关联最终 Token/finish；在 `_free_request` 删除请求表/完整映射前保留必要数据。`Request.take_prefill_stats()` 是一次性消费，应复用输出中的值而非再次调用。[V-S]（1516–1573,1888–1910）[V-REQUEST]
6. **step/IPC**：同步路径在 schedule 前开始、update 后结束计 host 墙钟，ms 转换与合同一致；为无 Token 的 chunk、纯缓存变化、抢占及取消也生成 envelope。取消/重置可能没有模型前向，需控制事件。异步路径另带 schedule ID / 完成 ID / mutation 顺序，不能把外层一次 `step_with_batch_queue` 的耗时视为某一模型步执行时间。[V-CORE]（443–472,484–596,1670–1698）
7. **前端/SSE**：统一 IPC 结构或独立有序 side channel 二选一；优先与模型输出共用 envelope。`core_client.py:1027` 当前只将有 outputs 或 stats 的消息入队，必须加入新指标/控制事件条件。AsyncLLM 在处理和拆分 outputs 前复制完整 envelope 给独立指标分发器，避免被逐请求队列拆散。[V-CLIENT][V-ASYNC]
8. **Router 应用**：沿用同一步先删除旧 ID、再完整 bids、再新缓存插入的顺序；增加 epoch/seq 校验及断流失效/恢复方案后才能保证增量账本可靠。本轮仅提出修改点，不实现它们。[R-CQ]

## 3. 字段可获取性矩阵

分类：A＝已有接口/结构已返回该**原始值**；B＝内部已有，只需采集/导出；C＝需要增加内部记录、数据结构或合同逻辑；D＝仍需运行验证。A **不表示当前 HTTP 已能直接对接 Rust**。端到端 SSE 的公共缺口（扩展 envelope/IPC、过滤、队列、路由）整体为 C；表中先标数据源获取难度，再注明跨端语义。普通单组同步路径的源码可达性见 §2；所有实际 NPU 行为另受 D 限制。

### 3.1 请求与执行信息

| 字段 | Rust 消费位置/必要性 | 引擎源码位置与语义 | 可获取性 | 需要的修改 | 验证方法 | 证据 |
| -- | -- | -- | -- | -- | -- | -- |
| `outputs` | CQ 468 起逐请求计账、742 起通知客户端；主路径必需；但允许空列表 | `Scheduler.update_from_output:1554,1618` → `EngineCoreOutputs.outputs:227`；当前 client 的本步输出列表，不是本步全部被调度请求，部分 prefill 无输出 | A 原始列表；无输出步补 envelope 为 C | 保留 outputs=[] 的调度/缓存消息；不可伪造 PREFILL chunk 输出 | CPU 空 chunk 重放已过；上卡长 prefill 核对每步完整性 | [R-CQ][V-S][V-TYPE] |
| `request_id` | DTO 必填 u64；关联 Entry、缓存及完成；必需 | `Request.request_id` / `EngineCoreOutput.request_id` 是内部字符串；API/输入处理会改写；标识值非计数 | B；映射生命周期 C | 注册/保留精确映射，取消、抢占、终止后延迟清理；不转换 UUID 为 0 | CPU 数字 ID 通过，原生字符串解析失败；上卡核对 API→Core→SSE | [R-HTTP]（184）[V-ID][V-COMPLETION] |
| `new_token_ids` | 非终止分支更新本地 Token/hash、all_tokens；终止步发送 Token 并结账；必需 | runner sampled → `_update_request_with_output` stop 裁剪 → `EngineCoreOutput.new_token_ids`；请求级、本轮最终新增 Token ID，非累计 | A；跨端多 Token 风险需 C | 原值传出，不能用文本反 tokenize；一次多 Token 需先解决 Router 生成计数按事件加一问题 | CPU 两 Token 非终止事件后 finish 留 all_tokens=1；上卡 spec/stop 边界 | [V-S]（1393–1430,1695）[A-RUNNER][R-ENTRY][R-CQ] |
| `state` | 只接受 PREFILL/DECODE，其他值 panic；PREFILL 会减 waiting；必需 | 原生 `RequestStatus.RUNNING` 不是协议 state；调度器明确统一 Token 进度、不划同名阶段；作者由产生输出时 computed==prompt 生成 PREFILL | B 原始进度；协议判定 C | 冻结本步进度并记录“首次真实输出”/resume 状态；不能在每个 prefill chunk 发 PREFILL | CPU chunk 输出为空时 waiting 不减；上卡异步/抢占恢复核对首次输出 | [V-S]（340–349,997）[Y-S]（930）[R-CQ]（479） |
| `is_finished` | bs/all_tokens 结账、Entry 删除/End 通知；必需 | `EngineCoreOutput.finished:199`＝`finish_reason is not None`；当前输出事件的终止状态 | A | 原样读取；保留 ABORT/ERROR 与正常完成的区分，不双发终止/ACK | CPU 正常结束及首 Token 结束；上卡 stop、max_tokens、abort | [V-TYPE][V-CORE]（1670）[R-CQ] |
| `prev_computed_tokens` | 主 CQ 解包忽略；simulator 才用于重建输入进度 | `NewRequestData.num_computed_tokens`、`CachedRequestData.num_computed_tokens` 是调度前进度；`_update_after_schedule:1010` 随即推进，异步时可能含未来 in-flight/placeholder | B | 启用 simulator 才需按本步冻结导出；不能输出完成时的 Request 当前值 | 静态确认；未测试 simulator 运行或 async 对齐 | [V-OUT]（31,112）[V-S]（1085）[V-ASYNC-SCHED][R-SIM] |
| `latency` | `Duration::from_millis`，请求内部 TPOT、TBT EMA、step tick；主消费者存在 | 通用 `core.step` 未存 yaullm latency；已有日志区间计时不是相同边界；Ascend profiling 模式的 `execution_time_ms` 仅特定开关下测 runner 并同步设备 | B 加 host timer；实际时序 D | 同步原型测 schedule→execute/sample 等待→update；0.23.0 的 sample 可能单独发生；异步另定义 duration/完成语义 | CPU 仅验证 12.345ms→12ms 解析；上卡对照 host/device 时间和开销 | [V-CORE]（403,443）[A-RUNNER]（2541）[Y-CORE][R-METRIC] |
| `step_id` | 主路径日志，无去重；simulator 用作锚；计账不依赖其数值，但可靠性必须有序号 | 现有 timestamp 为 monotonic 时间点，不是步计数；EngineCoreOutputs 无协议 step_id | C | 建立 engine_epoch + seq，区分 schedule/完成/控制；不能用 per-client 发送次数冒充模型前向次数 | CPU 重复 seq 的 Decode 被重复应用；上卡断流/重启/多前端 | [V-TYPE]（215）[V-CORE]（1264）[R-CQ]（675）[R-SIM] |

### 3.2 Prefill 工作量与 KV Cache

| 字段 | Rust 消费位置/必要性 | 引擎源码位置与语义 | 可获取性 | 需要的修改 | 验证方法 | 证据 |
| -- | -- | -- | -- | -- | -- | -- |
| `prefill_tokens` | `metric_delta.prefill_tokens_dec=m.prefill_tokens`；PREFILL 非终止再叠加 H_actual−H_pred；主路径必需 | `SchedulerOutput.num_scheduled_tokens` 是每请求本轮调度 Token 数，混有 decode/spec；prompt 长度与调度前 computed 可切分实际 prompt 工作；`PrefillStats.num_computed_tokens` 却是首次 prompt 总未命中量，非本轮增量 | B 原始数据；最终兼容合同 C | 按 prompt 区间冻结本轮工作，记录已执行/重算/命中；先解决旧公式负账本及首步结束例外，不能直接改一端公式 | 本轮真实 Rust 重放确认 -H_actual；chunk 无命中正常归零；NPU 核对重算/取消 | [V-OUT]（181）[V-S]（359,515,843）[V-STATS][Y-S]（535）[R-CQ]（467,521） |
| `prefill_token_budget` | DTO/转换搬运，当前主 CQ 和调度评分未读取；可暂缓 | 内部 `max_num_scheduled_tokens`、token_budget 及每请求分配数已有；yaullm 含义是 max budget 减 decode 量，非所有调度后剩余 Token；实例级本轮容量 | B | 需复原定义才导出；暂不实现可省略，由 DTO 默认 0，必须声明“不支持” | 静态消费者搜索；本次不验证原生 budget 等价 | [V-S]（105,360）[Y-S][R-HTTP][R-ADMIT] |
| `hit_token_cnt` | 仅 PREFILL 非终止主路径做真实命中校正；要求是 Router block_size 的整数倍；必需 | `KVCacheManager.get_computed_blocks:205` 返回本地实际可用 prefix 长度；scheduler 成功接纳后 `prefill_stats.set`，输出携带一次性 PrefillStats；请求级首次命中值，非每步新增/累计 hits 统计 | A 结构化 PrefillStats 原值；B 精确采集/保留；多层语义 D | 本地-only 下读 num_local_cached_tokens；num_cached_tokens 含 local+external，不能泛化等同旧 H；避免第二次 take 消费；成功接纳后冻结 | CPU serializer 命中对齐；目标命中方法静态确认；真实重复 prefix/全命中/重接纳上卡 | [V-KV]（205）[V-S]（609–658）[V-STATS]（243）[V-REQUEST]（298）[R-HASH]（217） |
| `new_block_hashes` | 当前主 PrefixBlockHash 不读该值；当前 simulator 也未发现业务读取；可暂缓 | `BlockStored.block_hashes`：本轮新插入完整缓存映射的后端内容 hash；bytes 或转换的低64位 int，带 group，非 Router xxh3 链 | A 原生事件值；跨 hash 系统 C（若使用） | 主路径可省略；启用其他消费者时做明确表示/映射，不截断后宣称与本地 hash 相同 | Python 原生插入方法探针；Rust 主路径用本地 hash 测试通过 | [V-POOL]（211）[V-EVENT][V-HASH]（78）[R-CQ]（596）[R-HASH] |
| `evicted_block_hashes` | 主路径不用；simulator mirror 用 `h[0]` 驱逐 orphan；仅 simulator 必需 | `BlockRemoved.block_hashes`，仅某个 physical copy pop 成功时报告；可能同一 hash 另有物理 copy | A 原生值；simulator hash 适配 C | 普通主路径暂缓；simulator 必须以已知物理 ID→Rust 本地 hash 映射转换，且处理 aliases，不能直接套 SHA 字节/u64 | 目标 duplicate-content probe 已过；simulator hash 桥接未实现/未验证 | [V-POOL]（365）[V-EVENT][R-MIRROR]（89,146） |
| `evicted_block_ids` | `block_hash.remove`，先删再增；主路径必需 | `_maybe_evict_cached_block` 内有 physical `block.block_id`，但原生 BlockRemoved 不保留它；实例块池中映射删除增量，不等于 ref_cnt 归零 | C 增加记录结构 | 在 pop 真正成功处保存旧 ID/hash/group；覆盖显式 evict/reset；增量事件与输出一致传递 | 目标方法 probe 验证 free≠evict、reuse 真驱逐；CPU 合法同 ID 删除新增通过；父块单独驱逐有风险 | [V-POOL]（333–400,419–495）[R-CQ]（608）[R-RADIX] |
| `cur_used_block_ids` | 稀疏 map；每个列出的请求调用 set_bids 替换完整列表；主路径必需；空 value 当前跳过，无法清空已有跟踪 | `KVCacheManager.get_block_ids(request_id):549` 返回 tuple[list[int],…]，按组的完整物理快照；cached request 的 new_block_ids 常为增量、resume 时是替换 | B（单组）；多组/空洞合同 C | 成功分配后、free/preempt 前复制完整列表；按请求有变化才发，未出现不能当空；不能用 allocate_slots 返回值替代 | CPU full map/新提交子集、reuse 重放已过；多组/空洞/重接纳上卡另验 | [V-KV]（545）[V-OUT]（112）[V-S]（1085）[R-HASH]（262）[R-CQ]（609） |
| `new_block_hashes_ids` | get_onto_hashes 从完整 bids 中定位新提交子段，以本地 Token 哈希插入；主路径必需 | `BlockPool.cache_full_blocks:279–292` 有 request、group、真实 inserted block ID；返回 None，原生 BlockStored 不带 request_id/physical ID；请求级本轮新建立映射增量 | C | 新增 journal/返回结构记录实际插入的 ID 与逻辑索引；处理 mask/null/不连续片段；不得用新分配 ID 或全部占用 ID 代替 | 目标 allocation≠commit probe；CPU 全链路映射插入；上卡分块、decode 完块和复用比对 | [V-POOL][V-SINGLE]（298）[V-EVENT][R-HASH]（276）[R-CQ]（636） |

### 3.3 特殊状态与附加字段

| 字段 | Rust 消费位置/必要性 | 引擎源码位置与语义 | 可获取性 | 需要的修改 | 验证方法 | 证据 |
| -- | -- | -- | -- | -- | -- | -- |
| `preempted_ids` | 主 CQ 只日志，不回补 waiting/prefill；simulator 会重置镜像/调度；主计账当前不使用 | `SchedulerOutput.preempted_req_ids:214` / `schedule:940`；本步实际被 preempt 的请求集合；`_preempt_request` free、computed=0、入 waiting | A 通用输出集合；B 导出；特殊 scheduler D | 传出 ID 映射；仅有该列表不能修复主账本；需明确 resume/recompute 如何对账；重写 schedule 单独采集 | CPU 32 Token 部分 prefill 后单发抢占，账本仍 P=32、waiting=1；未声称恢复正确 | [V-S]（974）[V-OUT][A-BALANCE][R-CQ]（455）[R-SIM] |
| `aborted_requests` | `ExtState::Term` 终止 ACK；只能引用已进入异常上下文的请求；主生命周期需要但受限 | `Scheduler.finish_requests:1825` 返回 (id,client)；Core 的 ABORT 消息、`_process_aborts_queue:600`、shutdown/pause 分支及 `_send_abort_outputs:1690` 路径不同；不是普通 finished_req_ids 集合 | B 结束原因已有；C 保存/聚合独立 ACK 并保证全部路径 | 在删除请求前保存 ABORT 原因与 ID；不能把所有 finish 都当 abort，也不能正常完成和 ACK 双重扣账；补取消发起通道须改 Router | CPU 首输出断连→ACK 正常归零；未经本地取消就发 ACK 会 panic；取消前 prefill 残留仍需专测 | [V-S]（1825）[V-CORE]（378,600,1670）[R-CQ]（824）[R-CLIENT]（321） |
| `op_exec_log` | Option；未找到主业务消费者 | 作者 serializer 本就未提供；目标也无本合同字段 | 不需要采集；若强制要求则 D（无合同） | 保持缺省 None | Rust 缺省反序列化通过 | [Y-TYPE][R-HTTP]（174） |

表内 `V-S:行号`、`R-CQ:行号` 等均对应 §10 的具体文件链接；跨端共同导出 C 不得被 A/B 标签掩盖。仅同名 Prometheus `kv_cache_usage`、prefix hit 累计计数不能满足上述逐请求、逐步合同。[V-STATS]（171）[V-PROM]

### 3.4 哪些由 Rust 自行维护、哪些可暂缓

无需引擎直接上报：`bs`、`waiting_reqs`、`all_tokens`、剩余 `prefill_tokens` 的当前账本值；路由预测 `H_pred`、输入 Token、本地 PrefixBlockHash/epoch、decision_epoch、客户端队列时间、由事件推导的 TBT EMA/TPOT/TPS。引擎提供真实事件供其校正；不能用实例级 counters 覆盖。[R-ADMIT][R-METRIC][R-HASH]

在 **lmetric-q + HTTP + 单组 + simulator 关闭** 的明确范围内，可暂缓 `prefill_token_budget`、`prev_computed_tokens`、`new_block_hashes`、`evicted_block_hashes`、`op_exec_log`；`preempted_ids` 当前只日志，但采集点已有，建议保留用于诊断。`step_id` 虽当前只日志，可靠传输设计仍列 P0，不能据此省掉序号。取消/抢占功能不能靠省略字段宣称支持。

## 4. KV Cache 专项分析

### 4.1 命中 Token 能否取得，是否与 yaullm 一致

**能取得，但等价有条件。** `get_computed_blocks` 的第二个返回值是本次 lookup 找到的本地最长可用前缀；可成功调度后存到 `PrefillStats.num_local_cached_tokens`。该值不是 Prometheus hits、不是每轮累计命中，也不是全部块数量乘 block_size。`PrefillStats.num_cached_tokens` 还加了 external KV。首次 prefill 的总未命中量 `num_computed_tokens` 也不是调度步增量。[V-KV]（205–240）[V-S]（609–658）[V-STATS]

作者 yaullm 首次接纳时将 `request.num_cached_tokens` 从 -1 设置为当时 computed，以后保留；普通本地单组、无 connector、首次调度下与 local hits 可对应。preempt 后 lookup、PD external cache、prefix read 被跳过、MTP/EAGLE、异构 KV 组都可能改变含义，不能普遍断言相同。[Y-S]（535–541）[V-KV][V-ASYNC-SCHED]

全前缀命中也需重算最后 Token 以取得 logits；目标将 lookup 上限设为 `request.num_tokens-1`，块对齐可使整块重算。Router 预测可能命中全部完整输入块；实际 H 小于预测是正常边界，不应强行改成预测值。[V-KV]（220–230）[R-HASH]

### 4.2 完整物理列表、新分配和新映射

物理 ID 是引擎块池中的整数索引，不是NPU内存地址；原生代码仍沿用 gpu_blocks / MEDIUM_GPU 等命名，不能凭该名字否定Ascend路径。**完整列表可直接在内部读到**：`get_block_ids(request_id)`，按 KV group 返回，包含缓存命中及新占用块。要在分配完成、释放前复制；NewRequestData 首次带全表，CachedRequestData 对 running 通常只带新块，resume 则替换，不能误当每步完整表。[V-KV]（545–551）[V-OUT]（112）[V-S]（1085–1144）

三个集合必须分开：

| 集合 | 原始产生点 | 正确含义 |
| -- | -- | -- |
| newly allocated | BlockPool.get_new_blocks、SingleTypeManager.allocate_new_blocks | 从 free queue 取出的物理块；可能尚未形成完整缓存、可能仅为 lookahead/external 预留 |
| used blocks | Coordinator.req_to_blocks / KVCacheManager.get_blocks | 请求当前占用的有序全表，含命中、未完整块；有组与 null 占位语义 |
| newly cached mappings | cache_full_blocks 中真正赋 hash 并 insert 的块 | 内容前缀→物理块映射本轮新建立；已分配块在本轮填满后才可能出现 |

`take_new_block_ids()` / `SchedulerOutput.new_block_ids_to_zero` 是用于内存清零的分配记录，会被 drain，且可能按条件才启用；它不是新缓存提交列表，不能为了观测额外调用它。[V-KV]（570）[V-OUT]（234）[V-SINGLE]（298）

**本轮新映射对应的物理 ID 可在插入处取得，但事后现成事件不能完整恢复。** `cache_full_blocks` 参数有 request、block list、group，并按 null/mask 过滤；新增 journal 是最小可靠采集方案。不应先用 num_cached_block 的差值截 slice 就宣称适用于稀疏 mask；计数可越过被跳过块。[V-POOL]（211–331）[V-SINGLE]（316–341）

### 4.3 真正驱逐、请求释放和物理复用

```text
allocate physical B → 分配/引用
  → 写入/缓存登记 hash H → cache map H→B
  → 请求结束/free（ref_cnt 降为0）→ B 回到 free queue，H→B 仍可命中
  → 下一次分配取到 B
      → _maybe_evict_cached_block 成功删除旧 H→B，保存旧 ID
      → B 用于新内容，之后才可能登记 H'→B
```

还存在 `evict_blocks` 显式删除仍被引用块的缓存映射，以及 `reset_prefix_cache` 全部清空。因而上图是常见复用路径，不是唯一驱逐路径。free 不代表 evict；请求完成不能将其全部块作为 evicted 上报。原始 `free_blocks` 只减引用并入 free queue，Ascend 包装器增加 dedupe/refcount 保护但仍委托该释放逻辑。[V-POOL]（333–495）[A-FREE]

`_maybe_evict_cached_block` 只有 hash 非空且 `cached_block_hash_to_block.pop(hash,id)` 成功才删除有效映射；保存 ID 要在此位置完成。内容相同的 hash 可对应多个物理 copy，pop 一个 copy 后另一个仍有效。原生 BlockRemoved 不带 copy ID；只按 hash 删除会过删，目标方法 CPU probe 已验证该条件。[V-POOL]（34–121,365–400）[V-EVENT]

`reset_prefix_cache` 返回 `AllBlocksCleared`，原协议没有独立 reset 字段。最小方案是在同一 engine epoch 内导出所有已知映射 ID 的删除，或定义 epoch/reset 控制消息并要求 Router 清空/重建；不能静默断线重连并保留旧树。[V-POOL]（462–495）[R-CQ]

### 4.4 hash 是否必须映射

主路径 **不需要比较 vLLM hash 与 Rust hash**：Router 用输入/新增 Token 自行计算 xxh3 前缀链，由物理 ID 找到新提交内容的本地位置，再更新 PrefixBlockHash。因此最小单模型、纯文本、无 LoRA/盐/外部 KV 原型可不实现后端 hash JSON。[R-HASH]（175–300）[R-CQ]（636–667）

但这个结论以一致 Token 序列、块大小、逻辑位置和缓存命名空间为前提：

- vLLM 默认 `prefix_caching_hash_algo="sha256"`；hash key 还可包含 LoRA、MM、cache_salt、prompt embeddings，内部拼 group ID；ExternalBlockHash 可为 bytes，配置为 int 时仅取低64位。[V-CACHE-CONFIG][V-HASH]（40–82,484–588）
- Rust 的主链不包含上述所有额外命名空间；启用这些能力时仅 Token hash 会混淆不同缓存内容，需要扩展 Router cache key 或限制范围，不能伪装相等。
- simulator 的 mirror 驱逐使用 `BackendBlockHash[0]`，并假定与本地 hash 相同；这个假定不能套到目标 SHA256。若开启 simulator，需要 ID→本地 hash 桥接，并处理多 copy。仅把 SHA256 拆成四个 u64 仍不能解决算法不一致。[R-MIRROR]（89–107,146）

### 4.5 块大小、多组与分层缓存

**新增 P0：Ascend 默认有效块大小风险。** 固定 `refresh_block_size:1369` 对普通非 hybrid 模型，prefix cache 或 chunked prefill 启用时会把非128值改成128；常规 AscendAttentionBackend 报告支持 `[128]`。真实原函数 CPU probe 将显式32改成128。不能假定把启动参数设为16就能沿用作者路径。[A-BLOCKSIZE][A-ATTN]

Rust `BlockHashState::new:181–190` 却用 `block.len()<16` 检查尾部；当 block_size=128、输入48 Token 时，会错误生成1个完整块 hash（正确应为0）。CPU 直接调用该构造器已确认。`append_tokens:244` 单次也仅消化一块，多 Token 大步长有额外风险。这里只审计，不改公式或哈希实现。[R-HASH]

多组 list 不能 flatten：group 的 block_size 可不同，含 sliding window/null 占位、Mamba checkpoint、EAGLE 组、压缩比例、PCP/DCP；`SingleTypeManager.remove_skipped_blocks` 会把旧位置换为 null；Ascend 的 coordinator 重新计算有效粒度及 LCM，还存在异步 SWA in-flight 生命周期补丁。[V-KV]（25–88）[V-SINGLE]（448）[A-HYBRID]（153–170）[A-SWA]

CPU/offload/Ascend store、connector medium 及跨节点 KV 不是同一个本地物理 ID 空间，不能把所有事件合入单实例树。本轮只确认这些实现存在及不能兼容 flatten，未完成其专项运行语义审计。最小原型应显式关闭这些能力；正式扩展需要 `(engine_epoch,pool/medium,group,logical_index,physical_id)` 结构及请求/缓存命名空间，属于 C。

## 5. SSE、事件顺序与当前正确性风险

### 5.1 模式、队列和 Serde

- 作者 **full** 输出 prefill、块映射、eviction、抢占/取消；**minimal** 将它们填0或空，仅保留步信息和 outputs，虽然 JSON 能解析，不能维持 LMetric/cache。当前 Rust 访问 `/v1/metrics` 不在 URL 中选择模式，服务端配置必须保证 full。[Y-TYPE]（91–124）[R-HTTP]（138）
- 作者 `metrics_queue` 初始 None，订阅才创建；创建前 `report_once_metric` 直接丢弃。队列无界，多个订阅者共享同一 get 队列，属于竞争消费，非广播；心跳5秒，不代表数据确认。Router work/receiver 并发启动，没有已确认的“订阅已就绪再接纳”屏障。[Y-ASYNC]（224–255）[R-CQ]（165–210）
- 目标 core_client 的 `if outputs.outputs or outputs.scheduler_stats` 过滤会让纯指标/纯缓存 envelope 不可见。stats 可关闭，且只发给一个前端；不能靠开日志让所有步偶然通过。[V-CLIENT]（1027）[V-S]（1637–1647）
- HTTP DTO `VllmMetric` 顶层字段大多带 serde default；`op_exec_log` 缺失为 None。逐请求只强制 `request_id`，缺 state 默认为 **DECODE**，其他计数/Token 可缺省。未知字段（例如作者 ttft）默认忽略。因此成功反序列化不证明采集齐全；缺字段会静默变成0/空/default state。[R-HTTP]（154–205）
- `EngineStepOutput` 这个统一内部类型有不同的默认注解；不能直接把其 Serde 要求当作 HTTP 入口要求。测试走真实 `VllmMetric→vllm_metric_to_step_output`，不是自定义镜像 DTO。[R-CLIENT]（276）
- JSON 解析失败时 CQ 记错误后 continue，已取出的这条事件没有重放；重连 Connected 只日志。`ensure_sse_connected` 对首个非 Connected 的成功项也只放过，若实际库先返回 Event，存在吞首项的条件风险；本轮未建立真实 HTTP/SSE 连接验证该条件。[R-CLIENT]（182–258）[R-CQ]（419–433）

### 5.2 需要处理的兼容性问题

| 风险 | 当前证据与影响 | 级别 |
| -- | -- | -- |
| Prefill 负账本 | 真实接纳 `L-H_pred`，作者新请求上报包含 H_actual，再次命中差值校正；64/32/32 用例结果 -32。score 的 `.max(0)` 可能掩盖它，而非修复 | P0，[Y-S]（535）[R-CQ]（467,521）[R-ADMIT][R-SCORE] |
| 首步即结束例外 | CQ PREFILL+finished 不走 H_actual−H_pred 校正；简单换成“未命中计算量”不能统一解决所有分支。CPU H_pred32/H_actual16、D48 后残留 -16 | P0，[R-CQ]（488–527） |
| 128 vs 16 部分块判定 | Ascend 常见128，Rust literal16导致部分块被当完整块，影响预测/新块映射。CPU 已确认 | P0，[A-BLOCKSIZE][R-HASH]（181） |
| 缓存树删除父块级联 | radix 插入两块1、2后仅 remove([1])，debug 会在 `nblock_canary<=len` panic；实现同时级联删除后续映射。目标缓存池允许独立驱逐 physical copy，不能假定总是整个后缀一起驱逐 | P0，[R-RADIX]（674–736）[V-POOL]（365）；上卡触发频率未知 |
| 缺少导出链路 | 目标原生 SSE 无这些指标，KV events 无 request/physical ID，不能以已有 Prometheus/BlockStored 宣称兼容 | P0，[V-EVENT][V-PROM][V-TYPE] |
| 事件重复/丢失/重置 | 主路径不按 step_id 去重，重复 Decode 重计；重复 PREFILL 二次减 waiting 在 debug panic；旧 epoch ID 可污染树，unknown rid 缓存更新 unwrap | P0，[R-CQ]（455–703）[R-CLIENT] |
| 取消 ACK 条件 | 未进入异常上下文就 ACK 会 expect panic；HTTP abort_request 本身 no-op，CQ frontend abort 也没有完成引擎取消发送；前 prefill 终止清理未扣 waiting/剩余P | P0，[R-CLIENT]（321）[R-CQ]（237,824–850,1094） |
| 抢占/重算 | preempt 列表主路径只日志，没有回补重算工作或重置账本；resume 若再次发 PREFILL 还可能重复减 waiting/违反 phase；仅字段可取≠支持抢占 | P0 语义合同，[R-CQ]（455,122）[V-S]（974） |
| 异步事件跨步归属 | schedule 提前推进 computed，某些缓存映射也在 allocate 时提前登记；全局 take_events 可在处理旧 batch 输出时同时带上更新 batch 的 mutation。需要冻结/带序号的 journal，不能按“当前 Request 状态”回填旧输出 | P0 正式 async 前，[V-KV]（437–451）[V-S]（997,1600）[V-CORE]（510–570）[V-ASYNC-SCHED] |
| 多 Token 事件 | Entry.append_tbt_to_tpot 的 generated_token_cnt 每事件+1，CQ all_tokens 却加 new_tokens.len；两个 Token 输出后结束仍留 all_tokens=1。speculation 下会破坏账本，块哈希单步只消化一块另需核实 | P1（原型禁 spec），[R-ENTRY]（168–188）[R-HASH]（243） |
| 多组/额外 key /外部缓存 | 当前 Rust 数值ID和纯Token hash无法表达；flatten 与直接 SHA 比较均不成立 | P2 扩展；若正式目标启用则升级 P0，[A-HYBRID][V-HASH][R-MIRROR] |

对于取消路径还须区分：普通 `EngineCore.abort_requests:378` 调用 `finish_requests` 后丢弃返回列表，**本身不发送 abort 输出**；`_handle_client_request:1348` 的普通ABORT也走此方法。输入线程把ABORT同时放入input_queue与aborts_queue（1535–1543），scheduler取消是幂等的，但新观测记录仍需避免重复。shutdown/pause则另走 `_send_abort_outputs`。`_free_request` 的 finished 集合混合正常/异常原因，不能直接复制成 aborted_requests。[V-CORE]（378,600,1330,1670–1698）[V-S]（1825–1910）

## 6. 无卡验证：实际执行内容、结果和限制

### 6.1 可复跑命令及隔离方式

```bash
python scripts/ascend_audit/inventory.py
python scripts/ascend_audit/target_probes.py
python scripts/audit_ascend_metrics.py
```

新增脚本只写本轮 `results/ascend_metric_audit/` 与临时目录。Rust driver 将当前 Router 复制到 `/tmp/ascend-router-audit-*`（排除 target/.git），把独立测试模块注入副本的私有模块，给副本的真实转换函数增加**仅测试可见性**。生产 `completion_event_loop`、接纳策略 `LmetricQ::schedule`、Serde 类型、BlockHashState、PrefixBlockHash、LMetric 算子原样执行。原源码 SHA256 记入 JSON；使用已有 Cargo target 缓存、locked/offline依赖，没有编译/启动引擎。

常规事件夹具使用作者合同中的 block_size=16；另有专门的128大小缺陷用例。这是模拟消费合同验证，**不表示固定Ascend版本可以按16运行**。

实际 Rust 命令由 driver 记录，核心为：`cargo +1.88.0 test -p router --lib --no-default-features --features vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule,lmetric-q --locked --offline --jobs 4 ascend_audit -- --nocapture --test-threads=1`。

初次独立编译缺少 `libssl.so/libcrypto.so` 无版本开发链接；driver 在 /tmp 为已有 `/usr/lib64/*.so.3` 建链接并设置该次 LIBRARY_PATH 后通过，没有安装/改动系统库。第一次部分夹具不满足合同，已修正；最终结果在下列产物中。没有把夹具错误作为引擎缺陷结论。

### 6.2 实际结果

**Rust 14 个测试通过，其中4个是显式预期 panic 的缺陷复现；“通过”不意味着当前实现正确或问题已修复。** 一个测试可含多个模拟场景。

| 模拟用例 | 执行对象与观察结果 |
| -- | -- |
| JSON/full/缺失/未知字段/ID | 真实 HTTP DTO 和 client 转换；归档 full roundtrip、12ms、H32、request map通过；未知字段忽略，缺项 default；缺 ID/原生 cmpl 字符串报错 |
| Prefill 四组 | L64；(H_pred,H_actual) 为 (0,0),(32,32),(16,32),(32,16)，均使用作者首次 D64：真实接纳+事件循环结果依次 0,-32,-32,-16 |
| chunk + Decode + finish | 空 outputs 的 D32，再 D32+首次PREFILL，再 Decode/finish，最终 bs/waiting/P/all 全0 |
| 首输出即结束 | H_pred32、H_actual16：D32 可恰巧归零；D48（纯实际计算量）仍为-16；证明结束分支未走普通命中修正 |
| 同 ID 先删再增 | 旧内容 ID1 删除，完整 bids=[1,2,3,4]，新增映射=[1,2,3,4]；完整事件循环保留新四块、旧hash消失；另测单块树物理 ID7 复用最终映射正确 |
| preempt 单事件 | 部分 prefill 已减32，发 preempted_ids 后仍 bs1/waiting1/P32/all64；没有回补/重新接纳 |
| frontend断连→ACK | 首次完整无命中 PREFILL 时响应通道关闭，实际 on_frontend_abort 再接 ACK；最终四项0。**未覆盖首次输出前取消** |
| 重复 Decode | 相同 step_id=2 的 Decode 发两次，all_tokens 从65增到67；说明无去重 |
| 多 Token 首输出→finish | 非终止首次输出两个Token，结束后 all_tokens=1，而 bs=0；记账不一致 |
| 4个预期 panic | 重复PREFILL（waiting下溢）、未先取消就ACK（异常上下文缺项）、仅删除radix父块（canary断言）、block_size128下48 Token 被误认为完整块（独立期望断言失败） |

[router_replay.log](../results/ascend_metric_audit/router_replay.log) 保存 Cargo 完整最终输出；其中插入模块导致副本行号偏移，**生产行号以本文与 sources.json 为准**。[router_replay.json](../results/ascend_metric_audit/router_replay.json) 保存模拟场景结果与原始源文件哈希。[测试模块](../scripts/ascend_audit/router_replay.rs)、[driver](../scripts/audit_ascend_metrics.py)。

**目标源码原方法7个 probe 通过。** 通过 AST 抽取原始 BlockHashToBlockMap、BlockPool 相关方法及 Ascend.refresh_block_size，使用显式模拟 Block/Queue/Request/Event 容器执行，验证：分配≠缓存提交；free零引用仍保留映射；复用触发真驱逐且原生事件无物理ID；take_events drain；同hash另一copy仍保留；reset全清；Ascend32→128配置覆盖。[target_probes.json](../results/ascend_metric_audit/target_probes.json)、[脚本](../scripts/ascend_audit/target_probes.py)。这不是实际完整 Scheduler/Allocator/Ascend patch 组合的运行测试。

**复用作者原 serializer 的3个 CPU 用例通过**（full、minimal、空outputs），输出重定向到本轮目录，不覆盖旧归档。[author_serializer/checks.json](../results/ascend_metric_audit/author_serializer/checks.json)。该脚本输出“未执行Rust循环”仅描述其自身；本轮另一独立 driver 实际执行了 Rust 循环。

### 6.3 Prefill 算术确认与尚未解决之处

真实 Rust 代码已把先前静态推导升级为 CPU 消费证据：

```text
L=64, H_pred=32, H_actual=32
LmetricQ 接纳：P=64-32=32
旧 yaullm 首次事件：D=(64-32)+32=64
CQ 普通 PREFILL 修正：H_actual-H_pred=0
事件循环结束：P=32-64-0=-32
```

这只证实所审计双方在该输入下不兼容，未执行 yaullm 完整 Scheduler。事件的 D 来自固定作者公式，并明确标记模拟。不能据此宣称 NPU 已产生相同事件，也不能推断作者配套实验版一定使用当前 Router。[Y-S]（535）[R-ADMIT][R-CQ]

对非终止首次输出，D 取本轮未命中 prompt 实算量能在受限无抢占场景闭合；但首输出即结束跳过命中差值、抢占重算、取消未完成 prefill 都使单一替换公式不足。下一步应定义 per-request 剩余工作和校正何时入账，再选统一 adapter 或 Router 合同变更；本次未选择/应用修复。

限制：没有完整引擎启动、模型/tokenizer 调用、真实网络/SSE 重连、跨进程 msgspec 扩展编码、NPU forward、异步patch组合、多client/多组/PD/全取消时序/完整抢占恢复实验；没有测试 sampler/stop 的真实输出。没有跑不相关的批量性能基准或把既有CPU吞吐当作本轮NPU结果。

## 7. 最小改动方案与 P0/P1/P2

下表都是**后续建议，未实施**。数据可获取不代表可以只增加一个 HTTP handler 完成适配。

| 优先级 | 最小工作 | 涉及文件、类/函数 | 完成条件 |
| -- | -- | -- | -- |
| P0-合同 | 固定有效block_size/单组配置，定义首次输出、结束、重算、取消、工作量/命中修正合同；核对作者配套Router版本；解决已复现的负P、128尾块判定及radix父块删除 | Router `scheduler/kvcache.rs::BlockHashState::new`、`radixtree/src/block_hash.rs::remove_inner`、`engine/colocation.rs::completion_event_loop`、`policies/dsl_runtime.rs::apply_default_after`；测试可先扩独立driver | 正常/边界事件重放满足账本不变量；不能靠clamp掩盖，修复另行评审 |
| P0-采集 | 成功schedule冻结请求进度/真实hits/完整bids；只在真实cache insert/pop记录new/evicted ID；reset/abort控制事件单独覆盖 | vLLM `core/sched/scheduler.py::schedule/_update_after_schedule/update_from_output/finish_requests`、`core/kv_cache_manager.py::allocate_slots/get_block_ids`、`core/block_pool.py::cache_full_blocks/_maybe_evict_cached_block/reset_prefix_cache` | 无Token步不缺报；新分配/新缓存明确区分；不额外drain原生队列 |
| P0-结构与IPC | 为观测数据加独立journal及携带结构（新增），扩 `SchedulerOutput` / `EngineCoreOutputs`，或同进程有序side channel；完善msgspec兼容和过滤 | `v1/core/sched/output.py`、`v1/engine/__init__.py`、`core.py::step/step_with_batch_queue/abort相关路径/process_output_sockets`、`core_client.py` 输出解码及1027过滤 | CPU IPC roundtrip；空outputs、纯reset/abort也可达前端；双方用同版schema |
| P0-SSE与身份 | 双向ID映射；独立分发队列，握手、epoch/seq、单消费者约束或广播；重连丢失时失效并恢复快照；有界队列不能静默drop | `v1/engine/input_processor.py` ID关联、`async_llm.py::_run_output_handler`；`entrypoints/openai/api_server.py::build_app:156/init_app_state:303` 注册新metrics模块（**当前无 serving_metrics.py，需新增或独立路由**）；Router `client.rs::recv_step`、colocation | 先订阅后接纳；drop/duplicate/gap可检测；不把普通completion SSE代替内部事件 |
| P0-取消/抢占 | 接通取消发起与引擎ACK映射，精确一次清理；明确剩余prompt工作/重算如何回补 | Router `client.rs::VllmEngineClient::abort_request` 及CQ异常状态机；vLLM `finish_requests/_free_request`、`core.py::_process_aborts_queue/_handle_client_request/_send_abort_outputs` | waiting/pre-fill取消、执行中abort、finish/ACK竞态、preempt/resume均可逐步对账 |
| P1-Ascend实际路径 | 打印并存档生效类、KV specs/groups、block/hash/scheduler block_size、async/PP/runner；验证Ascend包装方法；对已启用特殊scheduler补独立采集 | Ascend `platform.py`、`patch_balance_schedule.py`、`patch_async_swa_kv_lifetime.py`、`patch_kv_cache_coordinator.py`，必要时各特殊scheduler | NPU单卡实际走到采集点；关闭/开启模式分别验，不把默认配置当同步 |
| P1-开销与多Token | 验证指标新鲜度、CPU配额/背压、timer边界；修复生成计数按事件+1和多Token hash消费后才开spec | `policies/mod.rs::Entry::append_tbt_to_tpot`、`kvcache.rs::append_tokens`、引擎metric分发/timer | 单/多Token账本、tail块、finish一致；采集开/关对照，无预设性能收益 |
| P2-扩展 | 多KV组/SWA/Mamba/压缩/CP/PD/offload、LoRA/MM/salt 命名空间、simulator hash桥接、多前端/多实例 | Ascend hybrid coordinator/compress manager/connector；Router缓存key/事件结构与simulator | 先建立group+pool+逻辑位置合同，逐能力验证；正式目标若必需应升P0 |

首个工程原型的最小约束：固定本地v0.23.0双commit，单模型纯文本、单实例单KV组、无PD/connector/offload、无spec、显式同步且无pipeline batch queue；**不能把设置block_size16当作已可满足的约束**。先处理生效128与Router哈希问题。采集仅旁路观察，不改变Scheduler选择、BlockPool驱逐策略或LMetric评分。拿到NPU前可以在隔离副本实现/评审原型，但本次任务止于审计和测试，不启动这些生产改动。

## 8. 拿到 NPU 后的具体验证清单

| 假设/场景 | 必须采集的对照 | 验收方式（无预设性能结论） |
| -- | -- | -- |
| H1 采集点实际可达 | 生效Scheduler/基类、manager/coordinator、runner、async/queue配置和启动后block sizes | 每条被审计主链打点出现且与配置一致；特殊路径单独归档 |
| H2 命中定义 | 冷请求、重复前缀、整prompt对齐/不对齐、全命中需重算尾Token；记录lookup、成功admit的local/external H与Rust预测 | local-only H与实际复用前缀一致，block对齐；预测偏差通过合同正确消账 |
| H3 长prefill与空输出 | 每次schedule token区间、无Token chunk、SSE seq及Rust剩余P | chunk不能丢；第一次真实输出才减waiting；完成后P归零 |
| H4 新分配与cache建立 | alloc ID、insert journal(req/group/index/ID/hash)、完整bids | 未完整/lookahead不冒充cache；新增列表能在该请求完整映射中定位 |
| H5 真实驱逐和copy | 小KV容量、共享prefix、同hash并发prefill多copy、free后再命中、父块先驱逐、同ID多次复用 | free不删cache；真实pop才删除对应copy；Rust树不panic、不保留错映射；旧删在新增前 |
| H6 finish/抢占/取消 | 首Token即结束、普通结束、prefill前/中及decode时取消、自然finish与ACK竞态、资源抢占和resume | 每请求只结账一次；bs/waiting/P/all符合合同；排空活动账本全0，缓存可保留 |
| H7 async/ACL graph | 同步基线→async→graph逐个开启；schedule_id、completion_id、mutation_seq、in-flight快照 | 旧输出不借用未来状态；新缓存metadata与内容可用性对齐；同ID复用不跨epoch误配 |
| H8 传输恢复 | 订阅晚于请求、断流、重连、重复/丢seq、队列积压、引擎重启、cache reset、多个订阅者 | 明确失效/拒用旧缓存和恢复快照，不能仅日志后继续使用损坏增量状态 |
| H9 取消控制出口 | 原生ABORT、执行中abort queue、idle取消、pause/shutdown、KV load失败 | 所有出口均有原因和ID，metrics不依赖恰好有下一次forward |
| H10 延迟和采集开销 | schedule/execute/sample/update、IPC/front/SSE/Rust apply和决策使用时点；CPU throttling/队列年龄 | 区分host step耗时、device耗时、传输年龄和客户端TTFT；同配置开关采集对照，报告分布 |

跨进程计时不能直接相减 Python monotonic 与 Rust Instant 的未校准原点；需公共可校验时钟或分别测分段耗时。模式升级以逐步状态对账为门槛，不以上报字段数量或“服务能启动”验收。

## 9. 尚无法回答的问题

1. **最终部署参数/硬件型号**：目前只有本地安装版本，尚无本轮真实启动配置和NPU。异步、图、特殊scheduler、KV组及medium须运行时落盘；本报告的条件结论不能代替部署事实。
2. **作者实验双方的精确配套**：有固定yaullm与当前Router，没有证明这是作者论文配套版本；负P事实不能外推为论文结果错误。
3. **目标所有路径的实际缓存内容完成顺序**：源码中cache元数据可在schedule/allocate提前登记，真实NPU的stream、batch queue、PD延迟free需要实测；CPU模拟只能检查方法逻辑。
4. **完整抢占/前prefill取消/重置恢复的账本协议**：当前消费者本身不完整；字段源已找到，但需要先决定合同，再增加更完整重放与上卡证据。本次没有把未支持路径填假0包装为兼容。
5. **多组/分层/特殊模型支持**：确认结构上会破坏旧假设，没有逐个引擎配置实验；需要能力专项审计，不可估计为几个字段改名。
6. **可靠网络/IPC与性能开销**：本轮没有真实SSE服务器、网络断连及跨进程扩展schema测试；后续原型需要这些测试和NPU对照，当前不能给出“低开销/兼容成功”的结论。

## 10. 源码证据索引与产物

下面所有 vLLM/Ascend 链接固定到实际本地 commit；不是引用 latest/main。括号为本地路径与关键函数行号。Router 链接指当前工作树，需结合本轮 SHA256/dirty状态复核。

### Rust

- [R-HTTP]：`blitz-router/router/src/engine/vllm_http.rs`，请求header:106、SSE:138、VllmMetric:154、VllmRequestStatus:183、default_state:203。
- [R-CLIENT]：`router/src/engine/client.rs`，SSE握手:184、recv_step:231、DTO转换:278、HTTP abort no-op:321。
- [R-CQ]：`router/src/engine/colocation.rs::completion_event_loop:390`；本轮完整源码关键位置455/467/479/521/596/636/702/824，异常Term:1094。
- [R-HASH]：`router/src/scheduler/kvcache.rs::BlockHashState::new:178`、命中校正:217、append_tokens:244、set_bids:259、get_onto_hashes:275。
- [R-ADMIT]：`router/src/scheduler/policies/dsl_runtime.rs::apply_default_after:515`；[R-SCORE] 同文件 queued_tokens:137。
- [R-ENTRY]：`router/src/scheduler/policies/mod.rs::Entry:142/append_state:168/append_tbt_to_tpot:173`。
- [R-METRIC]：`router/src/scheduler/state.rs::LMetric:106/LMetricDec:162/SubAssign:190`；[R-RADIX] `radixtree/src/block_hash.rs::remove_inner:674`、断言:736。
- [R-MIRROR]：`router/src/scheduler/simulator/mirror.rs::apply_sse:88/backend_hash_to_key:148`；[R-SIM] `simulator/mod.rs:165`、`pctx.rs:610`、`sched.rs:183`。

### vLLM 与 Ascend

- [V-EVENT]：`vllm/distributed/kv_events.py::BlockStored:56/BlockRemoved:103/AllBlocksCleared:119`。

- [V-S]：`/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py`；Scheduler.schedule:340、命中:609、输出:935、抢占:974、进度推进:997、cached快照:1085、update_from_output:1329、输出对象:1554、KV take/publish:1600、finish_requests:1825、free:1888。
- [V-KV]：`vllm/v1/core/kv_cache_manager.py`；KVCacheBlocks:25、get_computed_blocks:205、allocate_slots:242、take_events:519、get_blocks/ids:545/549、cache_blocks:553、take_new_block_ids:570。
- [V-POOL]：`vllm/v1/core/block_pool.py`；多copy map:34、cache_full_blocks:211、get_new_blocks:333、真正驱逐:365、free:419、显式evict:443、reset:462、take_events:518。
- [V-SINGLE]：`vllm/v1/core/single_type_kv_cache_manager.py::cache_blocks:298/remove_skipped_blocks:448`。
- [V-CORE]：`vllm/v1/engine/core.py::EngineCore.__init__:96/step:443/step_with_batch_queue:484/_process_aborts_queue:600`、CoreProc输出发送:1545、控制终止:1670。
- [V-TYPE]：`vllm/v1/engine/__init__.py::EngineCoreOutput:170/EngineCoreOutputs:215`；[V-OUT] `vllm/v1/core/sched/output.py:31/112/181`。
- [V-ASYNC-SCHED]：`vllm/v1/core/sched/async_scheduler.py:12`；[V-REQUEST] `vllm/v1/request.py:173,298`；[V-STATS] `vllm/v1/metrics/stats.py::PrefillStats:243`。
- [V-CLIENT]：`vllm/v1/engine/core_client.py` 输出过滤:1027；[V-ASYNC] `async_llm.py::_run_output_handler:637/abort:711`。
- [V-ID]：`vllm/v1/engine/input_processor.py::assign_request_id:223`；[V-COMPLETION] `entrypoints/openai/completion/serving.py:144,180`；[V-HTTP] `entrypoints/openai/api_server.py::build_app:156/init_app_state:303`。
- [V-PROM]：`entrypoints/serve/instrumentator/metrics.py::attach_router:19`；[V-HASH] `vllm/v1/core/kv_cache_utils.py:40,56,78,484,563`；[V-CACHE-CONFIG] `vllm/config/cache.py:92–94`。
- [V-CONFIG]：`vllm/config/scheduler.py::get_scheduler_cls:168`，另见 `config/vllm.py:957–997`。
- [A-PLATFORM]：`/vllm-workspace/vllm-ascend/vllm_ascend/platform.py::pre_register_and_update:184`，worker:669，调度器替换:693–725；[A-PATCH] `patch/platform/__init__.py:20–54`；utils.adapt_patch:563。
- [A-BALANCE]：`patch/platform/patch_balance_schedule.py::BalanceScheduler:56/schedule:107`，模块末尾替换Scheduler；[A-HYBRID] `patch/platform/patch_kv_cache_coordinator.py::AscendHybridKVCacheCoordinator:60/get_kv_cache_coordinator:456`。
- [A-SWA]：`patch/platform/patch_async_swa_kv_lifetime.py:69,79,102,126`；[A-FREE] `patch/platform/patch_kv_cache_utils.py::_ascend_free_blocks:83`。
- [A-RUNNER]：`worker/worker.py:160,514`，`worker/model_runner_v1.py::NPUModelRunner.sample_tokens:2404`，输出:2529，特定profiling耗时:2541。
- [A-BLOCKSIZE]：`vllm_ascend/utils.py::refresh_block_size:1369`；[A-ATTN] `vllm_ascend/attention/attention_v1.py::get_supported_kernel_block_sizes:143`。
- [Y-S]：固定yaullm归档 `results/yaullm_audit/scheduler.py:316,535,930`；[Y-TYPE] `engine_types.py:73`；[Y-CORE] `core.py:260`；[Y-ASYNC] `async_llm.py:224–255`。

产物目录：[results/ascend_metric_audit](../results/ascend_metric_audit)。已有审计文件未覆盖；本次仅新增报告、独立测试/清单脚本及结果，不修改生产代码。最终产物/源码一致性核对见 [checks.json](../results/ascend_metric_audit/checks.json)。

[R-HTTP]: /root/lmetric-repro/blitz-router/router/src/engine/vllm_http.rs:154
[R-CLIENT]: /root/lmetric-repro/blitz-router/router/src/engine/client.rs:184
[R-CQ]: /root/lmetric-repro/blitz-router/router/src/engine/colocation.rs:390
[R-HASH]: /root/lmetric-repro/blitz-router/router/src/scheduler/kvcache.rs:178
[R-ADMIT]: /root/lmetric-repro/blitz-router/router/src/scheduler/policies/dsl_runtime.rs:515
[R-SCORE]: /root/lmetric-repro/blitz-router/router/src/scheduler/policies/dsl_runtime.rs:137
[R-ENTRY]: /root/lmetric-repro/blitz-router/router/src/scheduler/policies/mod.rs:142
[R-METRIC]: /root/lmetric-repro/blitz-router/router/src/scheduler/state.rs:106
[R-RADIX]: /root/lmetric-repro/blitz-router/radixtree/src/block_hash.rs:674
[R-MIRROR]: /root/lmetric-repro/blitz-router/router/src/scheduler/simulator/mirror.rs:88
[R-SIM]: /root/lmetric-repro/blitz-router/router/src/scheduler/simulator/mod.rs:165
[V-S]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L340
[V-KV]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/kv_cache_manager.py#L205
[V-POOL]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/block_pool.py#L211
[V-SINGLE]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/single_type_kv_cache_manager.py#L298
[V-CORE]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/engine/core.py#L443
[V-TYPE]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/engine/__init__.py#L170
[V-OUT]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/output.py#L181
[V-ASYNC-SCHED]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/async_scheduler.py#L12
[V-REQUEST]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/request.py#L298
[V-STATS]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/metrics/stats.py#L243
[V-CLIENT]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/engine/core_client.py#L1027
[V-ASYNC]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/engine/async_llm.py#L637
[V-ID]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/engine/input_processor.py#L223
[V-COMPLETION]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/entrypoints/openai/completion/serving.py#L144
[V-HTTP]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/entrypoints/openai/api_server.py#L156
[V-PROM]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/entrypoints/serve/instrumentator/metrics.py#L19
[V-HASH]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/kv_cache_utils.py#L40
[V-CACHE-CONFIG]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/config/cache.py#L92
[V-CONFIG]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/config/scheduler.py#L168
[A-PLATFORM]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/platform.py#L184
[A-PATCH]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/patch/platform/__init__.py#L20
[A-BALANCE]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/patch/platform/patch_balance_schedule.py#L107
[A-HYBRID]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py#L456
[A-SWA]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/patch/platform/patch_async_swa_kv_lifetime.py#L69
[A-FREE]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L83
[A-RUNNER]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/worker/model_runner_v1.py#L2529
[A-BLOCKSIZE]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/utils.py#L1369
[A-ATTN]: https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/attention/attention_v1.py#L143
[Y-S]: ../results/yaullm_audit/scheduler.py
[Y-TYPE]: ../results/yaullm_audit/engine_types.py
[Y-CORE]: ../results/yaullm_audit/core.py
[Y-ASYNC]: ../results/yaullm_audit/async_llm.py

[V-EVENT]: https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/distributed/kv_events.py#L56
