# vLLM-Ascend 0.23.0：实现前验证与请求事件原型

日期：2026-10-25（北京时间）。接续 [指标审计](ascend_metric_audit.md)，仅复核本轮实现涉及的路径。

本轮完成：真实 Rust 消费路径的基线复现、有限账本/分块修复、可选的请求与执行事件链路、CPU 合成事件的真实 MessagePack/ZMQ 和 HTTP/SSE 闭环。**没有启动真实模型引擎或执行 NPU 推理，没有 KV 缓存内容跟踪或性能复现结论。**

## 1. 分支与基线

- Router：`blitz-router`，从 `a005ceb3a6689d30514bd27e45ede0db933d0fea` 新建 `lmetric/ascend-minimal-events`。原分支为 `lmetric/camera-ready`。
- vLLM：在 `engines/vllm` 建立独立本地 Git 副本，从 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` 新建同名分支。使用该目录实现；没有替换当前 editable 安装或修改 `/vllm-workspace`。
- Ascend 插件：继续以已安装 `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd` 为目标，本轮没有修改插件。
- 修改前状态、已有修改文件副本和 SHA256：`results/ascend_minimal_events/baseline.json`、`baseline_files/`、`router_before.patch`。未 stash、清理、覆盖或提交原有改动；本轮与用户已有改动共享文件时，只局部增补。
- 本轮改动与旧改动的分离清单、差异、最终完整性检查见同目录 `changes.json`、`router_this_round.patch`、`engine_this_round.patch` 和 `preservation.json`。

## 2. 三个实现前问题

### 2.1 Prefill 账本

先在临时副本执行已有驱动，使用实际 `LmetricQ::schedule`、HTTP DTO/转换和 `completion_event_loop`，没有先修改消费逻辑。基线 14 个测试通过，其中 4 个是预期 panic 的缺陷复现，**不是 14 个正确性场景通过**。产物：`baseline_replay/router_replay.json` 与日志。

确认旧作者上报与当前 Router 组合存在负账本：输入64、预测命中32、实际命中32，作者本轮上报64并叠加命中修正，得到 `prefill=-32`。还确认首个输出同时结束时跳过命中修正，以及多 token 输出按事件加一后留下上下文计数残余。没有据此推断作者论文的配套版本或实验结果错误。

新协议 `request-v1` 的合同：

```text
接纳：P = L - H_pred
每个完成调度步：扣实际 prompt 工作 W_step，不含命中、decode token
首次真实输出：再扣 H_actual - H_pred，包括该输出同时结束的情况
正常结束：bs、waiting、prefill、all_tokens 最终归零
```

无抢占/取消时，`ΣW_step=L-H_actual`，因此 `P_final=(L-H_pred)-(L-H_actual)-(H_actual-H_pred)=0`。chunk 没有输出时仍传工作量，waiting 在首次真实输出前保持不变。每个事件的工作量与 token 是本步增量，首次命中是请求级固定量，不可逐步累加。

Rust 生产修复仅增加 **新协议首次输出即结束**的命中修正；旧协议扣账口径保留，回归测试仍明确复现旧协议负账本。另修复生成数量按实际 token 数计数，使多 token 输出后的完成结账闭合。没有修改 LMetric 评分公式或通过 clamp 掩盖负数。

源码：`router/src/engine/colocation.rs::completion_event_loop`、`policies/dsl_runtime.rs::apply_default_after`、`scheduler/kvcache.rs::set_real_token_hits_get_diff`、`policies/mod.rs::Entry::append_state`。

### 2.2 Block Size、group、物理 ID 与 hash

- 对已安装 Ascend `utils.py::refresh_block_size` 原函数做 9 组 CPU 探针，普通非 hybrid Qwen2 配置下，只要 prefix cache 或 chunked prefill 开启，请求16/32/128均得到128。这是配置方法证据，**不是运行后 KV 配置快照**。
- Router CLI `main.rs::kvcache_block_size` 默认仍为16；未来目标启动必须传 `--kvcache-block-size 128`，不能仅改引擎参数。
- 已复现并修复 `BlockHashState::new` 把 `block.len()<16` 写死的问题，改用实际 block_size。`append_tokens` 改为消化所有已完整块。测试覆盖16/32/128粒度、尾块、跨多个块追加，以及128粒度的物理 ID→本地 hash 对应。
- 当前 `KVCacheManager.get_block_ids(request_id)` 返回分组的完整列表；`CachedRequestData.new_block_ids` 通常是增量，不能当作完整列表。物理 ID 是引擎块池索引，不能跨 engine/group/pool 混用。
- Router 使用 seed=42 的 xxh3 token 链；vLLM 默认使用 SHA256，并有 group 等缓存命名空间。**两者算法不一致，也不要求字节相等**。未来主路径应以同一 token 块的位置和物理 ID 建立对应，而不是直接比较后端 hash。多组、SWA、外部 KV 和 simulator hash 桥接本轮未实现。
- 新事件携带 `block_size`、`kv_groups=1`；Rust 拒绝多组协议消息，并在真正扣账前检查每个已接纳请求的分块粒度，发现16/128不匹配则停止消费，事件不应用。**这尚未接通全局副本摘除机制，发生错误必须停止并重启本实验会话，不能继续向该副本路由。**

源码：Ascend `utils.py:1369`；vLLM `v1/core/kv_cache_manager.py::get_block_ids`；Router `main.rs:129`、`kvcache.rs::BlockHashState`、`colocation.rs::completion_event_loop`。探针：`preflight.json`。

### 2.3 Scheduler 与同步/异步执行路径

当前没有已初始化的目标引擎，没有 `/dev/davinci*`，第一次只读 `npu-smi info` 在15秒超时；本轮没有重试或启动模型。因此 `preflight.json` 的 actual_scheduler / actual_kv_groups / actual_engine_core_path 均明确为 null。

固定源码可确认：

- `SchedulerConfig.get_scheduler_cls()` 按显式 scheduler_cls/async_scheduling 选择；`VllmConfig` 在允许时可能默认开启 async，eager 不能代替关闭 async。
- Ascend `platform.py:693` 起还可能选择 recompute/dynamic batch/profiling chunk；`patch_balance_schedule.py::BalanceScheduler.schedule` 有自身重写路径。
- `EngineCore` 的实际 scheduler 实例由 `get_scheduler_cls()` 创建；`max_concurrent_batches>1` 使用 `step_with_batch_queue`，否则使用 `step`。async_scheduling 与 batch queue 是两个独立维度。

因此本轮没有只修改通用 Scheduler 的 `schedule()`。改在 **EngineCore 调用实际 scheduler 的入口**采集该次返回结构，并在同步/队列两条路径的 `update_from_output()` 之后附加事件。调度前进度取自该次 `NewRequestData/CachedRequestData`，不取随后已推进的 Request 当前进度。

每次 SchedulerOutput 保存私有快照及开始时间；异步调度提前推进了后续请求状态，也不能覆盖旧输出所属快照。seq 是完成的调度输出序号，不是客户端发送次数或 NPU 前向次数。合成测试执行原始两条 EngineCore 方法，并使用重写 schedule 的夹具；**没有实例化真实 Ascend Scheduler，不能证明所有特殊调度器已兼容**。运行时新增日志记录实际类名、单组 spec、scheduler/hash block_size、async 标志和队列大小。

## 3. 最小传输合同

启用开关：`VLLM_LMETRIC_EVENTS=1`。默认关闭。拟用配置：`configs/ascend_minimal_events.json`，标记为 INTENDED_ONLY，不能冒充运行快照。

```text
EngineCore 调用实际 Scheduler.schedule
  → SchedulerOutput.lmetric_snapshot（冻结本次调度）
  → 对应结果 update_from_output
  → EngineCoreOutputs.lmetric_event
  → 原生 msgspec / ZMQ
  → core_client 保留 metric-only 消息
  → AsyncLLM 在逐请求输出拆分前分发
  → GET /v1/metrics：独立 SSE
  → VllmStepReceiver：版本/epoch/连续seq检查
  → 原生 DTO 转换 → completion_event_loop
```

事件字段：`lmetric_protocol`、`engine_epoch`、`step_id`、`latency`、`latency_kind`、`execution_mode`、`kv_events=false`、`block_size`、`kv_groups`、本步 `prefill_tokens`、`outputs`。每个请求输出包含 `request_id`、`new_token_ids`、`state`、`is_finished`、`hit_token_cnt`、`prev_computed_tokens`。

- `state=PREFILL` 表示请求第一次真实输出；后续为 DECODE。中间 chunk 是 `outputs=[]`，不伪造生成 token。
- `latency` 为整数毫秒，从该次 schedule 开始到对应 update 完成。异步模式包含在队列中的停留时间，多个事件区间可能重叠；**不是 device time、客户端 TTFT、互不重叠的吞吐 tick，也不能据此比较性能**。
- API 读取明确的 `X-Request-Id`，经 trace_headers 传入 AsyncLLM，在内部随机 ID 创建后注册一一映射。没有按字符串前后缀猜测 ID；SSE 将确切内部 ID 映射回 Rust u64。
- 单订阅者；先订阅才能接纳请求。重复订阅拒绝；断开、队列满或未知身份使会话失效；Rust 对重复、跳号、epoch变化、缺字段和不支持的能力报错，此后不再恢复应用增量。心跳是 SSE 注释，不伪造执行步。
- 没有续传/状态恢复或自动摘除副本。错误后应停止本实验，重新建立引擎/前端/Router会话；不能沿用旧账本继续工作。
- 范围：单前端、PP=DP=1、一个 KV group、同一 scheduler/hash block_size、纯文本 token IDs、单 prompt、n=1。禁用 speculation、connector、外部 KV/EC、LoRA、multimodal、stop strings、echo。检测到抢占/恢复明确失败。abort ACK 尚未实现，启用原型时 abort 明确报错并要求重启会话。
- 本轮 **没有导出** new/evicted cache hash、物理映射、cache reset 等 KV 事件；能力标志明确为 false。预算、TTFT 和完整缓存容量也没有用虚构值填充。

## 4. 本轮文件与目的

### 隔离 vLLM 分支（9 个源码文件）

| 文件（相对 engines/vllm） | 目的 |
| -- | -- |
| `vllm/v1/engine/lmetric_events.py`（新增） | 配置限制、请求事件快照/完成、身份映射、单订阅分发和失效语义 |
| `vllm/v1/core/sched/output.py` | 本次调度私有快照 |
| `vllm/v1/engine/__init__.py` | EngineCoreOutputs 增加可选事件结构 |
| `vllm/v1/engine/core.py` | 实际调度入口及同步/队列完成点、运行信息、未支持 abort 的显式拒绝 |
| `vllm/v1/engine/core_client.py` | 防止空 token 指标消息被过滤 |
| `vllm/v1/engine/async_llm.py` | 内部 ID 注册、输出拆分前分发 |
| `vllm/entrypoints/openai/serving_metrics.py`（新增） | 独立 HTTP/SSE 端点 |
| `vllm/entrypoints/openai/api_server.py` | 仅启用时注册端点 |
| `vllm/entrypoints/openai/completion/serving.py` | 明确传递 Router ID，限制本轮请求模式 |

没有修改 BlockPool、KVCacheManager、Coordinator、Scheduler 选择或驱逐策略。

### Router 分支（10 个源码文件）

| 文件（相对 blitz-router） | 目的 |
| -- | -- |
| `router/src/engine/vllm_http.rs` | 版本、粒度和能力 DTO，严格序列校验及失效状态 |
| `router/src/engine/client.rs` | 保留协议/块粒度，接入真实 SSE 校验；测试可见性适配器 |
| `router/src/engine/colocation.rs` | 新协议首次结束的命中修正、块粒度检查，持久化重放测试入口 |
| `router/src/engine/ascend_request_tests.rs`（新增） | 实际接纳、实际完成循环和真实 SSE 接收测试 |
| `router/src/scheduler/kvcache.rs` | 修复硬编码16与一次只处理一个完整块 |
| `router/src/scheduler/policies/mod.rs` | 按实际 token 数更新生成计数，保留其余已有修改 |
| `router/src/scheduler/simulator/{mirror,mod,pctx,sched}.rs` | 仅补测试构造器的新字段 None，不改 simulator 算法 |

新增验证脚本：`scripts/ascend_minimal/{support,test_events,synthetic_server,preflight,finalize}.py`、`scripts/run_ascend_minimal_tests.py`；新增配置和本报告。报告链接也补到两仓库的相关文档。

## 5. CPU 测试与可复跑方式

```bash
cd /root/lmetric-repro
python scripts/ascend_minimal/preflight.py
python scripts/run_ascend_minimal_tests.py
```

最终实际结果：

| 检查 | 结果 |
| -- | -- |
| 实现前 Rust 基线 | 14通过，内含4个预期panic缺陷复现 |
| Ascend block-size 原函数 CPU probe | 9/9配置组合符合128覆盖规则 |
| 新增 Python CPU 测试 | 9通过 |
| 新增 Rust 定向测试（含独立合成 SSE） | 9通过 |
| Router 常规 lib 回归 | 34通过、2明确忽略；其中网络用例已在上一行单独执行，另一项需公开轨迹输入 |
| simulator feature 测试构造编译 | 成功，仅编译，不算运行通过 |
| 用户已有15个文件的基线保留检查 | 全通过；重叠修改可移除本轮增补后逐字节还原 |
| 已安装 vLLM / vLLM-Ascend HEAD 与工作树 | 与基线完全相同 |

Rust 定向9个用例同时包含在 lib 测试集里，以上数量不可简单相加当作互不重叠的测试数。

`results/ascend_minimal_events/tests.json` 和各 `.log` 保存实际结果。测试层次：

1. **实现前基线**：真实 Rust 消费循环，14个测试（4个预期panic）；旧合同问题可复现。
2. **Python CPU**：生产新模块、原始 msgspec 输出结构和选定 EngineCore/AsyncLLM 方法；测试隔离加载，未初始化完整 vLLM。未使用的 schema 依赖类型设为 Any，不能替代全包导入和完整引擎 IPC 验收。
3. **真实 ZMQ IPC**：独立 Python 子进程通过 loopback PUSH/PULL 传递原始输出结构的 msgpack；事件输入是合成的。
4. **Rust 回归**：预测/实际命中矩阵、分块、首次输出即完成、多 token、16/32/128尾块与映射、粒度不一致拒绝、独立序列错误用例。
5. **真实 HTTP/SSE**：独立 FastAPI/uvicorn 合成服务使用生产 serving_metrics、EventHub 和 RequestEvents；384输入、128命中、128粒度，两次128的计算 chunk，再 decode/finish。原生 Rust VllmStepReceiver→真实转换→真实 completion_event_loop，最后 bs/waiting/prefill/all_tokens 均为0。
6. **相关库回归与编译**：Router 常规 lib 测试；simulator feature 的测试构造器编译。网络集成用例在普通 cargo test 中明确 ignore，由专用 driver 启用；不会要求常规测试隐式启动服务。

驱动只为本次构建在 `/tmp` 建立已有 libssl/libcrypto 的开发链接，不安装系统库、不改用户 Cargo 配置。Rust 依赖使用已有锁文件离线构建。完整数量及 ignored 项以本轮 `tests.json` / 日志为准，不把 ignored 或编译成功算作执行通过。

## 6. 未确认的 Ascend 运行行为

没有真实模型启动配置和运行后 KV specs；实际类、异步/图模式、stream同步、placeholder/停止/取消竞态、NPU输出裁剪、完整跨进程初始化、采集开销均未验证。CPU方法重放证明的是指定输入下的代码与合同，不是设备执行时序。

当前原型没有自动恢复和全局副本失效传播；异常后不能继续生产路由。KV事件尚未实现，Router无法据本原型维护真实缓存树，不能声称缓存感知 LMetric 复现完成。旧协议负账本、radix父块删除缺陷也没有被本轮新协议修复范围覆盖。

## 7. 下一阶段：KV 事件采集的具体顺序（本轮不实施）

1. **真实单卡入口验收**：先用本报告拟用配置，记录最终 Scheduler/MRO、EngineCore路径、batch queue、block/hash/scheduler粒度、KV groups/spec/num_blocks。先同步eager，再单独验证异步与图模式；逐请求对账本轮字段。
2. **新增映射 journal**：在 `BlockPool.cache_full_blocks` 真正 insert 后，记录 `(epoch, mutation_seq, request_id, group, logical_index, physical_id, hash)`；不把 allocate/new_block_ids_to_zero 当 cache commit。
3. **完整占用快照**：成功分配后从 `KVCacheManager.get_block_ids` 复制完整分组列表；与新提交列表分开。先支持单组，禁止 flatten 混用。
4. **真实删除 journal**：在 `_maybe_evict_cached_block` pop 成功、清 hash 之前记录旧ID和内容；覆盖显式 evict/reset。free不报告驱逐，同hash多物理copy分别处理；不额外 drain 原生 take_events。
5. **协议升级与消费**：新能力版本才发送KV字段；按旧删→完整占用→新增映射应用。以物理ID/逻辑位置桥接Router本地hash。先解决已有radix父块删除测试，再开启此能力。
6. **生命周期与恢复**：实现取消发起/ACK、抢占重算账本，增加一致的epoch/reset和恢复快照；把接收失效传至调度器摘除副本。在这些路径对账通过前不扩性能矩阵。
7. **上卡与开销**：重复前缀、全命中需重算尾token、长chunk、资源驱逐、同hash多copy、同ID复用、取消竞态；分别记录同步/异步/graph数据可用顺序、事件年龄及开关采集的开销分布。无预设收益阈值。
