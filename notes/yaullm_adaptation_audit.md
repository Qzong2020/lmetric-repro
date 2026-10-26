# yaullm 优先适配审计与评审意见复核

日期：2026-10-23（北京时间）。本轮完成源码审计与计划修订，未安装作者引擎、未更改已安装vLLM/Ascend源码、未继续修改路由评分。

结论：先解决yaullm兼容反馈链路，再扩大两卡性能实验。问题是旧版V1定制补丁向当前Ascend V1的语义迁移，不是从V0重新设计V1；工作量应在接口原型验证后再估计。

## 固定版本与可复核证据

- 作者引擎：`aa897f04a5a891c39e96ca2d554cf310d2933fa2`，分支`lmetric/step-reporter-v2`。
- 作者首个定制提交`144a116`之前的基点：`6d8d0a24c02bfd84d46b3016b865a44f048ae84b`。已通过官方`vllm-project/vllm`提交API确认此提交存在，API响应归档为`results/yaullm_audit/upstream_base.json`。
- 完整区间差异：24个文件，新增889行、删除199行；其中`vllm/`下21个文件，新增572行、删除42行。包含文档、测试客户端和调试日志，不能整包视为必须迁移内容。
- 原始补丁：`results/yaullm_audit/author_changes.patch`；文件清单`changed_files.txt`。
- 独立有限历史副本：`data/yaullm-audit/`，未检出工作树；与正在使用的框架分离。
- 当前Ascend环境源码提交见`results/npu_smoke/versions.json`。

源码与模型原始来源：[作者引擎](https://github.com/blitz-serving/yaullm/tree/aa897f04a5a891c39e96ca2d554cf310d2933fa2)、[官方基点](https://github.com/vllm-project/vllm/commit/6d8d0a24c02bfd84d46b3016b865a44f048ae84b)、[固定Qwen2配置](https://huggingface.co/Qwen/Qwen2-7B/blob/453ed1575b739b5b03ce3758b23befdb0967f40e/config.json)。

## 对外部意见的判断

| 意见 | 核验结果与修订 |
|---|---|
| 作者yaullm是V0，需要重建跨进程V1链路 | 该前提不成立。作者代码已有`vllm/v1`、扩展`EngineCoreOutputs`、ZMQ解码及AsyncLLM到SSE的链路。跨进程兼容问题真实存在，但应沿作者V1实现迁移，不能以V0推导改造范围 |
| 8核带来事件陈旧和Host争用 | 需要补充测量。实际`cpu.max=800000 100000`是8核等效CPU时间配额，`cpuset.cpus.effective=0-191`，不是仅可用CPU0–7。不能直接照抄绑核编号；先测NUMA、节流和事件延迟，再选择亲和性及线程数 |
| Qwen2-7B固定版本原生上限32K | 与固定revision不符。下载到的原始config是`max_position_embeddings=131072`、`rope_scaling=null`；本地Qwen2.5-7B才是32768。实际引擎启动上限、KV容量和90K算子可用性仍未验证，不能据配置宣称长请求已可运行 |
| 两副本可能分歧不足、缓存过大 | 是应诊断的风险，不是已证实的失败。记录命中、驱逐和同快照策略分歧；保留自然容量实验，增加预先声明的容量扫描。不能为了做出优势而筛选窗口或只保留高分歧设置。5%及40%–70%不是已验证的普适验收阈值 |
| 缓存起点、预热与输出长度 | 采纳“同一冷起点+各策略同源预热”，热缓存分布允许由策略形成；排空不等于缓存清空。现有router请求已发`min_tokens=max_tokens=轨迹输出长度`，当前vLLM最小长度处理会屏蔽停止token，因此不能断言完全没有EOS控制。仍应显式核查最终SamplingParams、上下文停止/取消与逐请求实际输出，必要时补`ignore_eos` |
| 用本地Qwen2.5-7B做预检 | 采纳。已核对层数28、hidden3584、intermediate18944、Q heads28、KV heads4、vocab152064匹配。几何相近不等于运行耗时100%相同：上下文配置、tokenizer、权重等仍不同；其结果仅作工程预检，正式结果回到固定Qwen2-7B |

补充：SSE可以经已有IPC扩展，也可以通过独立有序通道传递；“必须改EngineCoreOutputs才可能取到状态”过于绝对。首选复用作者已有IPC方案，因为它更容易保持step与输出一致性。

## 作者真正的反馈链路与迁移落点

```text
block_pool / KVCacheManager / Scheduler
      → SchedulerOutput
      → EngineCore.step
      → EngineCoreOutputs（msgspec，跨进程）
      → core_client ZMQ解码与过滤
      → AsyncLLM输出处理、metrics_queue
      → serving_metrics / GET /v1/metrics
      → Rust EngineClient / colocation状态与缓存更新
```

| 层 | 作者主要文件 | 当前迁移必须证明的内容 |
|---|---|---|
| 缓存 | `v1/core/block_pool.py`、`kv_cache_manager.py`、`kv_cache_coordinator.py`、`single_type_kv_cache_manager.py` | 完整块提交、命中、真实覆写驱逐、物理ID复用、哈希/多组标记；不把引用归零当驱逐 |
| 调度 | `v1/core/sched/scheduler.py`、`output.py`、`v1/request.py` | chunked prefill、首次输出、抢占/恢复/取消、cached/computed token语义 |
| 核心与IPC | `v1/engine/core.py`、`__init__.py`、`core_client.py` | 字段序列化兼容、step顺序、无token输出但有prefill或cache变化的消息不可被过滤 |
| 前端与SSE | `v1/engine/async_llm.py`、`entrypoints/openai/serving_metrics.py`、`api_server.py` | 请求ID映射、队列积压、连接前握手、心跳、断流与背压；空闲阶段也不能静默丢缓存重置消息 |

当前vLLM已有BlockStored/BlockRemoved/AllBlocksCleared事件及每步调度数，可复用采集位置；但原生事件未完整提供作者协议的物理块编号与逐请求状态。订阅原生事件可以减少重复工作，不能把它视为直接兼容。也不能额外调用破坏性`take_events()`造成另一个消费者丢事件。

## 最重要的语义核验：不能按字段名直接搬运

作者scheduler的`PREFILL`事件是在**产生输出且`num_computed_tokens == num_prompt_tokens`时**生成；部分prefill步通常没有逐请求token输出，但会生成空`EngineCoreOutputs`承载step级指标。Rust端在`PREFILL`时才减少waiting。

因此现有协议中的waiting并不可以直接替换为V1内部waiting队列长度；`bs-waiting`也不自动等于V1内部running队列长度。迁移必须先保留作者跨端合同并记录与论文抽象定义的对应，不能把“进入running”当成原协议的`PREFILL`事件。

作者初次调度时`prefill_tokens += num_new_tokens + num_computed_tokens`，其中包含cached部分，配合Rust端预测/真实命中修正使用；后续chunk采用另一累计方式。这意味着“只发本步未命中token数”并不当然兼容现有Rust消费者。需要建立逐步账本，验证`prefill_tokens`和命中修正的联合效果，而非只比较同名字段。

作者历史还包含chunked prefill无输出上报、cached token计数、物理块累计、抢占ID、取消清理等修复。迁移必须保留这些修复的行为测试，不能只挑最早的step reporter补丁。

这些发现也限定此前CPU差分的证据范围：它验证构造快照上的公式，没有验证真实引擎怎样形成快照。本轮不据此再次修改基线系数。

## 优先实施顺序

1. **补丁与状态合同**：本轮已导出完整diff并识别链路；下一步逐字段列出生产者时点、消费者更新和不变量。选定单副本、单KV组、无speculation/PD的最小范围。该限制仅用于原型，后续正式模式需另验。
2. **单卡Qwen2.5-7B原型**：在工作区内的隔离引擎副本添加最小观测补丁，先贯通“长prefill无输出→IPC→SSE→Rust”；只读观测不得改引擎调度选择、路由评分或缓存驱逐策略。先eager同步路径，再单独验证正式异步/图模式。
3. **语义对账**：重复前缀、全前缀命中边界、多个prefill chunk、长短并发、抢占/恢复、取消、物理块覆写、缓存重置。对照引擎账本与Rust状态逐step检查；排空后活动/等待/prefill归零，但缓存内容可以保留。
4. **时延与开销**：记录core事件产生、前端接收、SSE发送、Rust应用及决策使用的事件序号/年龄；测P95/P99、CPU throttling、队列积压和采集开销。Python`monotonic`与Rust`Instant`不能直接当作同原点数值相减，需同宿主同时间命名空间的公共CLOCK_MONOTONIC时间基准或经过校验的校时方法。队列丢失/重连必须明确失效与恢复，不能静默路由。
5. **再扩双副本与正式模型**：先测两卡独立服务的延迟差异并记录NUMA，不能硬套2%统一阈值；再测共享CPU配额下反馈新鲜度。通过后准备固定Qwen2-7B，处理实际上下文与缓存容量，才进入完整性能矩阵。

首个里程碑不是“移植结束”或“跑出收益”，而是**单卡真实推理、跨进程SSE、Rust三方逐step状态对账通过**。在其通过前，暂停18组等批量性能对比；原来的1–3天估计不作为排期承诺。
