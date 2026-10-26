# KV Cache journal 最小协议与生命周期

日期：2026-10-25（北京时间）。接续指标审计与真实单卡请求事件验收。本轮产物目录：`results/kv_journal_20261025_j7bol44q/`。本文件定义实现合同；测试和 NPU 结果另按阶段保存，设计不冒充实测结果。

## 1. 已核实的数据源

- `BlockPool.cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id, block_mask=None)`：逐个跳过 null/masked 块，赋 group hash，然后调用 `cached_block_hash_to_block.insert`。只有成功 insert 后记录 cache_insert；逻辑下标为 `num_cached_blocks+i`。当前调用可能在 allocate_slots/schedule 内发生，属于缓存元数据登记，不是设备写入完成通知。
- `_maybe_evict_cached_block(block)`：旧 hash 非空且 `pop(old_hash, block_id)` 成功才是真驱逐；记录原 group/hash/ID。`evict_blocks(set[int])` 委托它，因此覆盖显式驱逐。没有请求关联时不构造 request_id。
- `BlockPool.free_blocks` 只减少引用和放回 free queue；不删除缓存映射，不发送 cache_remove。物理分配也不等于 cache_insert。
- `reset_prefix_cache()` 只有没有活动块（null block 除外）时成功；清 map/hash 后记录 cache_reset；失败不伪造 reset。
- `KVCacheManager.allocate_slots` 成功分配后，在该生命周期内显式复制 `get_block_ids(request_id)`，得到按 group 的完整占用表；不把 get_block_ids 本身变成回调。free 成功后记录 request_release，不能解释为逐块缓存驱逐。
- 真实 BalanceScheduler 当前普通配置委托上游 schedule；异步 AsyncScheduler 在 update 输出时还可能 cache_blocks。采集设于共同 BlockPool/manager 层，避免仅覆盖一个 schedule 实现。

## 2. 模式、范围与信任边界

请求协议 request-v1 保持原样；增加 `VLLM_LMETRIC_KV_EVENTS=1`，必须同时 `VLLM_LMETRIC_EVENTS=1`。新 envelope 协议名 `request-kv-v1`，`kv_events=true`，journal 版本 `kv-journal-v1`。

仅支持单卡 TP=PP=DP=1、单前端、单 FullAttention KV group、scheduler/hash block_size 相等、max_num_seqs=1、max_model_len<=8192、纯文本 token IDs、prefix cache、同步或异步 eager。拒绝 sliding window/chunked-local/hybrid、多组、speculation、connector/external KV、LoRA/MM、cache_salt、抢占/resume/abort；不将多组 ID 展平。Rust 首版只为 lmetric-q/radixtree、simulator 关闭启用。无恢复协议；错误或断流锁存失效，清树并停止使用该副本的过期状态。

journal 表示 **scheduler_cache_metadata**：它反映实际全局缓存查找表，而不是某次 NPU 写入已完成。异步完成输出可能携带已调度的后续批次元数据；不将 mutation_seq 与 step_id 一一配对，也不称为设备 cache commit。当前单卡 worker 有序执行是支持范围；异常时立即失效，不把未确认内容用于跨引擎数据迁移。

## 3. 字段

每个 EngineCore 的 tracker 生成一个 engine_epoch，journal 与请求 envelope 共用。step_id 继续表示已输出的完成/控制 envelope 的连续序号。journal mutation_seq 从1开始单调递增，只由引擎主线程的实际状态变化产生，不能由 SSE 重编号。

`kv_journal`：version、observation_kind、num_blocks（含 null block 的真实池容量）、mutations 数组。数组内按 mutation_seq 严格排序。没有变更时数组为空，不能表示 reset。

所有 mutation 有 mutation_seq、type、group_id=0、block_size。类型：

| type | 字段 | 语义 |
| --- | --- | --- |
| cache_reset | reason | 实际成功 reset；新空池 attach 时另以 reason=initial_empty 建立明确空基线 |
| cache_insert | block_id、cache_hash(hex)、logical_index、prefix_token_ids | 实际建立的全局内容→物理映射；逻辑位置与 token 前缀在 cache_full_blocks 中可靠获得；不要求 request_id |
| cache_remove | block_id、cache_hash(hex) | 对旧映射成功 pop 的精确删除；不按 hash 过删其他 copy |
| request_blocks | request_id、block_ids | 一组内按逻辑位置排序的完整占用列表；同时可含已缓存块及未完整块；只在变化后记录 |
| request_release | request_id、block_ids=[] | 请求占用已释放；缓存内容可继续命中；不删除全局映射 |

block_id 在 `[1,num_blocks)`，0为 null 不当作可缓存数据。cache_hash 仅为后端身份核验：删除必须匹配当前物理 ID 的旧 hash；不与 Rust xxh3 比较。prefix_token_ids 来自真实 Request 已知 token 序列，长度恰为 `(logical_index+1)*block_size`。Rust 用自己的 BlockHashState 重算整条本地前缀链，建立物理ID→本地hash路径。完整前缀证明避免依赖已结束请求 Entry，也支持全局、无 request_id 的删除；代价是重复 token 数据，首版严格限制缓冲和 payload，不作开销结论。

## 4. 所有权、发送与 reset

journal 由 EngineCore tracker 持有，attach 到该引擎实际 BlockPool。attach 前要求池缓存 map 为空；不静默假定现有映射已知。它独立于原生 take_events，不额外 drain/破坏 vLLM 原生 KV 发布器。

记录有固定最大条数和估算 JSON 字节预算，单条 token proof 也有限额；超限锁存失败并使会话停止，禁止静默丢弃。drain 原子移交 pending 列表，不重置 mutation_seq；不共享跨 epoch 全局队列。请求占用去重表仅保留活动请求，release 后移除。没有无限历史累积。

在真实 scheduler.update_from_output 后、EngineCoreOutputs 构造时 drain 全部已发生变更；这是一个一致的主线程元数据观察边界。异步情况下仍按真实 mutation 顺序传送，不能把下一次 schedule 变更错误归到本步 token 工作量。纯 KV/no-token envelope 不会被 core_client 过滤，AsyncLLM 在逐请求拆分前交给 EventHub，再走独立 SSE。

idle reset 必须生成独立控制 envelope（outputs=[]、prefill_tokens=0、latency=0、event_kind=control），不等待下一条推理请求。首版 reset 仅允许 scheduler 和 batch queue 都已排空；活动期间 reset 在变更前拒绝，避免未完成异步批次跨 reset。成功 reset 经 EngineCore output_queue→原生 ZMQ→core_client→AsyncLLM→SSE，step_id 和 mutation_seq 均连续。

EventHub 先转换 journal 的真实请求身份，再处理输出和结束身份注销；全局变更没有 request_id 不尝试映射。未知身份、队列满、引擎错误、断流都使会话失效，不猜测 ID、不重连继续使用旧账本。

## 5. Rust 原子消费顺序

1. SSE MetricSequence 先验证协议、epoch、step 连续性、能力与 journal 变更连续性、字段和 payload 边界。
2. 消费循环在扣账前验证并暂存整批 journal。按 mutation_seq 串行处理：reset 清空；remove 必须匹配旧 ID/hash；insert 必须是空闲映射槽，禁止缺少 remove 的物理 ID 复用。不能把一批内所有 remove 简单移到所有 insert 之前，否则同 ID 多次复用会乱序。
3. 独立更新请求完整占用表，release 清空/移除请求；占用不是缓存映射。需要时同步 Entry 的 bids，空列表有明确清空语义。
4. 对验证成功的权威物理映射构造本地可命中的前缀树；只有所有前驱块仍有缓存 copy 的路径才可匹配。一次提交后才对策略可见。策略观察/接纳的缓存版本使用单调 mutation_seq，不能复用重建后可能重复的树内部 epoch。全局 insert 不依赖当前 Entry 存活；cache_hash 只核对后端身份。
5. radix 父块删除必须有独立回归，验证单独删除父块不会 panic、不会留下可命中假前缀；物理权威表保留仍存活的子块，可在父前缀重新可用时重建。多 copy 删除只移除该物理 copy。
6. 解析、序号、容量、身份、物理映射或断流错误锁存无效，清除缓存视图；DSL 候选和接纳重检都排除无效副本。记录原因，停止该实验会话。恢复必须新建 engine/frontend/Router epoch。

## 6. 分阶段验证门槛

先写 Python 单元测试并保存未实现时的失败证据，再修改生产钩子；测试真实 BlockPool 方法的 insert/pop/reset 结果、free不驱逐、同 hash 多copy、ID复用、占用列表及缓冲边界，另测原生 IPC 与 EventHub 身份生命周期。Rust 检验重复/跳号/epoch、原子失败、父块删除、reset和失效路由。

真实 NPU 阶段不重复旧基础实验。使用现有Qwen3-0.6B，先同步小池多条有界请求：冷启动→重复前缀真命中→有限不同前缀触发真驱逐/ID复用→idle reset→重算。全部 SSE 由真实 Rust 接收器和 completion_event_loop 消费并与池状态对照，再单独验证异步。如发生无法解释的差异，立即关闭该会话并保存第一个阻塞。每个阶段单独保存报告；不下载权重、不替换环境、不无界压测。
