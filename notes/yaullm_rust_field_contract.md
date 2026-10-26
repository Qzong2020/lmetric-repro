# yaullm 字段产生与 Rust 消费：静态协议核对

日期：2026-10-23（北京时间）。范围：作者`yaullm@aa897f04a5a891c39e96ca2d554cf310d2933fa2`与当前`blitz-router@a005ceb3a6689d30514bd27e45ede0db933d0fea`的HTTP+SSE路径。Rust当前工作区还有此前测试改动，但本文涉及的接纳prefill公式、命中修正等关键表达式也核对了未修改HEAD。

本轮没有上卡、加载模型或修改生产代码。完成源码字段追踪；另外仅在CPU上执行作者原始纯序列化函数，并保存受限假设下的静态算术推导。**字段来源与消费已经查清，不等于这对版本的状态更新已经兼容或真实运行通过。**

## 1. 数据流和三个不同对象

```text
yaullm Scheduler / KVCacheManager / BlockPool（产生本步增量）
  → SchedulerOutput
  → EngineCore.step（补充耗时、step_id，复制状态增量）
  → EngineCoreOutputs（msgspec，经ZMQ跨进程）
  → core_client（解码和过滤）
  → AsyncLLM输出处理 → metrics_queue
  → simplified_engine_core_outputs_to_dict(mode=full)
  → GET /v1/metrics，data: JSON
  → Rust VllmMetric → EngineStepOutput
  → colocation：请求/缓存更新 → LMetric和PrefixBlockHash
```

这里有三个容易混淆的对象：

1. 引擎的`SchedulerOutput/EngineCoreOutputs`：Python内部结构，字段名并不完全等于JSON。
2. SSE中的`VllmMetric`：每个执行步的一组增量，不是完整的实例状态快照。
3. Rust的`LMetric`：在路由接纳时加账，按SSE事件减账后形成的本地状态；不是引擎直接发回的bs/waiting数值。

SSE JSON入口：[Python原始序列化函数](../results/yaullm_audit/engine_types.py:73)。Rust解析和无损字段搬运：[client.rs](/root/lmetric-repro/blitz-router/router/src/engine/client.rs:231)。

## 2. 接纳时 Rust 先记了什么

来源：[apply_default_after](/root/lmetric-repro/blitz-router/router/src/scheduler/policies/dsl_runtime.rs:509)。记`L`为输入token数，`H_pred`为路由器缓存树预测的命中token数：

```text
bs             += 1
waiting_reqs   += 1
prefill_tokens += L - H_pred
all_tokens     += L
```

同时保存请求的预测命中块数与决策时缓存epoch。这个时点是router选定并接纳请求，不保证请求已经进入EngineCore。因而静态对账不能把Rust的bs直接减去引擎当前请求数就叫“错误”，需要考虑尚在传输/提交缓冲中的请求。

`all_tokens`是当前请求的输入与已记录生成token的账本，不是物理KV显存占用；块对齐、重用和共享使两者不同。

## 3. 顶层JSON字段逐项对应

产生位置引用均指已归档的固定作者源码。主路径指当前colocation+LMetric/PrefixBlockHash；可选simulator另列。

| SSE字段 / Rust类型 | yaullm如何产生 | Rust如何消费 | 核对结论/限制 |
|---|---|---|---|
| `prefill_tokens: usize` | Scheduler每轮从0累计；已有running请求仍在prefill时加`num_new_tokens`；新接纳请求加`num_new_tokens + num_computed_tokens`。经SchedulerOutput→core→JSON同名字段 | 先设`metric_delta.prefill_tokens_dec=m.prefill_tokens`，PREFILL非终止分支再加命中修正，最后从LMetric中扣除 | 关键跨端语义；下面第6节存在静态算术不兼容迹象 |
| `prefill_token_budget: usize` | 初始`max_num_scheduled_tokens`，running中的decode消耗会扣减；内部叫`prefill_budget`，core映射成`prefill_token_budget` | 解析并搬运；当前源码未发现主路径实际读取其值的计算点 | 不等于普通的“本步剩余所有token预算”；不能随便填值后宣称兼容 |
| `latency: u64` | `EngineCore.step`用`time.time()`测schedule、execute、update的墙钟区间；内部秒，JSON为`int(seconds*1000)` | `Duration::from_millis`，用于请求TBT/内部TPOT、实例TBT EMA及TPS窗口更新 | 非客户端TTFT；不含后续ZMQ/SSE传输，且整数毫秒截断 |
| `outputs: Vec<...>` | update_from_output产生本步有实际输出的请求；部分prefill无逐请求输出时，仍可构造`outputs=[]`的EngineCoreOutputs | 主循环遍历更新请求状态、计数、token和完成；之后慢路径向客户端发送结果 | `outputs=[]`不表示整条消息无用，仍可能有prefill/缓存增量 |
| `new_block_hashes: Vec<BackendBlockHash>` | 缓存新增队列经poll读取并清空；hash整数先转hex，再拆为低位优先的u64数组 | 解析、搬运；主PrefixBlockHash插入并不直接使用这组后端哈希值 | 主路径以本地token哈希+物理ID建立映射；不能误判必须拿此字段直接比较所有hash |
| `evicted_block_hashes: Vec<BackendBlockHash>` | BlockPool真正移除旧缓存映射时产生旧hash，manager排队后poll | 主缓存删除用物理ID；可选simulator镜像按该hash处理驱逐 | 不能因主路径不用而删字段，启用其他路径需核对哈希表示 |
| `evicted_block_ids: Vec<u64>` | 分配物理块时，真正驱逐旧hash成功才记录旧物理ID；manager队列poll后清空 | 先调用`sctx.block_hash.remove(ids)`，再处理本步新块 | 先删旧映射再插新映射，物理ID复用时顺序重要；free引用不是驱逐 |
| `cur_used_block_ids: IntMap<u64,Vec<u64>>` | 内部为`used_block_ids`。scheduler累计请求占用块；仅有新增占用时在本步发出该请求的累计列表；抢占清该请求累计，结束删除跟踪项 | 对列出的请求调用`BlockHashState.set_bids`替换本地物理块序列；本步没出现的请求不清空 | 顶层是稀疏更新；每个出现的value是累计列表，不是仅新增ID。当前作者将KV组flatten，源码提示仅单组安全 |
| `new_block_hashes_ids: IntMap<u64,Vec<u64>>` | cache_blocks返回新建立完整缓存块的物理ID，按请求累积；poll复制后清空 | 对请求调用`get_onto_hashes(new_ids)`，以其本地token链对应段与物理ID插入PrefixBlockHash | 是本步新提交缓存内容的ID，不等于新分配ID，也不是该请求的全部占用ID |
| `op_exec_log: Option<String>` | 当前作者JSON序列化函数没有输出该字段 | Rust可缺省为None；本次路径未发现实际业务读取点 | 不能为了“字段齐全”虚构执行日志 |
| `preempted_ids: Vec<u64>` | schedule在请求因资源不足退回waiting时记录int请求ID，core附入输出 | 主colocation只记日志；可选simulator会调整其镜像/调度状态 | 当前主LMetric路径没有据此直接重建waiting/prefill；抢占后对账需专门验证 |
| `aborted_requests: Vec<u64>` | scheduler在_free_request发现aborted状态，按client汇总；update_from_output附到输出，JSON转int | 走异常状态机Term ACK；取得entry后减少bs及上下文token；当前这段清理没有同步扣waiting或剩余prefill | 不能仅凭收到列表就声称任意阶段取消都能归零；取消前后时点需要CPU重放/真实运行检验 |
| `step_id: u64` | 同步core.step里对每个发给client的EngineCoreOutputs递增并赋值 | 主路径写日志；simulator使用为锚点 | 主路径没有据此去重或丢弃乱序消息；多client时递增单位不是严格的一次模型前向 |

`BackendBlockHash`按编译feature为`[u64;1]`或`[u64;4]`。JSON对象key应保持十进制请求ID字符串，进入Rust后为u64。

产生点：[scheduler.py 305](../results/yaullm_audit/scheduler.py:305)、[scheduler.py 525](../results/yaullm_audit/scheduler.py:525)、[scheduler.py 595](../results/yaullm_audit/scheduler.py:595)、[manager队列](../results/yaullm_audit/kv_cache_manager.py:429)、[core.step](../results/yaullm_audit/core.py:260)。

主消费点：[计数更新](/root/lmetric-repro/blitz-router/router/src/engine/colocation.rs:455)、[缓存更新](/root/lmetric-repro/blitz-router/router/src/engine/colocation.rs:596)、[状态应用](/root/lmetric-repro/blitz-router/router/src/engine/colocation.rs:702)、[异常终止](/root/lmetric-repro/blitz-router/router/src/engine/colocation.rs:824)。

## 4. outputs中的逐请求字段

| SSE字段 / Rust类型 | yaullm如何产生 | Rust如何消费 | 注意事项 |
|---|---|---|---|
| `request_id: u64` | Python内部str；序列化用`int(output.request_id)`。作者completion补丁去掉cmpl前缀和子请求后缀 | 定位entries、异常条目及缓存ID映射，并关联客户端响应 | 当前新引擎可能改写内部ID，必须有明确映射；不能让UUID或cmpl字符串直接进入此字段 |
| `new_token_ids: Vec<u32>` | ModelRunner采样输出经_stop处理剪裁，存入EngineCoreOutput | 非终止快路径更新token/hash链和上下文计数；慢路径逐token向客户端发送；终止分支按现有账本结算 | 真实增量，不是截至当前的全部输出；完成步另有逻辑，不能简单每条都加长度 |
| `state: String` | 有输出时，`num_computed_tokens == num_prompt_tokens`则PREFILL，否则DECODE | PREFILL减少waiting，非终止时处理首次实际命中；DECODE处理后续输出；其他state会触发错误 | 表示作者协议事件；不能把“这个step仍在计算prompt”一律发成PREFILL |
| `is_finished: bool` | EngineCoreOutput.finished属性来自`finish_reason is not None` | 正常完成时减少bs、结算all_tokens、删除entry并发送End；也和异常状态机协调 | is_finished不表示KV内容被驱逐；PREFILL且立即结束的分支不会做普通非终止命中修正 |
| `hit_token_cnt: u64` | request首次接纳时num_cached_tokens由-1设为num_computed_tokens，以后保留；JSON从num_cached_tokens改名 | 在PREFILL且非终止分支设real_hit，并算`H_actual-H_pred`修正；同时用于Prefill通知日志 | 单位token，Rust要求是block_size整数倍；不是每个step新增命中量，也不直接覆盖缓存树 |
| `prev_computed_tokens: u32` | 内部叫num_computed_tokens，但实际填request.prev_computed_tokens；running步在schedule前取快照，新请求首次以缓存计算进度初始化 | 主colocation解包后忽略；可选simulator据此重建计算进度/预测输入 | 首次后的含义与字段名num_computed_tokens不同；重接纳/抢占分支也需审计，不能当完成时当前进度 |
| `ttft`（额外字段） | PREFILL事件附`time.time()-request.insert_to_waiting_time` | VllmRequestStatus没有该成员，Serde默认忽略未知字段 | 不能声称Rust已消费作者ttft；它也不是从客户端发送时刻开始的端到端TTFT |

产生点：[scheduler.update_from_output](../results/yaullm_audit/scheduler.py:924)、[request字段改名](../results/yaullm_audit/engine_types.py:123)。

Rust的[命中差值计算](/root/lmetric-repro/blitz-router/router/src/scheduler/kvcache.rs:217)返回“实际命中块数减预测命中块数”，并设置已同步块数；它不是单纯的统计日志函数。

## 5. 一条SSE到来后，Rust实际按什么顺序处理

1. 将刚发往引擎的条目并入completion跟踪表。
2. 从本步`prefill_tokens`起草扣减；遍历outputs处理状态、命中差值、生成token与完成计数。
3. 根据`evicted_block_ids`从缓存树删除旧物理块。
4. 用`cur_used_block_ids`更新列出请求的累计物理块列表。
5. 用`new_block_hashes_ids`将该请求本地token哈希的对应段插入缓存树。
6. 应用LMetric增减，随后通知可选simulator。
7. 向客户端转发token与完成通知，再处理frontend abort、backend fault及终止ACK清理。

因此迁移时至少有三种“顺序”要保留：步与步的顺序、同一步删除/新增的顺序、状态更新与向客户端通知的顺序。只把JSON字段补齐，不足以保证这些关系。

## 6. 新发现：prefill的静态算术不能直接闭合

这是对**所选两个代码版本的静态检查发现**，不是NPU实测，也不是对作者论文实验的判断。假定：一个请求、一次完成prefill、首次输出未结束、无抢占/取消，隔离掉其他请求；prompt长度L、预测命中H_pred、实际命中H_actual。

Rust接纳：

```text
P_start = L - H_pred
```

作者新请求分支：本步新算`L-H_actual`，而上报的prefill又加回已命中部分：

```text
D_yaullm = (L - H_actual) + H_actual = L
```

Rust在PREFILL非终止分支再增加命中差值扣减：

```text
C_rust = H_actual - H_pred
P_after = (L - H_pred) - D_yaullm - C_rust = -H_actual
```

举例：输入64 token，预测与实际都命中32 token。

| 步骤 | 数值 |
|---|---:|
| Rust接纳时增加P | 64-32=32 |
| 引擎本步新计算 | 32 |
| yaullm实际公式上报D | 32+32=64 |
| Rust命中差值修正C | 32-32=0 |
| 静态推导剩余P | 32-64-0=-32 |
| 按已完成prefill独立预期 | 0 |

该路径应当先用最小CPU事件重放核实，再核对作者配套router分支和实验版本。此处不能把旧字段当作“纯未命中计算量”原样移植到当前引擎，也不能因为发现问题就直接改某一端。

当前prefill是isize；`overflowing_sub`检查算术溢出不等于检查非负，release的isize saturating_sub也不自动钳到0。用于评分的`queued_tokens()`另有`.max(0)`，可能使负账本在评分时被掩盖。因此此前构造快照公式测试没有覆盖这个反馈问题。

另外PREFILL即结束的请求走不同分支，命中差值修正被跳过；取消/抢占也需要独立账本，不能用上式代替所有路径。

关键代码：[接纳扣预测命中](/root/lmetric-repro/blitz-router/router/src/scheduler/policies/dsl_runtime.rs:532)、[作者首次上报加实际命中](../results/yaullm_audit/scheduler.py:535)、[Rust基础扣减](/root/lmetric-repro/blitz-router/router/src/engine/colocation.rs:467)、[Rust再扣差值](/root/lmetric-repro/blitz-router/router/src/engine/colocation.rs:521)、[评分钳零](/root/lmetric-repro/blitz-router/router/src/scheduler/policies/dsl_runtime.rs:137)。

## 7. 缓存路径的具体例子

假定一个请求的token在Rust算出的完整块哈希依次是h0,h1,h2,h3；引擎占用物理ID是[2,3,4,5]，前两块命中已有缓存。

本步上报：

```json
{
  "cur_used_block_ids": {"101": [2, 3, 4, 5]},
  "new_block_hashes_ids": {"101": [4, 5]},
  "evicted_block_ids": []
}
```

Rust先保存完整物理序列，再定位[4,5]对应的第2、3块，把(h2→4,h3→5)插入缓存树。它不是读取`new_block_hashes`中的后端整数直接建主缓存树。

若后续物理块4被其他内容复用，SSE需先报告`evicted_block_ids:[4]`，Rust删除旧映射，然后才能建立新内容→4的映射。完成一个请求不应触发删除所有这些块，否则会丢失跨请求前缀复用。

注意：`new_block_hashes_ids`的顺序必须能在完整占用列表中定位。当前`get_onto_hashes`不匹配会返回Err，调用方会panic。多KV组flatten、丢失占用更新、缺少首步token或重复事件都会破坏这一假设。

## 8. 传输和控制路径还查到了什么

- **必须full模式**：作者minimal模式把prefill和缓存/抢占/取消字段填0或空数组。Serde仍可能正常解析，但不能支持真实LMetric状态与缓存跟踪。
- **先订阅再发请求**：metrics_queue在report_metrics时才创建；队列不存在时report_once_metric直接返回。当前队列无界且不是多订阅者广播池；多个订阅者可能竞争消费。
- **空token步会遇到过滤**：core_client只把有outputs或scheduler_stats的消息入队；作者通过补空EngineCoreOutputs并加stats让部分prefill步通过。仅缓存变化/空闲重置等仍需逐场景证明不会被丢掉。
- **同步补丁不是异步补丁**：作者同步core.step附加latency、step_id和缓存字段；检查到step_with_batch_queue另有执行路径，没有相同附加块，不能默认其兼容。
- **取消发起不完整**：当前Rust HTTP+SSE的`abort_request`实现是no-op；frontend断开后进入异常跟踪不等于已经给引擎发出abort。`aborted_requests`是后端通知，不能替代发起取消的通道。
- **step_id不能保证可靠传输**：主colocation只写日志，没有按序号防重扣减或自动补发；重连/重启需要单独设计恢复。
- **附加字段未必被消费**：`ttft`被忽略，`op_exec_log`作者没发，`prefill_token_budget`当前没有实际计算读取点。采集优先级要由真实消费者决定，不应把所有字段都当作LMetric直接输入。

## 9. 本轮实际CPU检查及完成边界

脚本：[audit_yaullm_field_contract.py](../scripts/audit_yaullm_field_contract.py)。运行：

```bash
python3 scripts/audit_yaullm_field_contract.py
```

脚本从固定作者文件抽取并执行原始纯序列化函数，不导入torch/vLLM模型引擎：

- full：核对毫秒转换、请求ID转int、字段改名、hash拆分和物理ID映射。
- minimal：确认prefill/缓存确实被清空，不能当完整反馈。
- 空outputs的prefill消息：确认序列化层保留本步prefill增量。

3个CPU序列化用例通过。另保存4组受限静态prefill算术结果，包括命中0、预测=实际、预测偏小与偏大；它们是来源受约束的算术推导，**没有执行完整Scheduler、Rust事件循环或真实推理**。

产物：[checks.json](../results/yaullm_audit/field_contract/checks.json)、[prefill_arithmetic.json](../results/yaullm_audit/field_contract/prefill_arithmetic.json)、[full.json](../results/yaullm_audit/field_contract/full.json)、[minimal.json](../results/yaullm_audit/field_contract/minimal.json)、[empty_prefill.json](../results/yaullm_audit/field_contract/empty_prefill.json)。

已完成：每个SSE字段的产生、改名、传输和当前Rust消费路径，以及主要版本/语义风险的静态梳理。

仍未完成且可先不上卡继续做：用真实Rust消费者最小重放上述命中例子、分块prefill、单token结束、抢占/取消/重复事件，确认剩余账本；核实作者配套实验版本；确定修复或兼容转换方案。完成后才进入当前Ascend引擎的最小移植和单卡真实验证。
