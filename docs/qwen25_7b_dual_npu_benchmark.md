# Qwen2.5-7B-Instruct × 双 NPU：有界性能对照

日期：2026-10-26（UTC）。前置正确性门禁：[同步/异步验证](qwen25_7b_dual_npu_validation.md)，机器可读门禁 [`correctness.json`](../results/dual_npu_20261025_hyjc_9nd/correctness.json)。原始结果根：[`results/dual_npu_20261025_hyjc_9nd/`](../results/dual_npu_20261025_hyjc_9nd/)。

## 1. 范围与结论

**本轮建立了 Qwen2.5-7B-Instruct 上可核验的有界流式测量，并完成 `lmetric-q` 与 `join-shortest-weight-q` 的配对对照。** 每个策略 2 个独立会话 × 每会话 3 类负载 × 8 请求 × 2 轮 = 96 条测量；两策略共 192 条测量 + 8 条 pilot。全部 HTTP 200，逐条真实后端、评分、缓存、生成 token 与账本对账通过，0 失败、0 超时。

| 负载 | 策略 | 请求/s | 输出 token/s | TTFT 均值 ms | TPOT 均值 ms | 每 8 请求实际命中 | 实际 Prefill | 两 replica 分布 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cache_friendly | lmetric-q | 1.145 | 9.157 | 278.31 ± 27.77 | 59.28 | 1536 | 968 | 7 / 1 |
| cache_friendly | join-shortest-weight-q | 1.226 | 9.809 | 122.66 ± 2.13 | 55.96 | 1280 | 1224 | 4 / 4 |
| mixed | lmetric-q | 1.225 | 9.798 | 140.22 ± 5.41 | 57.86 | 1024 | 1480 | 4 / 4 |
| mixed | join-shortest-weight-q | 1.218 | 9.741 | 179.89 ± 2.98 | 55.59 | 256 | 2248 | 4 / 4 |
| low_hit | lmetric-q | 1.203 | 9.627 | 196.70 ± 7.47 | 57.42 | 0 | 2504 | 4 / 4 |
| low_hit | join-shortest-weight-q | 1.204 | 9.629 | 192.84 ± 3.82 | 55.50 | 0 | 2504 | 4 / 4 |

表中均值 ± 样本标准差来自 **4 个独立重复**（2 个会话 × 2 轮）。这是小样本描述统计，不做显著性声明；吞吐量是固定稀疏到达轨迹下的完成速率，包含到达间隔，不是饱和容量。

**可复述的行为差异：**

- `cache_friendly`：LMetric 把 7/8 请求集中到已缓存前缀的副本，实际命中从 1280 提到 1536、Prefill 工作量从 1224 降到 968；代价是平均 TTFT 122.66 → 278.31 ms。**负载集中问题在 7B 上仍然存在。**
- `mixed`：LMetric 命中 1024（基线 256），Prefill 1480（基线 2248），平均 TTFT 140.22（基线 179.89 ms）。
- `low_hit`：两策略命中都为 0、Prefill 都为 2504，TTFT 196.70 vs 192.84 ms，差异落在轮次/顺序漂移范围内。
- TPOT 上 LMetric 略高（57.4–59.3 vs 55.5–56.0 ms）。差异很小且与策略/分布混杂，不能归因为策略本身的计算开销。

## 2. 实验控制

- 同一模型 `/models/Qwen2.5-7B-Instruct`，同一对物理 NPU（本轮为物理 0/7，逻辑 0/1），TP=PP=DP=1，同步 eager。
- 每实例 `max_num_seqs=1`、`max_model_len=1024`、`max_num_batched_tokens=128`、`gpu_memory_utilization=0.40`、`num_gpu_blocks_override=8`、block/hash 128、prefix caching、chunked prefill。除 `gpu_memory_utilization` 从 0.20 提高到 0.40（7B 权重 14.21 GiB 必需）外，与原双 NPU 实验一致。
- 输入用 Qwen2.5 tokenizer 重新生成：每条 313 token、输出固定 min=max=8。四会话的 `trace_definition.json` 由独立审计逐字节比较相同。
- 三类轨迹与原实验一致：`cache_friendly=[0,1,0,0,0,0,0,0]`、`mixed=[0,1,1,0,2,3,0,1]`、`low_hit=[10..17]`；固定到达偏移 `0,0.04,2,2.04,4,4.04,6,6.04` 秒。
- 每条轨迹开始前对两个引擎执行 idle reset，并等到 SSE/Rust 确认空缓存；每会话每引擎 7 次独立 control reset（审计核验）。
- 每策略先跑 2 条流式 pilot 验证计时语义与预热，再执行 48 条正式测量。正式测量不含 pilot。
- 策略顺序：先按记录种子 `4259862290` 随机得到 `lmetric-q → join-shortest-weight-q`（第一对）；由于随机结果与旧实验恰好同序，追加一对反向顺序 `join-shortest-weight-q → lmetric-q`。两对均为独立引擎会话、独立日志与账本。顺序记录：[`benchmark_ordering.json`](../results/dual_npu_20261025_hyjc_9nd/benchmark_ordering.json)。
- 同一模型下两策略的请求内容、Token IDs、输出长度、采样参数与到达计划完全相同；Router 对两种策略都通过同一 KV journal/缓存状态链路。

**未控制项：** 请求生成在主机线程中执行，实测到达迟到大多 <1.2 ms，但 `benchmark_baseline_attempt1` 出现一次 51.1 ms 的离群迟到；两对会话之间还存在环境漂移（见 §4）。

## 3. 测量定义

- **TTFT**：发送开始到第一个真实 token delta 的客户端接收时间；排除角色宣告、HTTP 头与 finish 事件。每条请求核验恰好 8 个 token delta，且与真实 SSE 的 8 个生成 token 一一对应。
- **TPOT**：`(第 8 个 token delta 接收时间 − 第 1 个)/(8−1)`，是客户端观测的平均 token 间隔，包含传输与 Router 开销，**不是设备 kernel 时间**。
- **请求/token 吞吐**：`8 / (末 DONE − 首发送)` 与 `64 / 同一区间`，包含固定到达间隔。
- **实际命中**：来自真实首 PREFILL 的 `hit_token_cnt`；**Prefill 工作量**来自完整 journal 归属（包含 `outputs=[]` 的 chunked prefill）。预测命中不冒充实际命中。
- **Router policy / KV prepare / step apply**：Router 进程内 host histogram 的 sum/count 差；不含命令队列等待，不含 SSE 接收等待；不是设备 kernel 时间。
- **未测量**：事件采集/日志/journal 的独立启停增量开销。debug 观测全程开启，端到端指标包含其影响，但本轮没有禁用对照，因此不给出该项数字。

## 4. 逐重复原始值

每个单元格是该策略该负载 4 个独立重复的值（顺序：pair1 两轮，pair2 两轮）。完整原始记录见各会话 `benchmark.jsonl`。

| 指标 | 负载 | lmetric-q | join-shortest-weight-q |
| --- | --- | --- | --- |
| TTFT ms | cache_friendly | 306.18, 298.15, 255.59, 253.29 | 122.77, 119.68, 124.63, 123.56 |
| TTFT ms | mixed | 145.95, 143.74, 135.40, 135.81 | 176.62, 178.24, 181.64, 183.07 |
| TTFT ms | low_hit | 205.65, 199.91, 191.87, 189.38 | 187.87, 191.87, 195.26, 196.35 |
| TPOT ms | cache_friendly | 65.46, 63.25, 54.74, 53.65 | 56.39, 54.05, 56.99, 56.42 |
| TPOT ms | mixed | 60.51, 60.65, 56.08, 54.22 | 55.17, 54.94, 56.04, 56.22 |
| TPOT ms | low_hit | 59.66, 60.27, 55.17, 54.56 | 54.24, 55.00, 56.31, 56.44 |
| 请求/s | cache_friendly | 1.126, 1.131, 1.159, 1.162 | 1.230, 1.231, 1.219, 1.224 |
| 输出 token/s | cache_friendly | 9.006, 9.046, 9.276, 9.300 | 9.844, 9.849, 9.751, 9.794 |
| 实际命中 | cache_friendly | 1536 ×4 | 1280 ×4 |
| 实际 Prefill | cache_friendly | 968 ×4 | 1224 ×4 |

**顺序/时段效应是真实存在的：** `cache_friendly` 的 LMetric TTFT 在第一对为 306/298 ms，在第二对（先跑基线、后跑 LMetric）为 256/253 ms；TPOT 同步从 65/63 降到 54.7/53.7 ms。基线在两对中稳定（TTFT 120–125 ms，TPOT 54–57 ms）。因此绝对值不能脱离顺序解释，但“LMetric 集中缓存流量、提高该负载 TTFT、降低 Prefill”在两对中方向一致。

### Host 时间（4 个重复合计）

| 策略 | policy µs / 调用 | KV prepare µs / 调用 | step apply µs / 调用 |
| --- | --- | --- | --- |
| lmetric-q | 319.69 / 96 | 82.38 / 964 | 341.49 / 964 |
| join-shortest-weight-q | 339.14 / 96 | 84.91 / 1008 | 334.87 / 1008 |

policy 每请求一次；prepare/apply 每个 SSE envelope 一次，不能误除到单请求。这些 host 时间相对 0.5–1.2 s 的端到端请求耗时占比很小；本轮没有测到“模型变大后 Router 占比下降”的证据，只是占比本来就很低。

### 到达偏差与失败

| 会话 | 最大迟到 ms | 平均迟到 ms | 测量请求 | 失败/超时 |
| --- | --- | --- | --- | --- |
| benchmark_lmetric_attempt2 | 0.40 | 0.14 | 48 | 0 / 0 |
| benchmark_baseline_attempt1 | 51.07 | 1.45 | 48 | 0 / 0 |
| benchmark_baseline_attempt2 | 0.83 | 0.16 | 48 | 0 / 0 |
| benchmark_lmetric_attempt3 | 1.20 | 0.18 | 48 | 0 / 0 |

## 5. 模型变化带来的影响（阶段 5）

以下区分三类陈述：**[本轮实测]**、**[与旧实验的描述性比较]**、**[待重复验证的推断]**。

### 5.1 更大模型是否改变 Prefill 耗时与缓存命中的绝对收益？

**[本轮实测]** 在本工作负载（313 token 输入、256 token 可命中、8 token 输出）下，命中把 Prefill 从 313 降到 57 token。`mixed` 中 LMetric 相对基线的 TTFT 差为 −39.7 ms，`cache_friendly` 为 +155.7 ms（集中流量的代价），`low_hit` 为 +3.9 ms。

**[与旧实验的描述性比较]** 旧 Qwen3-0.6B 的对应差值（同样按 LMetric−baseline）为 `mixed` −45.2 ms、`cache_friendly` +166.4 ms、`low_hit` +2.2 ms。**量级相近，没有观察到“7B 让缓存命中的绝对收益显著变大”。**

**推断：** 本轮端到端时间主要不是模型计算时间。依据是 7B 的 TPOT（55–59 ms）不高于旧 0.6B（约 63 ms），TTFT 也在同一量级（100–300 ms）。但这只是本工作负载/本环境的观测，不能外推为“模型规模与延迟无关”。

### 5.2 预测命中与实际命中是否一致？

**[本轮实测]** 是。192 条测量中每条都核验：决策时的预测命中、实际首 PREFILL 命中、以及 `实际 Prefill = 输入长度 − 实际命中`。聚合层面两策略在 `cache_friendly` 分别为 1536/1536 与 1280/1280，`mixed` 为 1024/1024 与 256/256，`low_hit` 均为 0/0。队列中的请求可能看到更新后的缓存，本轮驱逐/并发条件下未出现预测与实际的偏差。

### 5.3 缓存友好负载下负载集中是否仍然存在？

**[本轮实测]** 是。LMetric 每个轨迹 7/1 分布（4 个重复全部相同），基线 4/4；LMetric 平均 TTFT 278.31 ms，基线 122.66 ms。旧 Qwen3 实验也是 7/1 vs 4/4、302.39 vs 135.98 ms。**这是跨模型稳定复现的行为，不是 7B 新出现的现象。**

### 5.4 混合负载下 LMetric 相对基线是否变化？

**[本轮实测]** 方向不变：LMetric 命中 1024 vs 256、Prefill 1480 vs 2248、TTFT 140.22 vs 179.89 ms。旧 0.6B 为 154.61 vs 199.81 ms。本轮两者都低约 15–20 ms，但环境/顺序不同，不能归因于模型。

### 5.5 低命中负载下差异是否仍小？

**[本轮实测]** 是。命中都为 0、Prefill 都为 2504；TTFT 196.70 vs 192.84 ms，差异小于 `cache_friendly` 的顺序漂移（约 50 ms），也小于 LMetric 自身的轮次标准差（7.47 ms）与 baseline 的 3.82 ms 之和的量级。**不宣称两策略在此负载有可分辨差异。**

### 5.6 更大模型是否让 Router 额外开销占比下降？

**[本轮实测]** Router policy 约 320–339 µs/请求，prepare 约 82–85 µs/envelope，apply 约 335–341 µs/envelope；相对 0.5–1.2 s 的端到端耗时，policy 决策占比约 0.03%。**占比本来就很低，没有测到“因模型变大而下降”的变化。** 这些是 host 时间，未与设备 kernel 时间混算。

### 5.7 变化是否可能来自显存、KV 配置、到达偏差或顺序？

**[本轮实测/推断]** 是，至少以下混杂因素必须保留：

1. **显存与 KV 配置**：`gpu_memory_utilization` 必须从 0.20 提到 0.40；KV 池仍被 `num_gpu_blocks_override=8` 固定为 1024 token 容量，因此每 block 的 KV 体积（7B 更大）不改变块数，但改变了每块 HBM 占用。没有做显存容量/块数扫描。
2. **设备与环境**：容器重建后物理卡对从 0/6 变为 0/7；旧 Qwen3 结果与新结果不在同一容器会话中。
3. **顺序/时段**：两对反向顺序显示 `cache_friendly` 的 LMetric TTFT 有约 50 ms 的会话级漂移；虽然方向不变，绝对值不可直接跨会话比较。
4. **到达偏差**：一次 51 ms 离群迟到；其余 <1.2 ms，不足以解释 40–160 ms 的策略差。
5. **模型 tokenizer**：输入按 Qwen2.5 tokenizer 重新生成；虽然 token 数固定为 313，但 token 内容与旧实验不同，不能把全部差异归因于参数规模。

## 6. 与旧 Qwen3-0.6B 结果的描述性对照

| 负载 | 指标 | Qwen3-0.6B（旧） | Qwen2.5-7B（本轮） |
| --- | --- | --- | --- |
| cache_friendly | LMetric / baseline TTFT ms | 302.39 / 135.98 | 278.31 / 122.66 |
| cache_friendly | 命中 / Prefill（LMetric） | 1536 / 968 | 1536 / 968 |
| cache_friendly | 分布 LMetric / baseline | 7/1 / 4/4 | 7/1 / 4/4 |
| mixed | LMetric / baseline TTFT ms | 154.61 / 199.81 | 140.22 / 179.89 |
| mixed | 命中 LMetric / baseline | 1024 / 256 | 1024 / 256 |
| low_hit | LMetric / baseline TTFT ms | 219.96 / 217.79 | 196.70 / 192.84 |
| 全部 | TPOT ms（约） | 62–63 | 55–59 |

**这些数字不能直接相减当作“模型规模效应”。** 两次实验的容器、物理卡对、时段和测试框架都不同；本轮还把物理设备对参数化并在 0/7 上完成。可以确认的是：**路由决策模式与命中/Prefill 账本在换模型后完全保持**；延迟的绝对值变化需要同容器、同卡、交替模型的专门实验才能归因。

## 7. 证据与复跑

- 汇总与逐重复：[`benchmark_qwen25_7b.json`](../results/dual_npu_20261025_hyjc_9nd/benchmark_qwen25_7b.json)。
- 200 条请求级复核（全部 200、预测命中=实际命中、8 token delta）：[`benchmark_request_audit.json`](../results/dual_npu_20261025_hyjc_9nd/benchmark_request_audit.json)。
- 两对独立审计：[`benchmark_comparison_pair1.json`](../results/dual_npu_20261025_hyjc_9nd/benchmark_comparison_pair1.json)、[`benchmark_comparison_pair2.json`](../results/dual_npu_20261025_hyjc_9nd/benchmark_comparison_pair2.json)、[`inspect_pair1.log`](../results/dual_npu_20261025_hyjc_9nd/inspect_pair1.log)、[`inspect_pair2.log`](../results/dual_npu_20261025_hyjc_9nd/inspect_pair2.log)。
- 每会话：`requests.jsonl`（请求、完整 token 时钟、原始 chunk）、`reconciled.jsonl`（候选评分、选择、实际后端、命中、Prefill 工作量）、`benchmark.jsonl`（每轨迹到达计划、指标、host histogram 差）、`engine_0/1.log`、`gateway.log`、`trace_definition.json`、`*_metrics_before/after.txt`、`verified_state.json`。
- 独立审计核验：每会话 50 条唯一请求、流式文本与真实 token 整体解码一致、8 个 token delta、7 次/引擎空缓存控制、逐步物理表/占用表一致、最终双账本全 0、两对轨迹定义相同。

复跑（设备恢复后，使用独立新会话名）：

```bash
cd /root/lmetric-repro
export LMETRIC_MODEL=/models/Qwen2.5-7B-Instruct LMETRIC_GPU_UTIL=0.40 \
       LMETRIC_NUM_GPU_BLOCKS=8 LMETRIC_MAX_MODEL_LEN=1024 \
       LMETRIC_PHYSICAL_DEVICES=0,7
.venv/bin/python scripts/dual_npu/benchmark.py <结果根> lmetric-q <新会话名>
.venv/bin/python scripts/dual_npu/benchmark.py <结果根> join-shortest-weight-q <新会话名>
BENCH_OUT=<新对比文件名>.json .venv/bin/python scripts/dual_npu/inspect_benchmarks.py \
  <结果根> <lmetric会话> <baseline会话>
```

## 8. 限制

- 每种策略 4 个独立重复、每重复 8 请求；只有描述统计，没有置信区间或显著性检验。
- `max_num_seqs=1`、8-block 人工小池、正常 `length` 结束、回环客户端、eager、详细 debug 观测。
- 请求吞吐是稀疏到达下的完成速率，不是饱和容量；固定到达计划使两策略可比，但不能回答高并发行为。
- 没有测事件采集的独立增量开销；没有设备 kernel 时间；TPOT 含传输与 Router 开销。
- 没有覆盖 preemption/resume/abort、EOS/stop/取消、在途故障恢复、优雅关闭。
- 一次 51 ms 到达离群与会话级顺序漂移限制了细小差异的解释；`cache_friendly` 的策略差远大于这些噪声，`low_hit` 的差异则不能区分。
- 跨模型数字只作描述性对照，不作规模因果推断。
