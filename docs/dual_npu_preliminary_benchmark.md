# 双 NPU 小规模性能对照

日期：2026-10-25（北京时间）。前置 [同步/异步正确性验收](dual_npu_validation.md) 已通过，门禁保存于 [correctness.json](../results/dual_npu_20261025_0rkbshr4/correctness.json)。

## 范围和结果

**已建立可信的有界流式测量与对照链路；本结果不代表论文性能复现，也不据此宣称普遍加速。** 两策略分别独立启动同样的物理NPU0/6和模型，使用相同参数、完整journal、debug观测及生产HTTP网关。每策略2条流式pilot通过后，执行3类×8请求×2重复，共48条测量，合计96条测量+4条pilot。全部HTTP200，逐条真实后端/评分/缓存/生成token对账通过，无超时或推理失败。

| 负载 | 策略 | 请求/s | 输出token/s | TTFT均值 ms | TPOT均值 ms | 每8请求实际命中token均值 | 两轮A/B请求数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cache_friendly | lmetric-q | 1.140 | 9.117 | 302.39 | 63.42 | 1536 | {'0': 14, '1': 2} |
| cache_friendly | join-shortest-weight-q | 1.218 | 9.742 | 135.98 | 62.45 | 1280 | {'0': 8, '1': 8} |
| mixed | lmetric-q | 1.222 | 9.773 | 154.61 | 63.20 | 1024 | {'0': 8, '1': 8} |
| mixed | join-shortest-weight-q | 1.207 | 9.655 | 199.81 | 62.93 | 256 | {'0': 8, '1': 8} |
| low_hit | lmetric-q | 1.188 | 9.504 | 219.96 | 62.57 | 0 | {'0': 8, '1': 8} |
| low_hit | join-shortest-weight-q | 1.193 | 9.542 | 217.79 | 62.55 | 0 | {'0': 8, '1': 8} |

混合轨迹中LMetric减少实际prefill工作并降低本轮平均TTFT；缓存友好轨迹中，更多请求集中到A，平均TTFT反而升高。这说明命中与负载的取舍不能仅靠命中量判断。上述为小样本观测，不作统计显著性声明。

每行是两次有界重复的均值；两轮请求分布为计数求和。吞吐量是固定稀疏到达轨迹的完成速率，包含到达间隔，不是饱和容量；不使用这些小样本计算显著性或概括整体收益。

## 实验控制

- 同样两张910B2、Qwen3-0.6B、TP=PP=DP=1、同步eager；每实例max_num_seqs=1，KV block128、物理池8，7个非null块。
- LMetric最小化 `(max(0,pending_prefill)+uncached_tokens)*(bs+1)`；项目现有基线join-shortest-weight-q最小化 `3*waiting+bs`。只允许该基线通过相同journal能力门禁，未替换缓存消费逻辑。
- 每条input313 token，输出固定min=max=8；temperature等后端采样行为两策略完全相同。VllmClient沿用现有实现，未声称前端任意采样参数均已透传；本轮比较没有修改该路径。
- 每条轨迹开始前两个真实引擎各执行idle reset并等待SSE/Rust确认空缓存。模型/内核已由同样的流式pilot预热；每策略独立epoch，策略间不复用缓存。
- 固定到达偏移秒：0、0.04、2、2.04、4、4.04、6、6.04；计划和实际单调时间/迟到量全部保存。三类前缀轨迹 cache_friendly、mixed、low_hit 完整token IDs逐字节相同，独立audit验证。每轨迹8请求、两轮；没有无界并发或长时压力测试。
- 策略按LMetric后baseline运行，没有随机化顺序；两次重复不足以排除时段、温度等偏差。

## 测量定义及可核验性

TTFT=发送开始到首个真实token delta的客户端接收时间。排除最早的角色宣告、HTTP头和finish事件；每条请求均验证恰好8个token delta，与真实引擎SSE中的8个生成token一一对应。token特殊符号解码为空时仍是实际生成token事件，不把空角色chunk算首token。

TPOT=(第8个token delta接收时间−第1个token delta接收时间)/7。它是客户端观测的平均token间隔，包含传输/Router开销，不是设备kernel时长。逐个原始chunk和monotonic_ns均保留。所有测量为回环网客户端，未声称跨网络部署延迟。

请求吞吐量=8/(末DONE−首发送)；输出token吞吐量=64/同一区间。另保存输入+输出token accounting吞吐量，区分总token数和实际prompt工作量。实际缓存命中来自真实首PREFILL输出，工作量来自完整completion journal归属（包括outputs=[]的chunked prefill），没有用预测命中冒充实际命中。

Router决策时间使用原有policy schedule histogram，包含快照锁、策略与接纳账务，排除命令队列等待。新KV prepare histogram覆盖journal staging/tree rebuild；step apply覆盖completion host处理，不含SSE接收等待和最后debug状态输出。每个轨迹保存前后sum/count差及均值，不能把总计数误除到单请求。

以下host计时为48条测量的sum/count加权均值，单位µs；斜线后是调用数，prepare/apply每envelope一次，不能和每请求的policy调用混为一谈。

| 策略 | policy µs / 次数 | KV prepare µs / 次数 | step apply µs / 次数 |
| --- | --- | --- | --- |
| lmetric-q | 226.55 / 48 | 90.54 / 482 | 347.09 / 482 |
| join-shortest-weight-q | 240.98 / 48 | 96.98 / 504 | 352.72 / 504 |

**未可信测得的项目：事件采集/日志/journal的独立增量开销。** 本轮debug和真实池快照全程开启，端到端指标包含其影响；prepare/apply host时间可以实测，但没有启停对照，不能推算禁用采集后的性能。没有提供该项虚构数字。

## 原始证据和限制

[benchmark_comparison.json](../results/dual_npu_20261025_0rkbshr4/benchmark_comparison.json) 保存两轮汇总、延迟和所有host时间统计；每策略会话的 requests.jsonl 保存请求/完整token时钟/响应chunk，reconciled.jsonl 保存评分/选择/真实后端/命中/prompt工作量，benchmark.jsonl 保存每轨迹到达计划和metrics差，engine_0/1.log及gateway.log保存逐步完整证据，trace_definition.json保存相同原始输入。

- [benchmark_lmetric_attempt1/run.json](../results/dual_npu_20261025_0rkbshr4/benchmark_lmetric_attempt1/run.json)：50条真实请求，48条测量，[310, 208]帧状态对账。
- [benchmark_baseline_attempt1/run.json](../results/dual_npu_20261025_0rkbshr4/benchmark_baseline_attempt1/run.json)：50条真实请求，48条测量，[268, 272]帧状态对账。

独立验收同时核验两策略各50条唯一请求、流式文本与真实token整体解码一致、所有token时间戳、7次/后端的空缓存控制、逐步物理表/占用表一致以及最后双账本全0。

实际到达存在主机线程调度迟到，不能把计划时间当成真实时间。各策略最大迟到（ms）：lmetric-q=50.98, join-shortest-weight-q=1.03；完整逐条偏差已保存。该偏差和未随机化的策略顺序限制了细小性能差异的解释。

主要限制：max_num_seqs=1、0.6B小模型、8-block人工小池、正常length结束、回环客户端、详细debug观测、只有两轮、固定策略顺序、低负载有间隔到达。接下来应先扩展并发与协议合同，再在论文模型/缓存规模/轨迹下开展随机化、足量重复和观测开销对照，不能直接外推本表。
