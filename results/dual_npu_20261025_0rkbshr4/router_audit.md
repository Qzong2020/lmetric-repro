# 生产双后端审计

- `main.rs` 从 `Vec<String>` 的 client-config 为每个 URI 创建 `VllmClient` / `VllmEngineClient`。
- `Infer::new` 以 backend 数创建独立 `Arc<Mutex<ScheduleContext>>`，内有 LMetric、PrefixBlockHash、KvJournalState。
- `start_vllm_colocation_event_loop` 每个 backend 创建 work loop；take_step_receiver 每实例独占 SSE stream 与 MetricSequence（epoch、step、mutation watermark）。两个 `/v1/metrics` 可并行消费；EventHub 的单订阅限制是每个引擎，不是整个 Router。
- KvJournalState 的物理映射/requests HashMap 属于该副本；物理 ID 数值及计数器可以跨副本重叠。
- DSL capture_observations 各取独立 bs / waiting_reqs / prefill_tokens / all_tokens / cache epoch / cache hits。root_target 过滤失效状态，apply_default_after 在接纳锁内重检 routable、必要时重算命中。
- LMetric：min `(max(0, prefill_tokens) + input_tokens.len - hit_blocks*block_size) * (bs+1)`，未命中 token 用 saturating_sub。waiting 和 all_tokens 不单独进入 LMetric 分数；waiting 的 prompt 工作量包含于 prefill，bs 包含等候与运行请求。平分取先出现的副本。
- 工作循环通过本副本 VllmClient 转发预分词 token ID，并带全局 request ID；SSE 对账与 completion_event_loop 产生前端 token/结果。
- SSE 断流/协议错误锁存本副本失效并清树。已在 CPU 回归中发现：completion receiver 退出后 idle work loop 对关闭 channel 会 panic；先保存失败，再改为正常 break。不声称实现了在途取消/恢复。
- max_num_seqs=1 是各个引擎 scheduler 的单实例运行限制；没有限制 URI 列表长度。当前原型允许等候，两个后端各自1条运行；没有提高单实例 max_num_seqs。
- 基线选项目现有 `join-shortest-weight-q`：min `3*waiting+bs`。原 journal guard 仅允许 lmetric-q，先失败测试，再只增加该基线的明确许可；radix / 无 simulator / 协议连续性要求保留。未更改 LMetric 公式或 KV journal 语义。
- 新增 CANDIDATE / KV_STATE 为 debug 日志；记录生产观察与应用后权威物理表，而非另写调度器。新增 prepare / completion apply 时间 histogram，policy schedule 原本已有 histogram。所有详细观测开销计入本轮端到端时间；独立增量开销未测。
