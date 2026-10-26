# 阶段四：真实同步KV生命周期

最终源码/二进制、单张910B2、本地Qwen3-0.6B、同步eager、block/hash=128、单FullAttention group、真实物理池8块。每个会话最多8条推理请求，单条313输入/8输出，另有一次idle reset；启动240秒、驱动180秒硬界限，退出只操作本轮进程组。

sync_final：8条真实请求成功。87个SSE envelope连续，53条mutation连续（14insert、7remove、22完整占用更新、8release、2reset其中1为空初始基线）。Rust每一步物理映射与独立读取的真实BlockPool map和实际get_block_ids占用快照完全一致（87/87）。重复前缀真实命中256；有限其它前缀驱逐后重发0；reset控制事件在下一条请求前独立到达Rust，reset后重发0。每条结束四项账本0；最终占用表空、可释放块7、缓存map仍有2项，直接验证free≠evict。

证据：sync_final/verified.json、sse.jsonl、server.log、rust.log、run.json与NPU输出。首次sync_attempt1因独立观测器emit参数名冲突失败，已停止；修复观测器并单测后新会话成功，旧失败目录完整保留。sync_attempt2为修复策略epoch竞态前的串行预验，正式结论使用sync_final。继续独立异步验证。
