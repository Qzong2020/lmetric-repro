# 阶段四：真实异步KV生命周期

async_final：显式async_scheduling=True、enforce_eager=True，实际AsyncScheduler的MRO包含Ascend BalanceScheduler，执行step_with_batch_queue，队列容量2。8条有界真实请求成功，95个SSE envelope连续，53条mutation连续，95/95步真实BlockPool物理map和完整占用快照与Rust一致。mutation分类与同步相同；真实命中序列[0,256,0,0,0,0,0,0]，有限驱逐和idle reset后均真实未命中。所有请求结束账本0；最后执行队列空，无活动占用，物理ID复用已核验。此为元数据journal验证，不等于设备写入完成事件或性能结论。服务已停止、NPU释放。继续完整网关有限验收。
