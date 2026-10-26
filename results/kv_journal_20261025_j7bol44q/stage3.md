# 阶段三：SSE、Rust缓存树与失效保护

新版本SSE检查epoch、step_id和mutation_seq，重复/缺失/乱序/能力降级锁存失效。消费循环先验证并暂存物理映射和占用，再发布缓存视图；错误发生在扣账前时不应用该步。DSL候选剔除失效副本，并在接纳锁内重检。

已独立复现并修复两类问题：radix父块连带剪枝触发错误计数断言；同规模重建内部tree epoch重复导致旧命中快照仍被使用（cache_epoch_red.log中128误判为0命中）。后者改为使用连续mutation_seq作为策略缓存版本。

最终当前回归：Python14通过，原请求Python9通过，Router41通过/3明确忽略，radix17通过，观测脚本1通过。新增真实NPU用例在常规cargo test中明确ignored，由有界运行器单独执行，不把ignored算通过。进入最终同步和独立异步KV上卡验证；完整Router binary已离线构建。
