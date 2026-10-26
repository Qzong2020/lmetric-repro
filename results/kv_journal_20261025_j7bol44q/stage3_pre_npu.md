# 阶段三：Rust消费（上卡前）

已扩展request-kv-v1、mutation序号与epoch校验；全局物理映射以backend hash核验旧身份，用真实token proof重建本地xxh3链，不把两端hash当成相等。暂存整批更新后原子发布缓存树，请求占用表与缓存映射分离；失效清树并在DSL候选和接纳锁内重检。

父块单独删除的独立回归在修复前真实panic（radix_red_attempt2.log）；原实现连带剪枝数可大于显式删除ID数，已修正计数断言而不clamp。物理权威镜像保留孤儿子块，父前缀恢复后可重建可命中树。

当前Router lib回归37通过/2明确忽略，独立radix17通过。继续增加新协议负例与真实消费循环失效用例，再进入单卡真命中/驱逐/reset实验；此报告尚无NPU KV结果。
