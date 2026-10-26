# 阶段二：引擎侧 journal

完成有界 KVJournal 与成功 insert/pop/reset、完整占用和 free 生命周期钩子；原生 take_events 未被额外消费。复用 EngineCoreOutputs.lmetric_event / ZMQ / AsyncLLM / EventHub，增加 idle reset 控制 envelope，SSE 队列另有32MiB总字节预算。request-v1仍独立保留。

先运行测试得到 ModuleNotFoundError 的未实现失败证据 python_red.log，再接生产逻辑。最终 Python 13项通过（python_final.log）；原请求原型9项回归通过（request_python_regression.log），其中包括原生跨进程ZMQ。均为CPU合成输入，未声称是NPU验证。本阶段没有改缓存分配/释放/驱逐策略或安装环境。进入Rust端扩展和真实上卡前的原子性/失效测试。
