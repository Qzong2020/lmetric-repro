# 双 NPU 有界验收工具

全部从 `/root/lmetric-repro` 使用 `.venv/bin/python`，复用既有安装和源码 PYTHONPATH。

1. `snapshot.py` 创建唯一 `results/dual_npu_...` 根目录，保存 incoming Git 工作和历史产物哈希。
2. 为已安装 OpenSSL 库在结果目录的 `build_lib` 建立临时 linker symlink；`build.py <结果根>` 离线测试并构建两个独立 policy binary。
3. `test_driver.py` 检验独立状态核验器；Rust `dual_cpu` 测试生产策略、journal、独立序号/映射、双 completion loop 和断流退出。
4. `experiment.py <结果根> sync <唯一会话名>`；同步通过后，再用 `async` 和新名称运行。启动每引擎240秒、每HTTP请求30秒、每场景不超过16请求；整个 correctness session 为13请求。
5. `verify_correctness.py <结果根> <会话名>` 离线对账，创建新的 audit 子目录，只有所有检查通过才写 `verified.json`。保留原 driver failure。受控停机的终止错误帧单独分类，不视为普通 completion；仅明确允许停止的 A 可以在离线审计中出现此帧。
6. 手工将已通过会话名写为结果根 `correctness.json` 的 `sync`/`async`。`benchmark.py <结果根> <policy> <唯一会话名>` 要求两项 `verified.json` 都通过，先流式pilot，再3种×8请求×2重复；2条pilot+48条测量。策略为 `lmetric-q` / `join-shortest-weight-q`。
7. `inspect_benchmarks.py <结果根> <lmetric会话> <baseline会话>` 复核原始记录/轨迹/重置与计时。
8. `finalize.py <结果根>` 复核历史证据和所有 HEAD、既有源码文件，保存最终补丁、NPU状态及产物哈希。

本环境 `npu-smi info -m` 实测物理0/6对应 logical0/1。每个子进程 `ASCEND_RT_VISIBLE_DEVICES` 使用0/1；不得把显示的物理6直接填入 runtime 可见设备列表。`npu_loaded_1.txt` 必须与实际 `VALIDATION_RUNTIME.pid` 对应到两张物理卡。

所有预热和推理均走统一生产 Router；直接后端操作仅为可经SSE核验的 idle reset。不会在断流后复用引擎会话，不会连接第二个SSE订阅者。只终止本轮 `start_new_session=True` 创建的进程组。

TTFT = 发送开始到第一个真实 token delta 接收（排除角色宣告）；TPOT = 首末token delta接收时间差/(生成token数−1)。核验每个delta和真实SSE token数一一对应，原始chunk内容和单调时钟均保存；HTTP header/角色chunk不能冒充TTFT。吞吐量区间为首发送到末DONE，包含固定到达间隔，不是饱和吞吐量。

详细debug观测开启，计时包含其影响。prepare/apply histogram是已明确边界的host执行时间；不等于通过启停对照测得的事件采集增量开销。该项未测，不推算。
