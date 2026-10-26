# Qwen2.5-7B-Instruct × 双 NPU：路由正确性验证

日期：2026-10-25（实验执行，UTC）／2026-10-26（容器重启后整理）。原始结果根：[`results/dual_npu_20261025_hyjc_9nd/`](../results/dual_npu_20261025_hyjc_9nd/)。

## 1. 本轮实际结论

| 阶段 | 状态 | 结论 |
| --- | --- | --- |
| 模型与环境核验 | 通过 | 本地 Qwen2.5-7B-Instruct 完整（339 张量 / BF16 / 无量化）；vLLM-Ascend 0.23.0 与 CANN 9.1.0 加载成功 |
| 单卡加载与推理冒烟 | 通过 | 权重加载 67.6 s，Prefill/Decode、KV Cache、`request-kv-v1`、`kv-journal-v1`、Router 流式消费均正常 |
| 双卡独立部署 + 同步 eager | 通过 | 13 条真实请求，A=6/B=7；独立 epoch、物理 Block 命名空间、逐步 journal 与账本全部对账 |
| 双卡异步 eager | 通过 | 独立新会话重跑通过（`async_attempt2`），13 条请求，A=6/B=7，事件 82/91、mutation 40/41 |
| 性能对照（阶段 4） | 通过（有界） | 2 对反向顺序会话、每策略 4 个独立重复；3 类负载 × 8 请求 × 2 轮/会话，共 192 条测量 |

**结论边界：** 现有证据支持“Qwen2.5-7B-Instruct 上单/双卡真实推理、同步与异步缓存感知路由正确”，以及“在本轮 313-token / 8-token、`max_num_seqs=1`、8-block 小池的有界条件下，LMetric 与基线的行为模式”。不支持论文级性能复现或统计显著性声明。

## 2. 模型、软件与设备核验

### 2.1 模型文件（不是按目录名判断）

模型路径 `/models/Qwen2.5-7B-Instruct`，完整核验结果见 [`model_integrity.json`](../results/dual_npu_20261025_hyjc_9nd/model_integrity.json)。

- 4 个 safetensors 分片，index 声明 339 个张量、15,231,233,024 字节张量数据；分片头内 339 个张量，**缺失 0、多余 0**。
- 全部张量 `BF16`；`quantization_config` 不存在（非量化权重）。
- 关键形状与 `config.json` 全匹配：28 层、hidden 3584、28 个 Q attention head、4 个 KV head（GQA）、head_dim 128、intermediate 18944、vocab 152064、`max_position_embeddings=32768`。
- `Qwen2ForCausalLM` 在 vLLM 0.23 registry 与 vLLM-Ascend 均有注册；vLLM-Ascend 文档本身以 Qwen2.5-7B-Instruct 为示例。
- tokenizer 可加载，vocab 151665；313/700 token 的 decode→encode 往返各 5/5 通过。基准输入用该 tokenizer 重新生成，未复用 Qwen3 的 Token IDs。
- 软件：vLLM 0.23.0+empty、vLLM-Ascend 0.23.0、torch 2.10.0+cpu、torch-npu 2.10.0.post4、CANN 9.1.0、driver 26.0.rc1、Python 3.12.13。

### 2.2 设备映射

设备映射在容器重建后发生过一次变化，不能沿用旧值：

| 会话 | `npu-smi info -m` 实测 | 逻辑 ID | 进程 PID |
| --- | --- | --- | --- |
| 单卡冒烟、同步验证 | 物理 0 → logic 0；物理 6 → logic 1 | 0/1 | 183150 / 185163 |
| 异步验证、性能对照 | 物理 0 → logic 0；物理 7 → logic 1 | 0/1 | 7860 / 10113（异步） |

两个子进程始终设置 `ASCEND_RT_VISIBLE_DEVICES=0/1`；物理 ID 通过 `VALIDATION_RUNTIME` 的 PID 与 `npu_loaded_*.txt` 交叉核验，测试框架新增 `LMETRIC_PHYSICAL_DEVICES` 以记录每轮真实物理卡对，未把显示的物理 ID 直接当作 runtime 可见设备。

- 两卡各 65536 MiB HBM；同步会话加载后整卡占用 18268 / 18230 MiB（基线约 3.4 GiB + 引擎 14853 MB）。
- 异步与性能会话在物理 0/7 上，单个引擎进程 HBM 14853 MB。

## 3. 配置与唯一变更

保持原双 NPU 实验配置：

- TP=PP=DP=1；`--enforce-eager`；单 FullAttention KV group；block/hash size=128。
- `max_num_seqs=1`、`max_model_len=1024`、`max_num_batched_tokens=128`、`num_gpu_blocks_override=8`（1 个 null + 7 个非 null，总 1024 token 容量）。
- 开启 prefix caching 与 chunked prefill；Router 为生产二进制，`--chat-template-mode none`，预分词 Token IDs 走 `/v1/completions`。

**唯一必需变更：`gpu_memory_utilization` 0.20 → 0.40。** 0.20 × 60.96 GiB ≈ 12.2 GiB 小于 14.21 GiB 的权重，按原值无法加载。0.40 下实测：权重 14.21 GiB、峰值激活 0.03 GiB、非 torch 0.02 GiB；可用 KV 10.13 GiB 被 `num_gpu_blocks_override=8` 截为 8 个 block。未降低精度、未替换权重、未改系统依赖。

## 4. 单卡冒烟（阶段 1）

证据：[`smoke_single_attempt3/smoke.json`](../results/dual_npu_20261025_hyjc_9nd/smoke_single_attempt3/smoke.json)。

- 加载到 `/health` 200：67.6 s；HBM 18268 MiB。
- 非流式请求（313 token 输入、8 token 输出）：HTTP 200，`usage={prompt_tokens:313, completion_tokens:8, total_tokens:321}`；实际命中 0，Prefill 工作量 313。
- 流式请求（相同输入）：实际命中 256 token，Prefill 工作量 57；TTFT 84.8 ms，TPOT 58.2 ms，恰好 8 个真实 token delta。
- 两条请求生成 Token IDs 完全一致（`[10339,1246,264,1681,16211,1526,264,4128]`），流式文本与整段 decode 一致。
- `request-kv-v1`、`kv-journal-v1`、SSE、Rust 路由与用量账本均工作。

**异常记录：** 单卡首条请求耗时 28.5 s，第二条同前缀请求 TTFT 84.8 ms；判断为一次性 kernel/JIT 预热成本。双卡会话的 preflight 首请求为 0.94 s，未复现该量级，说明该异常不是稳定的模型级成本；仍据此放宽了请求超时。

## 5. 双卡同步验证（阶段 2/3）

### 5.1 部署

- 两个独立引擎，端口 18181/18182；统一生产 Router 18184，`clients.json` 为两个真实 URI。
- 引擎顺序启动，避免启动期显存竞争；每个实例独立 KV Cache、engine_epoch、SSE 事件流和日志。
- 在任何测试请求之前先建立两路 SSE 订阅，再由受控 idle reset 得到可核验的空缓存。

### 5.2 逐请求对账（同步，13 条）

预期/实际选择、预测命中、bs、pending prefill、评分、真实命中与真实工作量全部一致。分数为生产 `CANDIDATE` 观测值。

| 场景 / label | req | A/B 预测命中 | A/B bs | A/B pending prefill | A/B 分数 | 选择 | 实际命中 | 实际 Prefill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| preflight_first | 0 | 0/0 | 0/0 | 0/0 | 313/313 | A | 0 | 313 |
| preflight_second | 1 | 0/0 | 1/0 | 313/0 | 1252/313 | B | 0 | 313 |
| A_only_target | 2 | 256/0 | 0/0 | 0/0 | 57/313 | A | 256 | 57 |
| controlled_B_warm_first | 3 | 0/0 | 0/0 | 0/0 | 700/700 | A | 0 | 700 |
| controlled_B_warm_second | 4 | 0/0 | 1/0 | 700/0 | 2026/313 | B | 0 | 313 |
| B_only_same_target | 5 | 0/256 | 0/0 | 0/0 | 313/57 | B | 256 | 57 |
| controlled_A_warm_first | 6 | 0/256 | 0/0 | 0/0 | 700/444 | B | 256 | 444 |
| controlled_A_warm_second | 7 | 0/256 | 0/1 | 0/444 | 313/1002 | A | 0 | 313 |
| both_cached_load_first | 8 | 256/256 | 0/0 | 0/0 | 57/57 | A | 256 | 57 |
| both_cached_load_second | 9 | 256/256 | 1/0 | 57/0 | 228/57 | B | 256 | 57 |
| both_cold_first | 10 | 0/0 | 0/0 | 0/0 | 313/313 | A | 0 | 313 |
| both_cold_second | 11 | 0/0 | 1/0 | 313/0 | 1252/313 | B | 0 | 313 |
| invalid_A_healthy_B | 12 | —/0 | —/0 | —/0 | —/313 | B | 0 | 313 |

- 评分公式逐行可复算：`(max(0,pending_prefill)+new_tokens)*(bs+1)`，`new_tokens = max(0,313-hit_blocks*128)`。例如第 9 条 A 为 `(57+57)×2=228`，B 为 `57×1=57`，故选 B。
- 场景 A/B 交换通过：同一 313-token 目标分别只在 A、只在 B 缓存时，Router 跟随真实缓存位置，不是固定偏向 A。
- 双缓存 + 负载冲突通过：第 9 条两侧都命中 256，但 A 已有 57 token 待处理且 bs=1，LMetric 转向 B。
- 单副本失效通过：仅在全部 idle 后终止本轮 A 进程组；Router 锁存 A 无效，后续请求只候选健康 B，B 状态未被污染。
- 独立命名空间：两个实例物理 Block ID 数值可重叠但内容不同，两个序号空间均从 1 开始，epoch 不同。

### 5.3 离线审计

[`sync_attempt2/verified.json`](../results/dual_npu_20261025_hyjc_9nd/sync_attempt2/verified.json)：

- 13 条请求全部真实执行，A=6、B=7；
- 连续 completion/control envelope：A 76、B 84；mutation：A 40、B 41；
- 每个 envelope 用独立 journal 重放 → Rust 权威表 → 独立 BlockPool 快照三方一致；
- 请求完成后两副本 bs/waiting/prefill/all_tokens/活动与异常表全 0，请求占用表为空；
- 真实物理卡绑定、eager、单 KV group、8 blocks 均从运行时对象核验，不是配置意图。

聚合（仅描述本轮 13 条，不作性能结论）：单请求耗时均值 712.7 ms、中位 598.7 ms、范围 476.7–1230.9 ms；输入 4843 token、生成 120 token；实际命中合计 1280 token；实际 Prefill 工作量合计 3563 token。

## 6. 异步 eager：首次中断，重跑通过

### 6.1 被中断的会话（不作为异步结论）

[`async_attempt1/INTERRUPTED.json`](../results/dual_npu_20261025_hyjc_9nd/async_attempt1/INTERRUPTED.json)。

- 独立新会话，未复用同步 epoch/账本；engine 0 已完成加载并 `/health` 200，engine 1 正在加载权重。
- **未发出任何测试请求**，未建立 SSE、未进行 reset，因此没有任何异步路由证据。
- 会话最后写入 2026-10-25 14:00:05 UTC；随后容器被外部替换（新容器 PID 1 于 2026-10-26 02:25:47 UTC 启动），且新容器一开始没有挂载 `/dev/davinci*`。
- 该轮明确标记为中断，不能算异步通过。

### 6.2 重跑（`async_attempt2`，通过）

设备重新挂载为物理 0/7 后，用全新会话重跑并离线审计，[`async_attempt2/verified.json`](../results/dual_npu_20261025_hyjc_9nd/async_attempt2/verified.json)：

- 13 条真实请求，A=6、B=7；`async_scheduling=true`、`enforce_eager=true`、单 KV group、8 blocks 均从运行时对象核验。
- 连续 envelope：A 82、B 91；mutation：A 40、B 41；两副本 epoch 不同，序号空间各自从 1 开始。
- 逐请求决策与同步完全同构：A 独有前缀→A（命中 256，prefill 57）；B 独有→B；双缓存首条→A、第二条因 A 的待处理工作→B；双冷缓存→A/B；失效 A 后仅 B 接纳。
- 两实例真实绑定物理 0/7，SSE 在请求前建立，结束后两副本账本全 0，请求占用表为空。
- `shutdown_panic_after_all_requests=false`。

## 7. 失败现场与修复记录

| # | 现象 | 首个明确原因 | 修复 | 证据 |
| --- | --- | --- | --- | --- |
| 1 | 单卡第 1 次：900 s 超时，但引擎日志显示已启动 | 容器 `HTTP(S)_PROXY=127.0.0.1:7891`，urllib 把 loopback 健康检查也送进代理 | 仅对 `127.0.0.1/localhost` 设置 `NO_PROXY` | `smoke_single_attempt1/engine_0.log` |
| 2 | 单卡第 2 次：reset 等待超时，KV_STATE 解析为空 | 新构建 Router 的 tracing 输出 ANSI 转义，`KV_STATE state=` 被颜色码打断 | 解析器剥离 ANSI；Router 环境设 `NO_COLOR=1` | `smoke_single_attempt2/gateway.log` |
| 3 | 同步第 1 次：`Address already in use` | 上一会话刚关闭，18181/18184 处于 TIME_WAIT，端口预检未用 SO_REUSEADDR | 预检加 `SO_REUSEADDR`（仍拒绝真实监听者） | `sync_attempt1/run.json` |
| 4 | 离线审计 `relative_to` 报错 | 结果根以相对路径传入，而 `tempfile` 返回绝对路径 | 改用绝对路径重跑；审计逻辑未改 | `sync_offline_verification_attempt2.log` |
| 5 | 单卡首请求 28.5 s | 一次性 kernel/JIT 预热；双卡未复现 | 请求超时 30→120 s，结果等待 35/40→180 s；基准先用 pilot 预热 | `smoke_single_attempt3/smoke.json` |
| 6 | 异步与首轮性能会话在启动期被终止 | Claude 会话结束会杀掉其前台等待与后台进程组（容器 PID 1 未变） | 长会话改用 `setsid nohup` 脱离进程组，短轮询推进；被中断的 `async_attempt1`、`benchmark_*_attempt1` 保留原样 | `async_attempt1/INTERRUPTED.json`、`benchmark_lmetric_attempt1/run.json` |
| 7 | 容器重建后设备消失、恢复后物理 ID 变为 7 | 外部容器替换，`/dev/davinci*` 先缺失后以 0/7 重新挂载 | 判定为环境阻塞；设备恢复后重新 `npu-smi info -m`，参数化 `LMETRIC_PHYSICAL_DEVICES=0,7` 并重跑 | `environment_after_session_loss.txt`、`async_attempt2/run.json` |

本轮没有修改 `engines/vllm` 或 `blitz-router` 的源码；两个仓库 HEAD 仍为 `0fc695f` / `a005ceb`，修改/未跟踪文件数量与轮次开始时一致。所有测试装置改动只在 `scripts/dual_npu/`：

- `experiment.py`：模型/显存/block/物理设备参数改为环境变量（默认仍为 Qwen3-0.6B 与 0/6 原值）；loopback 代理绕过；ANSI 容忍；`NO_COLOR`；`SO_REUSEADDR`；超时放宽；引擎/Router 命令行抽成方法。
- `benchmark.py`：结果等待超时放宽。
- `inspect_benchmarks.py`：对比输出文件名可用 `BENCH_OUT` 指定，默认不变。
- 新增 `smoke_single.py`：单卡冒烟驱动，复用同一 Session/验证器。

## 8. 未覆盖与限制

- 仍是 2 张 910B2、8-block 小池、`max_num_seqs=1`、eager、313-token 输入 / 8–16 token 输出、回环客户端。
- 没有覆盖队列、preemption/resume/abort、EOS/stop/取消、在途故障恢复与优雅关闭。
- 单卡 28.5 s 首请求异常只记录、不解释为稳定模型成本。
- 跨模型比较仅为描述性：本轮容器、物理卡对（0/6→0/7）和日期均与旧 Qwen3 实验不同，不能把差异归因于模型规模。
- 性能结果与限制见 [`qwen25_7b_dual_npu_benchmark.md`](qwen25_7b_dual_npu_benchmark.md)。
