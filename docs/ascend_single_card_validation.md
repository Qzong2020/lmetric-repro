# LMetric 请求事件原型：真实 Ascend 单卡验收

日期：2026-10-25，北京时间。接续 [请求事件原型报告](ascend_minimal_events.md)。原始服务日志保留主机 UTC 时间（北京时间加 8 小时）。

## 1. 结论与范围

**本轮基础请求事件验收通过；随后独立进行的异步 eager 验收也通过。** 本地 Qwen3-0.6B 确实在一张 Ascend 910B2 上完成推理；包含原型的完整 vLLM 包参与真实模型执行、原生跨进程输出及 SSE 传输；Rust 使用真实 `VllmStepReceiver` 和生产 `completion_event_loop` 消费事件，四项基础账本归零。

成功范围为单卡、单前端、单 KV group、每个独立会话一条纯文本请求、n=1。没有实现 KV Cache journal，没有多卡、图模式或性能对比，也不据此声称 LMetric 已完成昇腾适配。Rust 验收通过隔离测试驱动调用生产接纳策略和消费循环；**没有启动完整 Router 网关/外部客户端路由链路**。

请求运行期间没有遇到引擎启动阻塞、协议失效、序号跳变或账本异常。服务关闭阶段仍有问题：采集关闭和异步会话在 SIGTERM 后记录 `EngineDeadError`，并有 semaphore 清理警告，详见第 8 节；不将其表述为完全无错误的生命周期验收。

主要机器可读结果：[summary.json](../results/ascend_single_card_validation/summary.json)。

## 2. 环境、版本、模型与 Git

### 2.1 NPU

初始 `npu-smi info`：工具/驱动显示 `26.0.rc1`；NPU 0 为 `910B2`，Health=OK，HBM `3469 / 65536 MB`，无运行进程。torch-npu 实测 `is_available=True`、`device_count=1`、名称 `Ascend910B2`。

原始输出：[initial_npu_smi.txt](../results/ascend_single_card_validation/initial_npu_smi.txt)。普通推理期间引擎 PID 18919 出现在 NPU 进程表，模型权重加载日志为 1.1397 GB；完整普通推理日志记录 device=npu。各会话的 `*_npu_smi.txt` 保存设备占用。最终 NPU 0 无运行进程、HBM 恢复至 `3469 / 65536 MB`：[final_npu_smi.txt](../results/ascend_single_card_validation/final_npu_smi.txt)。

### 2.2 实际导入版本

| 组件 | 版本 | 初始实际导入位置 |
| --- | --- | --- |
| Python | 3.12.13 | `/usr/local/python3.12.13/bin/python` |
| PyTorch | 2.10.0+cpu | `/usr/local/python3.12.13/lib/python3.12/site-packages/torch/__init__.py` |
| torch-npu | 2.10.0.post4 | `/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/__init__.py` |
| vLLM | distribution `0.23.0+empty`，module `0.23.0` | `/vllm-workspace/vllm/vllm/__init__.py` |
| vLLM-Ascend | distribution `0.23.0` | `/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py` |

PyTorch 的 `+cpu` 是安装包版本字符串；设备是否可执行另由 torch-npu 和本轮实际 NPU 模型请求证明。版本、metadata 路径和 editable `direct_url.json` 原文见 [initial.json](../results/ascend_single_card_validation/initial.json)。

### 2.3 Git 与既有修改保全

`/root/lmetric-repro` 根目录不是 Git 仓库。两个嵌套仓库的情况为：

| 仓库 | 当前分支 / HEAD | 本轮开始时 |
| --- | --- | --- |
| `engines/vllm` | `lmetric/ascend-minimal-events` / `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | 7 个 tracked 文件修改、3 个 untracked 文件，原型实际存在于工作树 |
| `blitz-router` | `lmetric/ascend-minimal-events` / `a005ceb3a6689d30514bd27e45ede0db933d0fea` | 含已有 tracked/untracked 修改，`request-sim` 子模块显示未跟踪内容；完整状态和所有分支见 initial.json |
| 已安装 vLLM | detached HEAD / 同上 `0fc695fc…` | clean |
| 已安装 Ascend 插件 | detached HEAD / `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd` | clean |

本轮没有切换、重置、清理、stash 或提交这些仓库，也没有修改其生产源码。开始与结束的 HEAD、status、完整 diff、采集到的 dirty/untracked 文件 SHA256（vLLM 10 份、Router 24 份）一致：[preservation.json](../results/ascend_single_card_validation/preservation.json)。未宣称对未纳入该文件清单的子模块内部每个文件作了逐字节验证。

### 2.4 本地模型

扫描到本地 Qwen3-0.6B、Qwen2.5-3B/7B、Llama-3.1-8B 及更大的模型。选择已有 `/models/Qwen3-0.6B`：Qwen3ForCausalLM，bfloat16，单个权重文件 `model.safetensors`，1,503,300,328 bytes。模型目录配置 SHA256、文件大小与候选模型清单均在 initial.json 中。未下载模型；启动时设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。

## 3. 安全加载原型与普通推理

初始先使用已安装 vLLM，`VLLM_LMETRIC_EVENTS=0`，启动普通服务。模型成功上 NPU；输入 `The capital of France is`，HTTP 200，返回 ` Paris. The capital of France is also`，prompt=5、completion=8、finish_reason=length。

证据：[baseline_server_attempt1.log](../results/ascend_single_card_validation/baseline_server_attempt1.log)、[baseline_response.json](../results/ascend_single_card_validation/baseline_response.json)、[baseline_npu_smi.txt](../results/ascend_single_card_validation/baseline_npu_smi.txt)。该首轮服务没有在进程内部额外打印模块 `__file__`，安装路径由同环境独立导入记录确认；该首轮实际 Scheduler/MRO 和 step 方法标记为**未确认**，不以拟用配置填充。

此后通过仅对验收进程设置：

```bash
PYTHONPATH=/root/lmetric-repro/engines/vllm:$PYTHONPATH
```

加载原型，继续使用现有 Ascend 插件及 site-packages 依赖。没有 pip 安装、替换 editable 安装、编译 vLLM/Ascend 或修改系统库。完整包导入成功，原生 `EngineCoreOutputs` schema 包含 `lmetric_event`；随后真实模型运行也成功，因此本模型/路径不需要环境替换。此结论不外推到全部模型与自定义算子。

[launch.py](../scripts/ascend_single_card/launch.py) 只包装原始初始化/step 以打印真实对象、返回结果后的状态和身份映射，不替代 scheduler、executor 或输出；前端和 EngineCore 子进程均打印 `VALIDATION_IMPORT`：

- vLLM：`/root/lmetric-repro/engines/vllm/vllm/__init__.py`
- EngineCore：`/root/lmetric-repro/engines/vllm/vllm/v1/engine/core.py`
- Ascend：`/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py`

同源码关闭开关的独立普通推理也通过，见第 7 节，其进程内路径和真实配置有完整观测日志。

## 4. 真实生效的配置与执行路径

启动命令保存在 [serve.sh](../scripts/ascend_single_card/serve.sh)。同步使用 `--no-async-scheduling`，异步独立会话使用 `--async-scheduling`；两者都显式 `--enforce-eager`。不是由 eager 推测异步开关。

所有正式会话均使用 TP=PP=DP=1、`_api_process_count=1`、max_model_len=1024、max_num_seqs=1、max_num_batched_tokens=128、prefix caching/chunked prefill 开启、block_size=128、gpu_memory_utilization=0.25。运行快照中 speculation、KV transfer、EC transfer、LoRA 均为 None；纯文本模型，未启用多模态。

| 运行后实测值 | 原型采集关闭 | 同步正式会话 | 异步独立会话 |
| --- | --- | --- | --- |
| EngineCore PID | 32768 | 29888 | 35426 |
| Scheduler | Ascend `BalanceScheduler` | Ascend `BalanceScheduler` | vLLM `AsyncScheduler` |
| async_scheduling | false | false | true |
| enforce_eager | true | true | true |
| 实际 EngineCore 方法 | `step` | `step` | `step_with_batch_queue` |
| batch_queue_size | 1 | 1 | 2 |
| batch_queue | None | None | 真实存在，结束时 length=0 |
| scheduler/hash block size | 128 / 128 | 128 / 128 | 128 / 128 |
| KV group 数 | 1 | 1 | 1 |
| 实际 num_blocks / cache_num_gpu_blocks | 1029 / 1029 | 1029 / 1029 | 1028 / 1028 |
| 可用于 KV 的运行时 memory bytes | 15106679296 | 15106793984 | 15095464448 |

容量是初始化后的真实 `scheduler.kv_cache_config.num_blocks`，不是预设配置。保留 null block 后的可分配块数、各时点真实 free/resident 物理块数及完整物理映射：**未确认**。会话间容量差异不作为采集开销或性能结论。

共同的实际 KV spec：

```text
FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128,
 dtype=torch.bfloat16, kv_quant_mode=NONE, page_size_padded=None,
 head_size_v=128, sliding_window=None, attention_chunk_size=None)
```

同步实际 MRO：

```text
vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler
 → vllm.v1.core.sched.scheduler.Scheduler
 → vllm.v1.core.sched.interface.SchedulerInterface → abc.ABC → object
```

异步实际 MRO：

```text
vllm.v1.core.sched.async_scheduler.AsyncScheduler
 → vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler
 → vllm.v1.core.sched.scheduler.Scheduler
 → vllm.v1.core.sched.interface.SchedulerInterface → abc.ABC → object
```

实际类的源码路径、完整 parallel config、队列调用及事件序号见 [sync_final_evidence.json](../results/ascend_single_card_validation/sync_final_evidence.json)、[async_evidence.json](../results/ascend_single_card_validation/async_evidence.json)；原始 `VALIDATION_RUNTIME` / `VALIDATION_STEP` 日志分别在对应 `*_server.log`。未使用 `configs/ascend_minimal_events.json` 的 INTENDED_ONLY 内容冒充运行快照。

## 5. 真实同步事件与 Rust 账本

使用普通服务 `/tokenize` 对真实文本编码，得到 313 个 token；原文和全部 token 见 [prompt.json](../results/ascend_single_card_validation/prompt.json)。每个新引擎只发送一条该文本的 tokenized 请求。Rust `VllmClient` 实际发送 `X-Request-Id: 2026102501`，max_tokens=min_tokens=8、stream=false。n 未在 HTTP body 显式发送，服务端 schema 默认 n=1，原型在接纳时检查 n 必须等于 1；实际响应只有一个 choice。

先由生产 `VllmStepReceiver` 建立 `GET /v1/metrics` 并确认 HTTP 200，再由真实 `VllmClient.add_request` POST `/v1/completions`。握手文件由接收器在获得 HTTP 200 后写入；驱动等待该证据后才提交。不是先请求后补订阅。服务日志和 Rust 的 `REAL_SUBSCRIBED_BEFORE_POST` 可交叉核对。

同步正式会话的运行时映射：

```text
外部 ID 2026102501
内部 ID cmpl-2026102501-0-855b44da
engine_epoch 77d357596ba44f3995b327ff94a21169
```

映射取自真实 `EventHub.register` 日志，SSE 将该内部 ID 映射为整数 2026102501；未按 ID 字符串前后缀猜测。Rust 原始 SSE、DTO 转换后对象和每步实际扣账都已记录。

| step_id | prompt 工作量 | 真实输出 | 应用后 bs / waiting / prefill / all_tokens |
| --- | --- | --- | --- |
| 接纳 | — | — | 1 / 1 / 313 / 313 |
| 1 | 128 | 空 chunk | 1 / 1 / 185 / 313 |
| 2 | 128 | 空 chunk | 1 / 1 / 57 / 313 |
| 3 | 57 | 首 token，PREFILL | 1 / 0 / 0 / 314 |
| 4–9 | 每步 0 | 每步一个 DECODE token | bs=1、waiting=prefill=0，all_tokens 逐步为315至320 |
| 10 | 0 | 第8个 token，DECODE，is_finished=true | 0 / 0 / 0 / 0 |
| 11 | 0 | outputs=[]，结束清理步 | 0 / 0 / 0 / 0 |

实际 hit_token_cnt=0（全新引擎的单请求）；Σprefill=313。输出的 `prev_computed_tokens` 为 `[256,313,314,315,316,317,318,319]`。两个无输出 prefill chunk 没有伪造 token，waiting 在第一次真实输出后才归零。

真实生成 token：`[5209,10339,1246,264,1681,16211,1526,264]`，解码为 ` Please explain how a request passes through a`，与真实 HTTP 文本完全一致；HTTP usage=313+8=321，finish_reason=length。SSE 结束字段为 is_finished，未冒充其包含独立 finish_reason 字段。

step 1–11 连续，epoch 不变。原始 latency（整数 ms）：`[274,64,74,74,60,60,62,61,60,61,0]`，每条 latency_kind=`schedule_to_completion_host_ms`、execution_mode=`sync`。这是 host schedule 到对应完成的时间，含软件路径及等待；不是 device time、客户端 TTFT 或性能对比。

证据：[sync_final_server.log](../results/ascend_single_card_validation/sync_final_server.log)、[sync_final_rust.log](../results/ascend_single_card_validation/sync_final_rust.log)、[sync_final_events.jsonl](../results/ascend_single_card_validation/sync_final_events.jsonl)。

## 6. 独立异步验收

同步完整对账通过后，新建异步 eager 会话；没有复用断开的 SSE 或旧账本。外部 ID 仍为 2026102501，真实内部 ID 为 `cmpl-2026102501-0-883b4ca6`，新 epoch=`82d5d44b38594b9e8be4cce0054fc447`。

结果：真实 SSE step 1–12 连续，prefill 工作量仍为128、128、57，输出仍为8个 token，token ID/HTTP 文本一致，step 10 正常结束。step 11、12 均为无 token、prefill=0 的尾部完成/清理事件，也经过真实消费循环；四项账本在 step 10–12 均保持0。执行队列从在途 length=1 排空到0，最终 scheduler running=waiting=0、requests=[]。

latency ms 为 `[325,146,157,145,131,130,129,129,129,66,2,0]`，execution_mode=`async`，包含队列停留，区间可能重叠；不进行同步/异步性能比较。初次填队列时 step 方法调用有 seq=0，随后完成序号逐步增加：**方法调用次数、完成事件数和模型前向次数不是同一概念**。

两种正式会话在结束并排空后均观测到 EventHub queue_size=0、identities=0、failed=None（订阅尚活跃时）。随后驱动退出、主动断开订阅，并关闭引擎；原型按设计要求新会话，未继续向旧会话路由。

证据：[async_server.log](../results/ascend_single_card_validation/async_server.log)、[async_rust.log](../results/ascend_single_card_validation/async_rust.log)、[async_events.jsonl](../results/ascend_single_card_validation/async_events.jsonl)、[async_detokenize.json](../results/ascend_single_card_validation/async_detokenize.json)。

## 7. Rust 路径、采集开关与验收驱动边界

Rust 驱动在 `/tmp/lmetric-real-npu-router` 中复制当前用户工作树，离线编译，复用已有锁文件和依赖。仅在副本增加真实 NPU 验收用例，以及握手证据/原始 SSE 日志；原接收器解析、协议序列校验、DTO 转换和 `completion_event_loop` 算法均保持原样。接纳走真实 `LmetricQ::schedule`；不是 Python 重写账本或使用合成事件喂入消费者。

临时目录只为已有 `/usr/lib64/libssl.so.3`、`libcrypto.so.3` 创建构建链接，没有安装系统开发包；没有改用户 Cargo 配置。验收代码和差异：[real_npu.rs](../scripts/ascend_single_card/real_npu.rs)、[router_observation.patch](../results/ascend_single_card_validation/router_observation.patch)、[router_driver.patch](../results/ascend_single_card_validation/router_driver.patch)、[rust_build_final.log](../results/ascend_single_card_validation/rust_build_final.log)。每个正式真实模型用例均为 1 passed；没有把旧 CPU 合成测试数量加入本轮通过数。

同一原型源码开关对照：

| 行为 | VLLM_LMETRIC_EVENTS=0 | VLLM_LMETRIC_EVENTS=1 |
| --- | --- | --- |
| `/v1/metrics` | 实测404 | Rust先订阅，实测200 SSE |
| 没有 SSE 订阅时发普通请求 | 实测成功 | 本轮正式会话均先订阅，未另做拒绝用例 |
| 313输入、8输出的模型请求 | 实测HTTP200 | 同步/异步均实测HTTP200 |
| tracker seq | 实际 None | 同步1–11；异步1–12 |
| 执行模型/依赖来源 | 同一原型源码和现有Ascend插件 | 相同 |

证据：[off_server.log](../results/ascend_single_card_validation/off_server.log)、[off_result.json](../results/ascend_single_card_validation/off_result.json)。未测开关的性能差异。

首次探索同步会话 `sync_*` 的驱动在 step 10 请求结束后即退出，服务日志随后还有 step 11。因此它不能证明全部尾部事件已消费，未作为最终验收依据。关闭该会话后，仅改验收驱动以继续读取至完成后空闲2秒；正式 `sync_final_*` 和 `async_*` 核对 Rust 收到的最大 seq 与引擎完成最大 seq 相同，清理事件也全部应用。没有在未完整观测的旧会话继续路由，也没有为消账修改原型或 KV 逻辑。

## 8. 错误、限制与下一步

- 启动和实际请求阶段没有出现需安装、重编译引擎或改变依赖才能解决的阻塞。保留的 warning 包括自定义 `VLLM_LMETRIC_EVENTS` 不在 vLLM env registry、Ascend 参数重设、CANN/driver padding 提示及模型默认 sampling 参数；采集是否启用由真实 tracker/SSE 验证，未由 warning 推断。
- **关闭阶段问题**：`off_server.log` 的 UTC 10:27:04 和 `async_server.log` 的10:28:33，在主动 SIGTERM、shutdown mode=abort timeout=0、engine manager 已停止后，`AsyncLLM.output_handler → get_output_async` 报 `EngineDeadError`。同步正式会话未出现该异常。多个会话退出时还有 `resource_tracker` 的1个 semaphore 清理警告，关闭管理器日志含 force killing remaining processes。推理/清理事件和零账本均先于这些关闭日志完成；最终 NPU 无进程。最可能是已有关闭时序与后台读取竞争（这是推断，未定位根因）；关闭采集时同样可见，不能归为仅由事件采集引起。下一步最小工作是单独复现关闭流程并检查 output handler 取消与 engine manager 停止顺序，本轮不修复，也不据此声称优雅关闭已验收。
- 验收驱动曾因 `ValidGenerateRequest` 无 Clone 编译失败；改成在转移 Entry 所有权前提交实际请求后通过。一次从错误 cwd 调用 Cargo 导致离线找不到 async-stream，改在原配置的副本目录后通过。原始失败日志保留，均不是模型/插件兼容性阻塞。
- 初始受限沙箱报 `bwrap: Failed to make / slave: Permission denied`，后续通过获准的沙箱外命令执行本轮检查；未因此覆盖环境。
- 只验证冷启动单请求的基本链路。真实 prefix hit/预测偏差、取消/抢占、断流恢复、长时间连续服务、完整网关、图模式、多组/多卡及 KV 物理内容和驱逐一致性均未验收。事件明确 `kv_events=false`；没有导出 KV journal。

本轮没有继续修改 KV Cache 逻辑。保留现有依赖和用户工作树，所有实验服务均已停止。

## 9. 产物和本机复跑

产物根目录：[results/ascend_single_card_validation](../results/ascend_single_card_validation)。[artifact_sha256.json](../results/ascend_single_card_validation/artifact_sha256.json) 保存脚本及结果哈希。

本机已编译好真实 Rust 驱动；在空闲NPU上，每次使用新会话和新结果名称。先在一个终端启动，再在另一个终端运行驱动；驱动等待health并自行先订阅：

```bash
cd /root/lmetric-repro
bash scripts/ascend_single_card/serve.sh sync > /tmp/lmetric-sync-server.log 2>&1
# 另一终端；新名称避免覆盖本轮证据和握手文件
python scripts/ascend_single_card/run_session.py sync_recheck
# 完成后停止该服务，再单独验证 async；不复用原订阅/账本
```

`run_session.py` 使用本轮已生成的313-token prompt和具体Rust binary路径；它是本机验收驱动，不是通用部署入口。重建方法见 [prepare_router.py](../scripts/ascend_single_card/prepare_router.py)，为保护已有副本，临时目标已存在时明确拒绝覆盖。无需更改当前已安装的 vLLM-Ascend 环境。
