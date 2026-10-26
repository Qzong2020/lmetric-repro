# 论文复现对照：arXiv 2603.15202v3

后续执行状态：vLLM 基线差异已修正，CPU 公式及真实轨迹测试已完成，见[论文 CPU 测试报告](paper_cpu_test_report.md)。以下保留首次审计时的发现和待办，不代表修正后的当前状态。

目标：[Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling](https://arxiv.org/abs/2603.15202v3)。本次仅阅读、核对和拟定复现步骤，未改算法、未启动新实验。

论文 PDF、HTML 与算法原图保存在 `data/paper/`。以下将已核实的事实、代码差异和待确认事项分开记录。

## 版本与材料

- 论文固定为 v3（2026-06-05），避免后续版本变化。
- 本地 router：`a005ceb3a6689d30514bd27e45ede0db933d0fea`，分支 `lmetric/camera-ready`；只读查询时与作者该公开分支 HEAD 一致。这不等于已取得作者用于各图的实验提交与参数。
- 本地 request-sim 子模块：`8a265cb7392d3b4554c469774130063c8c87e108`，来源为 `blitz-serving/request-sim`。
- [实验编排仓库](https://github.com/blitz-serving/MetricsTestRunner) 查询时 HEAD 为 `3ad55e81923f8f33044aa68dc178858c6694971e`；尚未完整核对每张图的配置。
- [公开轨迹](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) 查询时 HEAD 为 `5f7439c51ec248a0c585f7d90a41a6f57773b912`；尚未下载原始数据文件。文件版本、采样区间和论文场景映射仍需记录。

## 复现对象

原文定位：§5/图17为主算法；§5.2为热点分析及两阶段检测；§4.1为硬件、模型和轨迹设置；§6/图22–23为端到端比较；§6.2另列研究基线。主实验使用 H20、真实模型及轨迹，报告 TTFT 和 TPOT。[论文原文](https://arxiv.org/html/2603.15202v3)

## 代码审计结果

### 1. 主乘法公式与图17一致：不应移除 `+1`

图17在比较前把候选请求计入两个指标。本地代码的实际表达式为：

`(queued_prefill_tokens + input_tokens - cached_prefix_tokens) × (bs + 1)`

对应 `router/src/scheduler/policies/lmetric.rs:18` 和 `dsl_runtime.rs:143`。`apply_default_after` 在选定目标后写入 admission 状态，不会给所有候选实例重复提交请求。当前三项新增策略测试仅验证局部行为，尚不能证明真实引擎反馈始终保持这些指标的语义。

证据：[图17原图](../data/paper/fig17-method.png)、[论文§5](https://arxiv.org/html/2603.15202v3#S5)。

### 2. vLLM 基线存在可改变路由决策的口径差异（P0）

图6(a)给出 `4Q + R`。[图6原图](../data/paper/fig6-baselines.png)

本地 `vllm.rs:29` 实际为 `4 × waiting + bs`。`state.rs:111` 定义 `running = bs - waiting`，因而按现有字段定义，本地表达式相当于 `5Q + R`。

例如，候选 A：等待2、运行0；候选 B：等待0、运行9。图6的分数为 A=8、B=9，本地表达式为 A=10、B=9，选择方向不同。

这是一项静态代码审计发现。本轮未更改基线。后续应先确认真实引擎的 waiting/running 事件口径，再对齐表达式、添加差分用例，并同步相关文档。此前 CPU 实验只可标为“当前仓库的 join-shortest-weight-q”，不能直接称为论文 vLLM 基线复现。

### 3. 热点保护未在当前主策略路径中找到（P1）

当前 `lmetric-q` 只有单个乘法选择表达式；搜索其调度和状态路径，未发现 §5.2 两阶段检测的接入。需明确复现的是基础乘法策略，还是含热点保护的扩展版本。补齐此项属于算法复现范围，应与通用服务异常处理区分。

### 4. 基线集合尚未对齐（P0/P1）

此前只运行 lmetric、轮询、当前仓库负载策略、纯缓存命中策略。它们不足以覆盖论文主比较中的 Bailian、Dynamo 和预测式 llm-d。

本地存在 `bailian-impl-q`、`dynamo-q`/`dynamo-po-q` 和 `least-ttft-q`，但名称存在不代表已经完成对齐：需分别核查归一化、采样/选最优方式、参数、PD共置语义及预测网格。`most-hit-q` 不能替代论文中的预测式 llm-d。`least-ttft-q` 也只有在对应预测模型有效时才有比较意义。

## 现有 CPU 结果的复现地位

`notes/cpu_test_report.md` 的 144 组结果保留为工程预实验，数值不变。其请求、服务时间模型、缓存配置及负载来自我们构造，且上面的基线差异尚未处理；不能据此确认或否定论文性能结论。

`scripts/cpu_benchmark.py` 记录 TTFT、E2E 和 token 数，但没有保存每个 token 的接收时间序列，也未汇总每请求 TPOT。仅据现有数据不能声称已复现论文全部指标。真实实验前需要明确指标定义与实现，避免用结束事件后的时间替代最后一个 token 时间。

## 下一步仅做与论文直接相关的工作

1. **算法对照**：核实 waiting/running、候选请求计入时机及 prefill token 更新，建立论文公式与真实策略函数的逐状态差分用例；优先解决 vLLM 基线差异。
2. **真实轨迹准备**：使用作者公开轨迹及其回放工具思路，核对时间单位、token 长度、哈希前缀共享和 tokenizer 重编码。公开轨迹为匿名 block hash；重建文本用于保留性能特征，不用于评估生成语义。[数据说明](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)、[回放工具](https://github.com/blitz-serving/trace-replayer)
3. **实验清单**：从编排仓库逐项锁定模型 revision、引擎版本、参数、轨迹文件与窗口、基线调参和负载缩放方式；公开配置无法确定的项目标为未确认，不自行猜测。
4. **指标准备**：记录首/末内容 token、输出 token 数及请求完成状态，明确每请求 TPOT、分位数与 CDF 的聚合方式。先验证统计口径，再执行真实模型实验。
5. **硬件实验**：原硬件条件下的对照才可核对原文性能数字；如使用 NPU，则将其界定为硬件迁移下的复现实验，并记录偏离项。CPU阶段先完成前述算法、数据和指标校验。

首个端到端目标建议限制为一个公开轨迹、一个模型、LMetric 与已核对的 vLLM 基线，完成 TTFT/TPOT CDF 与负载扫描，再扩展其余基线。具体模型、窗口和参数应从论文及作者实验材料中确认。

当前不把通用空闲轮询优化、额外合成负载扫描列为复现里程碑；只有阻塞实验正确性或发压能力时才处理。
