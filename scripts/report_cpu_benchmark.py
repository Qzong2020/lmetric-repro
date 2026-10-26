#!/usr/bin/env python3
"""Validate paired runs and generate the Chinese CPU experiment report."""
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/cpu/full"
POLICIES = ["lmetric-q", "round-robin-q", "join-shortest-weight-q", "most-hit-q"]
SCENARIOS = {"no_share": "无共享前缀", "shared": "共享前缀", "mixed": "长短请求混合", "hotspot_burst": "热点切换与突发"}


def avg(rows, field):
    return statistics.mean(r[field] for r in rows)


def main():
    rows = json.loads((FULL / "summaries.json").read_text())
    config = json.loads((FULL / "config.json").read_text())
    expected = len(config["policies"]) * len(config["scenarios"]) * len(config["loads_rps"]) * len(config["seeds"])
    assert len(rows) == expected, f"incomplete matrix: {len(rows)}/{expected}"
    grouped, paired = defaultdict(list), defaultdict(dict)
    for row in rows:
        assert row["successes"] == row["requests"] == row["backend_completions"] == row["schedule_count"]
        assert row["drained"]
        grouped[row["scenario"], row["load"], row["policy"]].append(row)
        paired[row["scenario"], row["load"], row["seed"]][row["policy"]] = row
    for pair in paired.values():
        assert set(pair) == set(POLICIES)
        assert len({r["trace_sha256"] for r in pair.values()}) == 1
    aggregates = []
    for (scenario, load, policy), group in grouped.items():
        assert len(group) == 3
        aggregates.append({
            "scenario": scenario, "load": load, "policy": policy,
            "ttft_p95_mean_ms": avg(group, "ttft_ms_p95"),
            "ttft_p95_min_ms": min(r["ttft_ms_p95"] for r in group),
            "ttft_p95_max_ms": max(r["ttft_ms_p95"] for r in group),
            "ttft_p99_mean_ms": avg(group, "ttft_ms_p99"),
            "e2e_p95_mean_ms": avg(group, "e2e_ms_p95"),
            "completed_rps_mean": avg(group, "completed_rps"),
            "cache_hit_ratio": sum(r["total_cached_tokens"] for r in group) / sum(r["total_input_tokens"] for r in group),
            "schedule_mean_us": avg(group, "schedule_mean_us"),
            "dispatch_cv_mean": avg(group, "dispatch_cv"),
        })
    with (FULL / "aggregates.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    aggregate = {(r["scenario"], r["load"], r["policy"]): r for r in aggregates}
    total = sum(r["requests"] for r in rows)
    lm = [r for r in rows if r["policy"] == "lmetric-q"]
    changes = [100 * (1 - pair["lmetric-q"]["ttft_ms_p95"] / pair["join-shortest-weight-q"]["ttft_ms_p95"]) for pair in paired.values()]
    reductions = sum(c > 0 for c in changes)
    lines = [
        "# lmetric-q：CPU 调度实验报告", "",
        f"生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}。",
        "",
        f"完成 {len(rows)} 组正式实验，共 {total:,} 个测量请求，成功率 100%；另有每组 {config['warmup_requests']} 个预热请求。所有组均通过请求计数、后端完成计数、调度调用计数和结束状态归零检查。",
        "",
        f"与按负载选择的 `join-shortest-weight-q` 相比，`lmetric-q` 在 {len(changes)} 个配对设置中有 {reductions} 个 TTFT P95 较低，{len(changes) - reductions} 个较高或持平。应结合下面各场景判断，不宜合并成一个总体提升比例。",
        "",
        "## 环境与测试边界", "",
        "- aarch64 / 鲲鹏 920。宿主机可见 192 核、约 2 TiB 内存；容器 CPU 配额为 8 核、内存上限 32 GiB。",
        "- Rust 1.88.0，本地已有工具链。项目固定 1.86.0 无法编译当前锁文件中的 proptest 1.12.0（要求 1.88）。未修改锁文件或项目工具链文件。",
        "- 四种策略使用真实 Rust router 的 release 构建、相同代码与观测指标，每次只启用一个策略 feature；关闭 ngrok，使用 HTTP/SSE 后端。",
        "- 每组 4 个 CPU 模拟后端，各自运行于独立子进程；发压器也与后端分开运行。router 配置 4 个 Tokio worker 和 2 个校验 worker。只监听 127.0.0.1。",
        "- 未加载模型权重，未使用 NPU/GPU；使用本地 WordLevel tokenizer 精确控制 token 数。结果衡量调度行为及模拟服务条件下的收益，不衡量模型生成质量，也不能换算为真实 NPU 吞吐或 TTFT。",
        "",
        "## 实验方法", "",
        "4 策略 × 4 场景 × 3 负载 × 3 随机种子；种子为 17、29、43，每组 300 个测量请求。低/中/高负载为 40/120/360 req/s。它们来自独立服务模型与预跑覆盖的轻载到排队区间，不是经过真实硬件标定的利用率。",
        "",
        "同一场景、负载、种子的四个策略使用完全相同的请求文本、token 数和到达时间，并核对轨迹 SHA256。策略执行顺序按种子打乱。每组重新启动 router 与空缓存后端，先用相同的 40 请求轨迹预热并排空，再测量；策略造成的缓存状态差异保留。编译和性能实验分开运行。",
        "",
        "开放环 Poisson 到达；热点场景每 60 请求的前 15 个使用 4 倍瞬时速率，其余间隔乘 1.25，并每 100 请求切换主要共享前缀。共享场景输入 512 token、共享前 384 token、8 个前缀组；混合场景输入 64/256/1024/2048 token，输出 16/32/64 token；无共享场景输入 512、输出 32；热点场景输入 1024、共享 896、输出 32。",
        "",
        "后端采用 FCFS、连续 decode 批处理、非分块 prefill。每步模拟耗时（ms）为 `2 + 1000 × 未缓存 prefill token数 / 64000 + decode请求数 × 0.08 + decode上下文token数 / 1024 × 0.015`。该式不读取路由策略或评分。每副本最多 16 个执行请求，prefill 预算 4096 token；每个完整 prompt block 为 16 token，缓存容量 4096 block，按 LRU 淘汰未被活跃请求引用的 block。已计算的前缀才可命中，至少重算一个 token；生成 token 不纳入缓存。此模型未模拟真实算子、chunked prefill、抢占、跨设备通信或真实 KV 内存分配。",
        "",
        "TTFT 从 HTTP 请求发送至首个非空 token，排除初始 role/空内容事件；额外保存从计划发送时间计算的 TTFT 和发压延迟，防止漏看发压端滞后。吞吐使用从计划回放起点到最后请求结束的完整时长，包含排空时间，因此与设定的发送速率不同。下表为三个种子的分位数均值；括号为三个运行的最小值到最大值，不是置信区间。",
        "",
        "## 高负载 TTFT P95（ms）", "",
        "| 场景 | lmetric | 轮询 | 按负载选择 | 仅缓存命中 | lmetric 对按负载选择的变化 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, label in SCENARIOS.items():
        vals = [aggregate[scenario, "high", p]["ttft_p95_mean_ms"] for p in POLICIES]
        change = (vals[0] / vals[2] - 1) * 100
        cells = []
        for policy, value in zip(POLICIES, vals):
            r = aggregate[scenario, "high", policy]
            cells.append(f"{value:.1f} ({r['ttft_p95_min_ms']:.1f}–{r['ttft_p95_max_ms']:.1f})")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {change:+.1f}% |")
    lines += ["", "负变化代表 lmetric 延迟更低；正变化代表更高。只有 3 次重复，不能据此声明统计显著。", "",
              "## 全部负载结果", "",
              "| 场景 | 负载 | lmetric P95 ms | 轮询 P95 ms | 按负载 P95 ms | 缓存优先 P95 ms | lmetric token命中率 | 按负载 token命中率 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for scenario, label in SCENARIOS.items():
        for load, rate in config["loads_rps"].items():
            vals = [aggregate[scenario, load, p]["ttft_p95_mean_ms"] for p in POLICIES]
            hits = [aggregate[scenario, load, p]["cache_hit_ratio"] for p in ["lmetric-q", "join-shortest-weight-q"]]
            lines.append(f"| {label} | {rate} req/s | " + " | ".join(f"{v:.1f}" for v in vals)
                         + " | " + " | ".join(f"{v:.1%}" for v in hits) + " |")
    lines += ["", "## lmetric 的 CPU 开销与验证", "",
              f"- 调度调用平均耗时范围 {min(r['schedule_mean_us'] for r in lm):.2f}–{max(r['schedule_mean_us'] for r in lm):.2f} µs，包含状态快照锁及 admission 记账，不包含请求在命令队列等待的时间与 histogram 写入开销；未单独扣除计时器开销。",
              f"- 各运行调度 P95 所在 histogram 桶的上界范围 {min(r['schedule_p95_upper_us'] for r in lm):.0f}–{max(r['schedule_p95_upper_us'] for r in lm):.0f} µs；这是桶上界，不能当作精确分位数。已扣除预热样本。",
              f"- router 采样峰值 RSS 范围 {min(r['router_peak_rss_mib'] for r in lm):.1f}–{max(r['router_peak_rss_mib'] for r in lm):.1f} MiB，每 100 ms 采样，可能漏掉更短瞬时峰值。",
              f"- router 测量期 CPU 使用率范围 {min(r['router_cpu_percent_one_core'] for r in lm):.0f}%–{max(r['router_cpu_percent_one_core'] for r in lm):.0f}%（100% 为一个核）；空闲采样范围 {min(r['router_idle_cpu_percent_one_core'] for r in lm):.0f}%–{max(r['router_idle_cpu_percent_one_core'] for r in lm):.0f}%。空闲测量每次仅 0.2 s，用于识别轮询开销。",
              "- 当前 router work loop 在没有请求时反复 `next_request` / `yield_now`，因此空闲也消耗多个 CPU 核。这个开销属于现有路由循环，不能归因于 lmetric 的评分公式；本轮保留原有行为。",
              f"- lmetric 发压延迟 P99 范围 {min(r['dispatch_lag_ms_p99'] for r in lm):.2f}–{max(r['dispatch_lag_ms_p99'] for r in lm):.2f} ms；原始结果还包含按计划到达时间计算的 TTFT。",
              f"- 正式矩阵累计报告 {sum(r['evicted_blocks'] for r in rows):,} 次缓存 block 淘汰。所有组请求结束后，后端活动引用以及 router 活动请求、等待请求、待 prefill token、上下文 token 均归零。",
              "- 工作区 45 项 Rust 测试通过（含 router 24 项），另有 6 项 Python 测试通过。Python 测试覆盖：未计算缓存不可复用、活跃 block 不可淘汰、block ID 复用、HTTP/SSE 状态及 token 数一致、独立后端进程及状态复位、轨迹与 tokenizer 长度一致，以及调度 histogram 扣除预热样本。完整日志见 `results/cpu/validation/`。",
              "- 未过滤的工作区测试有 2 项 request-sim 测试因缺少作者本机的数据集/tokenizer 文件失败：`test_gen_string_latency_scaling` 和 `test_inflate_latency`。保留原始失败日志，仅在后续工作区检查中按名称排除；未修改这两个测试。CPU 实验使用独立轨迹与本地合成 tokenizer，不依赖这些外部资源。",
              "", "## 解释结果时的注意事项", "",
              "`most-hit-q` 没有负载限制。在本实现中，`select_max_by` 的并列最大值取后一个候选，因此全冷缓存时会固定选择最后一个副本；随后缓存局部性继续吸引请求。它在本实验中通常集中在单个副本，适合作为缓存与负载取舍的对照。不能把它的差距当成相对生产 llm-d 系统的性能提升；主要基线应使用轮询和按负载选择。",
              "",
              "共享前缀收益取决于流量、前缀分布和服务模型；应同时看命中率、排队、端到端延迟与各副本分配。即使 lmetric 在某些模拟场景胜出，也仍需同模型、同硬件的真实引擎对比，才能判断实际部署收益。",
              "",
              "最初发压器与模拟后端共用一个事件循环的批次出现较大的发送滞后，已终止并排除（`results/cpu/coupled-event-loop-excluded/`）。最终矩阵改用每后端独立进程。进程仍共享容器 CPU 配额，未固定 CPU/NUMA 亲和性；小幅延迟差异可能来自系统调度噪声。",
              "", "## 副本数扩展", ""]
    scaling_file = ROOT / "results/cpu/scaling/summaries.json"
    if scaling_file.exists():
        scaling = json.loads(scaling_file.read_text())
        lines += ["长短请求混合、360 req/s、种子 17、每组 300 请求；只用于观察扩展趋势，各副本数仅运行一次。", "",
                  "| 副本数 | TTFT P95 ms | 调度平均 µs | 调度 P95 桶上界 µs | router 峰值 RSS MiB | CPU（单核100%） |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for r in sorted(scaling, key=lambda r: r["replicas"]):
            lines.append(f"| {r['replicas']} | {r['ttft_ms_p95']:.1f} | {r['schedule_mean_us']:.2f} | {r['schedule_p95_upper_us']:.0f} | {r['router_peak_rss_mib']:.1f} | {r['router_cpu_percent_one_core']:.0f}% |")
    else:
        lines.append("扩展实验尚未完成。")
    lines += ["", "## 文件与复现", "",
              "- `configs/cpu_benchmark.json`：完整参数。",
              "- `scripts/cpu_backend.py`：独立模拟服务模型与有限 KV 缓存。",
              "- `scripts/cpu_benchmark.py`：真实路由器、HTTP/SSE 测量与状态校验。",
              "- `results/cpu/full/summaries.csv`：每组全部指标；`aggregates.csv`：三个种子汇总。",
              "- `results/cpu/full/cases/`：每请求数据、router 日志、Prometheus 前后快照和资源采样。",
              "- `results/cpu/full/traces/`：固定请求轨迹。",
              "- `results/cpu/environment.json`、`manifest.json`：环境、源文件与二进制校验。",
              "- `results/cpu/pilot-isolated/` 和 `smoke-isolated/`：最终链路验证及参数预跑；早期 `pilot/`、`smoke/`、`preflight-incomplete/`、`coupled-event-loop-excluded/` 均不纳入正式结果。",
              "", "安装 Python 依赖后执行 `bash scripts/run_cpu_tests.sh`。可只回放现有二进制：", "",
              "```bash", "python3 scripts/cpu_benchmark.py --mode smoke --output results/cpu/smoke-new",
              "python3 scripts/cpu_benchmark.py --mode full --output results/cpu/full-new",
              "```", "",
              "完整结果是合成工作负载实验，不能作为论文硬件结果的复现。"]
    (ROOT / "notes/cpu_test_report.md").write_text("\n".join(lines) + "\n")
    print(f"Validated {len(rows)} cases, {total} requests; wrote notes/cpu_test_report.md")


if __name__ == "__main__":
    main()
