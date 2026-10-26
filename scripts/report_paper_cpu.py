#!/usr/bin/env python3
"""Summarize measured paper-conformance checks without inventing serving results."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_cpu"


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    before = read("results/paper_cpu/before/formula_differential.json") if (OUT / "before/formula_differential.json").exists() else None
    after = read("results/paper_cpu/after/formula_differential.json")
    real = read("results/paper_cpu/real_trace_differential.json")
    traces = read("results/paper_cpu/trace_inventory.json")
    replay = [read(f"results/paper_cpu/replay-{letter}/summary.json") for letter in "AB"]
    config = read("configs/paper_cpu/inputs.json")
    assert after["lmetric_mismatches"] == after["vllm_mismatches"] == real["mismatches"] == 0
    for row in replay:
        assert row["checked_requests"] == 2000
        assert all(row[key] == 0 for key in ["length_errors", "shared_block_errors", "distinct_hash_alias_errors", "timestamp_errors"])
    assert all(not any(t["schema_errors"].values()) for t in traces)
    assert real["requests"] == sum(r["checked_requests"] for r in replay)
    assert real["cross_request_prefix_queries"] == real["requests"]
    assert real["cross_request_prefix_mismatches"] == 0
    date = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    lines = ["# 论文对应的 CPU 测试结果", "", f"完成时间：{date}", "",
        "结论：本轮覆盖的 LMetric 公式、修正后的 vLLM 基线以及公开轨迹 token 重建检查全部通过。验证范围为算法与数据一致性；未运行模型推理，不提供模拟 TTFT/TPOT 优势百分比。",
        "", "## 对应论文与版本", "",
        "依据 [arXiv:2603.15202v3](https://arxiv.org/abs/2603.15202v3) 的图6(a)、图17及实验指标要求。原图在 `data/paper/fig6-baselines.png`、`fig17-method.png`。",
        "",
        f"公开轨迹固定提交 `{config['trace_revision']}`；tokenizer 使用 `{config['tokenizer_model']}` 的 `{config['tokenizer_revision']}`。只下载 tokenizer，未下载模型权重。文件 SHA256 见 `configs/paper_cpu/inputs.json`。",
        "", "## 1. 真实策略与独立论文公式的差分", "",
        "真实 Rust `Policy::schedule` 被直接调用；参考公式独立使用 queued/running/prefill/hit 等夹具输入，不调用生产代码的评分辅助函数。测试覆盖1–16个候选副本、完整及不完整块、不同缓存命中与队列状态、并列最小分数，并检查 admission 计数及匹配实际命中反馈时的零修正。并列时接受任一最小分数候选。",
        "", "| 项目 | LMetric | vLLM基线 |", "|---|---:|---:|"]
    if before:
        lines.append(f"| 修正前，5000组状态 | {before['lmetric_mismatches']} 次差异 | {before['vllm_mismatches']} 次差异 |")
    lines += [f"| 修正后，同批5000组状态 | {after['lmetric_mismatches']} 次差异 | {after['vllm_mismatches']} 次差异 |",
        "",
        "修正前 `4*waiting + bs` 根据本地 `bs=queued+running` 定义会将等待请求重复计权；修正为 `3*waiting + bs`。LMetric评分公式未改。对应回归用例：A有2个等待、0个运行，B有0个等待、9个运行；修正后选择A。相关 DSL、注释及项目说明已同步。",
        "",
        f"另对回放器生成的 {real['requests']:,} 条真实轨迹 token 输入，在16个受控副本快照下执行 {real['decisions']:,} 次真实策略决策，零差异。这些快照是验证夹具，未冒充真实引擎在这些时刻的状态。有限样本零差异不等于形式化证明或真实推理链路验证。",
        "", "## 2. 完整公开数据校验与重建", "",
        "| 数据 | 完整请求数 | 时间跨度秒 | 输入token P50/P95 | 输出token P50/P95 | 重建样本 | 重复完整块检查次数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for trace, sample in zip(traces, replay):
        inp, output = trace["input_tokens"], trace["output_tokens"]
        lines.append(f"| {trace['file']} | {trace['requests']:,} | {trace['duration_seconds']:.3f} | {inp['p50']:.0f}/{inp['p95']:.1f} | {output['p50']:.1f}/{output['p95']:.1f} | {sample['checked_requests']} | {sample['repeated_full_blocks_checked']:,} |")
    total = sum(t["requests"] for t in traces)
    repeats = sum(r["repeated_full_blocks_checked"] for r in replay)
    lines += ["",
        f"共 {total:,} 条原始请求通过官方LFS文件大小、SHA256、JSON结构、16-token块数量及时间顺序校验。各取文件顺序前2000条，由作者现有 `BailianDataset::inflate` 和 `TokenSampler` 实际生成文本，再用 Rust tokenizers 0.21.4 与真实 Qwen2 tokenizer 重编码。",
        "",
        f"4000条样本长度、时间戳均匹配；{repeats:,} 次重复完整块的token序列一致；未发现不同匿名完整块映射到同一token块的别名。只检查可缓存的完整16-token块，未将不足16-token的尾块算作已提交缓存。未再次应用chat template。",
        "",
        f"另将实际token构成的 `BlockHashState` 与生产 `PrefixBlockHash`（RadixTree）组合查询，以匿名块构造的独立前缀树为参考：{real['cross_request_prefix_queries']:,} 次跨请求前缀查询零差异，累计匹配 {real['cross_request_prefix_hit_blocks']:,} 个完整块。此项假定之前的请求均已完成且缓存容量足够，不模拟驱逐与并发完成顺序。",
        "",
        "实际重建文本使用作者回放器的随机生成逻辑，生成文本并非固定种子；本次实际token输入已归档在 `replay-A/tokenized.jsonl` 与 `replay-B/tokenized.jsonl`，并记录校验和。重复运行验证的是长度和共享关系等不变量，具体随机文本可以变化。",
        "",
        "完整数据还记录了逐分钟请求数与输入/输出分布。`potential_prefix_token_ratio` 仅为无限缓存且所有先前请求已完成时的完整前缀复用比例，不是真实引擎缓存命中率。原始输入存在很长请求，后续真实模型实验必须另行对齐上下文长度、截断或过滤策略，不能直接静默裁剪。",
        "", "## 3. 指标统计检查", "",
        "6项Python检查通过：空role事件不计入TTFT；结束信号和usage晚到不延长decode区间；真实输出token数可不同于chunk数；单输出token无TPOT；不完整/失败及时间倒序流不生成有效样本；CDF按请求构造。",
        "",
        "TTFT=首内容到达−发送；TPOT=(末内容到达−首内容到达)/(真实输出token数−1)。输出计数来自usage或后端权威计数，缺失时TPOT留空，不用chunk数替代。这是客户端观测口径，尚未与论文原始统计脚本完成逐字段比对。",
        "",
        "当前 `scripts/paper_metrics.py` 是独立统计模块，尚未替换线上回放器的统计路径。本轮只验证已知事件序列上的统计正确性，没有对模型输出测得真实TTFT或TPOT。",
        "", "## 验证与复现", "",
        "- 工作区47项Rust检查通过；真实轨迹策略专项另1项通过。",
        "- Python统计检查6项通过。",
        "- 2项已有request-sim测试仍依赖作者本机文件，按名称排除；新增真实数据专项使用本轮固定的公开文件。",
        "- 使用本机Rust 1.88.0：既有锁文件中的proptest要求该版本。未改工具链文件或锁文件。",
        "- 本轮全部为CPU操作，未调用NPU/GPU。算法差分与回放器使用debug构建，不以耗时比较策略性能。",
        "", "```bash", "bash scripts/run_paper_cpu_tests.sh", "```", "",
        "初始错误基线的失败日志保留在 `before_tests.log` 和 `before/formula_differential.json`；复现脚本只运行修正后的代码，不重新引入旧错误。",
        "", "## 尚未完成的论文实验", "",
        "完整Bailian/Dynamo/预测式llm-d等基线配置及调参仍待逐项核对；§5.2两阶段热点保护尚未加入当前主策略，本轮不验证其保护效果；真实引擎的waiting/running事件口径、逐步状态反馈及TTFT/TPOT性能收益尚待真实模型实验。",
        "",
        "此前144组自建模拟结果保持原样，归为工程预实验，不作为本轮论文性能复现证据。"
    ]
    report = ROOT / "notes/paper_cpu_test_report.md"
    report.write_text("\n".join(lines) + "\n")
    paths = [report, ROOT / "configs/paper_cpu/inputs.json", ROOT / "scripts/run_paper_cpu_tests.sh"]
    paths += list((ROOT / "scripts").glob("*paper*.py"))
    paths += list(OUT.glob("*.json")) + list(OUT.glob("*.log"))
    paths += list(OUT.glob("after/*.json")) + list(OUT.glob("before/*.json"))
    paths += list(OUT.glob("replay-*/*.json")) + list(OUT.glob("replay-*/*.jsonl"))
    paths += [ROOT / "blitz-router/router/src/scheduler/policies" / name for name in ["vllm.rs", "lmetric.rs", "paper_tests.rs"]]
    paths.append(ROOT / "blitz-router/request-sim/examples/paper_trace_check.rs")
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated {total} source requests, {real['requests']} replay inputs and {after['decisions']+real['decisions']} corrected policy calls")
    print(report)


if __name__ == "__main__":
    main()
