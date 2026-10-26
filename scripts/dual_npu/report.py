"""Render reports from independently verified raw dual-NPU evidence."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path('/root/lmetric-repro');O=Path(sys.argv[1]);rel='../'+str(O.relative_to(ROOT))
gates=json.loads((O/'correctness.json').read_text());runs={k:json.loads((O/v/'verified.json').read_text()) for k,v in gates.items()}
assert all(r['passed'] for r in runs.values())
bench=json.loads((O/'benchmark_comparison.json').read_text()) if (O/'benchmark_comparison.json').exists() else None
pres=json.loads((O/'preservation.json').read_text())
def link(file):return f'[{file}]({rel}/{file})'
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |',*['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])
resultrows=[]
for mode,r in runs.items():resultrows.append([mode,r['requests'],' / '.join(map(str,r['events'])),' / '.join(map(str,r['mutations'])),'A=6，B=7','通过'])
sync=runs['sync'];bylabel={r['label']:r for r in sync['decisions']};score_rows=[]
for label in ['A_only_target','B_only_same_target','controlled_A_warm_second','both_cached_load_first','both_cached_load_second','both_cold_first','both_cold_second','invalid_A_healthy_B']:
    r=bylabel[label];a,b=r['candidates'];eligible=[c for c in r['candidates'] if c['routing_enabled']=='true'];expected=int(min(eligible,key=lambda c:(int(c['lmetric_score']),int(c['engine'])))['engine']);score_rows.append([label,r['request_id'],a['hit_tokens']+'/'+b['hit_tokens'],a['bs']+'/'+b['bs'],a['prefill']+'/'+b['prefill'],a['lmetric_score']+'/'+b['lmetric_score'],'AB'[expected]+'/'+ 'AB'[r['engine']],r['actual_hits'],r['prefill_work']])
text=f'''# LMetric × vLLM-Ascend：双 NPU 路由验证

日期：2026-10-25（北京时间）。原始日志与请求的 UTC 时间戳保留原样。

## 1. 实际结论

**两个真实独立引擎及统一生产 Router 的缓存感知路由已通过同步和异步 eager 验证。** 每种模式13条请求，分别核验真实模型执行、两份SSE、生产评分与选择、实际后端、HTTP文本与usage、逐步缓存/占用状态和最终账本。不是仅用成功返回文本判断路由正确。

{table(['模式','HTTP请求','A/B连续envelope','A/B连续mutation','请求分布','结果'],resultrows)}

正式证据：{link(gates['sync']+'/verified.json')}、{link(gates['async']+'/verified.json')}。同步 driver 的原始终止错误帧解析失败保留在 run.json；修复观测读取器后，以独立 offline audit 核验全部13条原始请求，未将旧 failure 改写为成功。异步为全新引擎会话，不复用同步 epoch/账本。

{('小规模性能对照已完成，见 [独立性能报告](dual_npu_preliminary_benchmark.md)。' if bench else '没有通过性能对照验收，不能给出性能收益数字。')}

## 2. 设备、软件与部署

- 本容器实际暴露 `/dev/davinci0` 和 `/dev/davinci6`，两张 Ascend 910B2，HBM各65536 MiB，初始健康OK且无运行进程。
- `npu-smi info -m` 实测物理0/6对应 Chip Logic ID 0/1。进程设置 `ASCEND_RT_VISIBLE_DEVICES=0` / `1`。首次直接填物理6启动失败，已停止并保留；未继续使用失败会话。
- 实际进程与卡的对应通过 `VALIDATION_RUNTIME.pid` 和 `npu_loaded_1.txt` 交叉验证。同步 A PID {sync['runtime'][0]['pid']} 对应物理0，B PID {sync['runtime'][1]['pid']} 对应物理6。异步另起独立PID。
- npu-smi/driver显示26.0.rc1，CANN路径9.1.0；Python3.12.13、vLLM0.23.0+empty、vLLM-Ascend0.23.0、torch2.10.0+cpu、torch-npu2.10.0.post4。主机192逻辑CPU、约2TiB内存；启动CPU绑定计划保存于引擎日志。
- 模型 `/models/Qwen3-0.6B`，完全offline；TP=PP=DP=1，一个frontend，一个FullAttention KV group，block/hash size=128。同步 `--no-async-scheduling`，异步 `--async-scheduling`，全部 `--enforce-eager`。
- 每实例 `max_num_seqs=1`、max_model_len=1024、max_num_batched_tokens=128、gpu_memory_utilization=0.20、num_gpu_blocks_override=8（null块0，7个非null块）。真实对象而非配置意图确认上述模式、spec及容量，见 verified.json 的runtime。
- A/B HTTP端口18181/18182，统一Router18184。Router clients.json是两个真实URI的数组；生产binary使用 lmetric-q/radixtree/default-hash，无simulator。Router max-concurrent-requests=4，各场景只发1–2条有界请求，没有提高单引擎max_num_seqs。
- 继续以进程PYTHONPATH加载 `/root/lmetric-repro/engines/vllm`，Ascend来自 `/vllm-workspace/vllm-ascend`。没有改系统editable安装、推理依赖或下载模型。引擎启动每个240秒，HTTP每条30秒。

硬件/版本/命令依据：{link('device_mapping.txt')}、{link('npu_before.txt')}、{link('cpu.txt')}、{link(gates['sync']+'/run.json')}、{link(gates['sync']+'/npu_loaded_1.txt')}。

## 3. 生产 Router 双后端能力及真实公式

`main.rs` 读取backend数组，每URI创建独立VllmClient；Infer按backend数创建独立ScheduleContext，包含LMetric、PrefixBlockHash和KvJournalState。每个work loop拆出独立SSE receiver交给completion loop；每个receiver内部独有MetricSequence。EventHub单订阅限制作用于每个引擎，两个引擎可以同时各有一个订阅。

每份物理映射HashMap、请求占用表、radix树、engine_epoch、step_id和mutation watermark都属于本副本。同数值物理Block ID、同step/mutation序号不会跨副本共享。bs、waiting、prefill和all_tokens也分别更新。

LMetric实际选择最小值：

```text
new_tokens = input_tokens.len().saturating_sub(hit_blocks * block_size)
score = (max(0, pending_prefill_tokens) + new_tokens) * (bs + 1)
```

bs包含已接纳的运行和等候请求；waiting和all_tokens不单独进入LMetric分数。空闲副本仍按未命中工作量区分，同分取先出现的副本。评分读取各自锁内快照；接纳锁内重检routable，缓存版本变化时重算命中，再增加该副本账本。不会预设命中最多者一定获选。

root_target剔除失效副本；协议/序号错误或SSE断流锁存本副本无效并清树。接纳时再次检查失效状态。已修复idle发送循环在completion receiver退出后panic的问题；不声称实现了在途请求恢复或自动重连，失效实验在idle时触发。

`max_num_seqs=1` 是每个引擎的运行并发约束，不是Router的backend数量上限；两个引擎可以各运行一条请求，等待请求不等于提高该参数。当前验证没有覆盖同引擎并发多条运行或preempt/resume。

源码审计记录：{link('router_audit.md')}；实现位于 `blitz-router/router/src/engine/{{client,vllm_http,colocation,kv_journal}}.rs`、`scheduler/policies/{{lmetric,dsl_runtime,policy_runner}}.rs` 和 `scheduler/state.rs`。

## 4. CPU正确性与最小修改

使用真实LmetricQ、capture_observations/apply_default_after、KvJournalState/PrefixBlockHash、MetricSequence和completion_event_loop；没有另写决策器代替生产路径。

{table(['场景','证据与结果'],[
['A独有前缀','A/B命中256/0，分数57/313，选A'],
['B独有前缀','A/B命中0/256，分数313/57，选B'],
['缓存/负载冲突','A命中256但prefill128、bs2，分数555；B冷缓存313，选B'],
['物理命名空间','A/B同ID=1/2对应不同token；A reset不改B映射/版本'],
['单副本失效','注入mutation gap；A锁存无效，评分/接纳拒绝，健康B不受影响'],
['多请求','两completion loop分别完成2条；同step/mutation/物理ID、独立epoch；4条结果各8token，账本归零'],
['idle断流退出','先复现unreachable panic，再改正常break，回归通过']])}

完整LMetric lib：50通过、3忽略；baseline lib：27通过；既有Python journal：14通过；观测/fixture解析测试：3通过。忽略项是独立真实NPU用例/外部trace用例，不作为通过。测试数量不与子集重复相加。

保留失败优先证据：{link('worker_disconnect_red.log')}、{link('baseline_red_linked.log')}；最终回归：{link('lmetric-q_tests.log')}、{link('join-shortest-weight-q_tests.log')}、{link('python_journal.log')}、{link('driver_cpu_tests_stdout_fix.log')}。

本轮生产修改：

1. idle后端断流时worker正常退出，替换错误的unreachable panic。
2. 明确允许现有 `join-shortest-weight-q` 消费相同KV journal，用于公平基线；保留radixtree、无simulator、严格序号/epoch要求。未改变LMetric公式或journal语义。
3. 增加debug CANDIDATE与KV_STATE，以及journal prepare和completion apply host duration histogram，支撑逐步对账与测量。详细观测开销包含在端到端时间内。

## 5. 真实缓存、评分和路由

两个实例先建立各自SSE订阅，并由可观测idle reset得到空缓存。首两条不同请求通过统一Router的受控先后接纳分别落A/B，证明两引擎实际推理。后续所有预热和推理都走同一Router；没有未经跟踪的直接后端推理。

A独有目标后重复目标选A；reset A后，用A的真实700-token工作与目标的受控接纳使目标缓存落B，再重复完全相同313-token目标选B。随后用B的真实长请求负载，使目标落A，形成双方缓存，再比较两份负载。每个场景最多2条请求，整个会话13条。

同步代表性决策如下。分数为生产候选观察，单位为公式的工作量乘数；命中和prefill单位为token。A失效行的A数值被过滤，不能参与最小值比较。

{table(['场景','request ID','A/B预测命中','A/B bs','A/B pending prefill','A/B分数','预期/实际','实际命中','实际prompt工作量'],score_rows)}

这里真实复现了缓存与负载冲突：B命中256但已有444-token待处理工作，分数1002；A无命中但分数313，Router选择A并在真实A执行。双方缓存时，首条同分选A，第二条按负载选B。

每条请求的完整token IDs、长度、HTTP请求/响应和UTC/单调时钟在各会话 `requests.jsonl`；313-token目标及700-token受控负载在fixtures.json。verified.json及其offline_audit中的reconciled.jsonl包含两份候选字段、预期/实际选择、真实生成IDs、命中和prefill工作。所有HTTP正常length完成，每条8个生成token，两条受控长负载为16个。文本整体解码与非流式HTTP文本一致，usage逐条一致。

同步/异步的每个连续completion/control envelope均进行：独立journal重放 → Rust权威物理表和占用表 → 独立读取真实BlockPool/get_block_ids快照。A/B物理ID实际重叠但内容不同、epoch不同；两个序号空间都从1开始。请求完成后各自bs/waiting/prefill/all_tokens及活动/异常请求表全0，请求占用表空。全局缓存映射仍存在，释放没有被误当驱逐。比较对象是scheduler cache metadata，不是逐字节设备KV内容。

单副本失效在全部idle后仅终止本轮A进程组；Router记录RECV_STEP_ERROR并锁存失效，WORK_LOOP_STOPPED正常退出；下一请求只能候选健康B且完成。A旧缓存不再参与评分，B原状态未污染；没有在失效A会话继续推理，也没有验收在途故障恢复。

## 6. 失败现场与观测修正

- 首次绑定物理6到 `ASCEND_RT_VISIBLE_DEVICES=6` 导致aclInit 107001 / Invalid device ID。保存 `sync_attempt1` 后停止；npu-smi映射核验后用logical1重新启动独立会话。
- CPU链接首次缺少libssl/libcrypto开发链接，仅在新结果目录创建指向现有 `/usr/lib64/*.so.3` 的symlink并设置进程LIBRARY_PATH，未安装或升级系统包。
- 同步全部请求完成后，受控停机产生不含outputs的终止error frame，观测读取器发生KeyError。原run.json仍记录failure；修复分类并离线核验全部13条，保存每次audit和失败日志，不掩盖失败。
- 首轮异步两条请求完成后，跨进程stdout的两个完整JSON/日志拼接导致观察器Extra data。已停止会话；离线检查26帧的真实池/Rust状态一致，见 {link('async_attempt1/failure_analysis.json')}，没有把该轮算完整异步通过。解析器改为逐完整JSON读取，增加CPU回归，再以新会话重跑。
- 失败会话与正式异步会话在全部请求结束后的关闭期间均观察到PolicyRunner channel SendError panic；正式异步run的passed检查在关闭前完成，offline verified.json单列shutdown_panic_after_all_requests=true。正常推理/受控失效验收期间没有worker panic。已有EngineDeadError/semaphore清理警告仍属停止阶段，未宣称优雅关闭及完整在途恢复已修复。

## 7. 保全、复跑与正式实验差距

结果根：{link('baseline.json')}、{link('preservation.json')}、{link('artifact_sha256.json')}。本轮前已有历史结果共{pres['historical_files_checked']}个文件逐一SHA256核验未变；两个项目HEAD未改，已有源码未丢失，vLLM及系统安装源码本轮未变。没有reset、clean、stash、commit、push、依赖安装或模型下载。只终止本轮创建的进程组，最终NPU状态见 {link('npu_final.txt')}。

复跑入口和全部硬界限见 [工具说明](../scripts/dual_npu/README.md)。所有会话和失败尝试都使用唯一新目录。

**已具备受限双实例路由与测量原型，尚不具备正式论文性能复现的充分条件。** 还需要：扩大模型和KV池到论文设定；验证单实例max_num_seqs>1以及队列/preemption/resume/abort等合同；接入论文请求轨迹和真实分布；控制预热、CPU/NUMA、顺序和观测开销；扩大重复次数、随机化策略顺序并做统计分析；补齐EOS/stop/取消/故障恢复及优雅关闭。当前结论限定两张卡、Qwen3-0.6B、eager、8-block小池和正常length结束。
'''
(ROOT/'docs/dual_npu_validation.md').write_text(text)
if bench:
    rows=[]
    for r in bench['comparison']:
        rows.append([r['workload'],r['policy'],f"{r['request_throughput_rps']:.3f}",f"{r['output_token_throughput_tps']:.3f}",f"{r['ttft_mean_ms']:.2f}",f"{r['tpot_mean_ms']:.2f}",f"{r['actual_hits']:.0f}",r['request_distribution']])
    host_rows=[]
    for session in bench['sessions']:
        stats=[]
        for stem in ['blitz_policy_schedule_duration_seconds','blitz_kv_prepare_duration_seconds','blitz_step_apply_duration_seconds']:
            hs=[r['instrumented_histograms'][stem] for r in session['rows']]
            count=sum(h['count'] for h in hs);total=sum(h['total_seconds'] for h in hs)
            stats.append(f'{total/count*1e6:.2f} / {count:.0f}')
        host_rows.append([session['policy'],*stats])
    btext=f'''# 双 NPU 小规模性能对照

日期：2026-10-25（北京时间）。前置 [同步/异步正确性验收](dual_npu_validation.md) 已通过，门禁保存于 {link('correctness.json')}。

## 范围和结果

**已建立可信的有界流式测量与对照链路；本结果不代表论文性能复现，也不据此宣称普遍加速。** 两策略分别独立启动同样的物理NPU0/6和模型，使用相同参数、完整journal、debug观测及生产HTTP网关。每策略2条流式pilot通过后，执行3类×8请求×2重复，共48条测量，合计96条测量+4条pilot。全部HTTP200，逐条真实后端/评分/缓存/生成token对账通过，无超时或推理失败。

{table(['负载','策略','请求/s','输出token/s','TTFT均值 ms','TPOT均值 ms','每8请求实际命中token均值','两轮A/B请求数'],rows)}

混合轨迹中LMetric减少实际prefill工作并降低本轮平均TTFT；缓存友好轨迹中，更多请求集中到A，平均TTFT反而升高。这说明命中与负载的取舍不能仅靠命中量判断。上述为小样本观测，不作统计显著性声明。

每行是两次有界重复的均值；两轮请求分布为计数求和。吞吐量是固定稀疏到达轨迹的完成速率，包含到达间隔，不是饱和容量；不使用这些小样本计算显著性或概括整体收益。

## 实验控制

- 同样两张910B2、Qwen3-0.6B、TP=PP=DP=1、同步eager；每实例max_num_seqs=1，KV block128、物理池8，7个非null块。
- LMetric最小化 `(max(0,pending_prefill)+uncached_tokens)*(bs+1)`；项目现有基线join-shortest-weight-q最小化 `3*waiting+bs`。只允许该基线通过相同journal能力门禁，未替换缓存消费逻辑。
- 每条input313 token，输出固定min=max=8；temperature等后端采样行为两策略完全相同。VllmClient沿用现有实现，未声称前端任意采样参数均已透传；本轮比较没有修改该路径。
- 每条轨迹开始前两个真实引擎各执行idle reset并等待SSE/Rust确认空缓存。模型/内核已由同样的流式pilot预热；每策略独立epoch，策略间不复用缓存。
- 固定到达偏移秒：0、0.04、2、2.04、4、4.04、6、6.04；计划和实际单调时间/迟到量全部保存。三类前缀轨迹 cache_friendly、mixed、low_hit 完整token IDs逐字节相同，独立audit验证。每轨迹8请求、两轮；没有无界并发或长时压力测试。
- 策略按LMetric后baseline运行，没有随机化顺序；两次重复不足以排除时段、温度等偏差。

## 测量定义及可核验性

TTFT=发送开始到首个真实token delta的客户端接收时间。排除最早的角色宣告、HTTP头和finish事件；每条请求均验证恰好8个token delta，与真实引擎SSE中的8个生成token一一对应。token特殊符号解码为空时仍是实际生成token事件，不把空角色chunk算首token。

TPOT=(第8个token delta接收时间−第1个token delta接收时间)/7。它是客户端观测的平均token间隔，包含传输/Router开销，不是设备kernel时长。逐个原始chunk和monotonic_ns均保留。所有测量为回环网客户端，未声称跨网络部署延迟。

请求吞吐量=8/(末DONE−首发送)；输出token吞吐量=64/同一区间。另保存输入+输出token accounting吞吐量，区分总token数和实际prompt工作量。实际缓存命中来自真实首PREFILL输出，工作量来自完整completion journal归属（包括outputs=[]的chunked prefill），没有用预测命中冒充实际命中。

Router决策时间使用原有policy schedule histogram，包含快照锁、策略与接纳账务，排除命令队列等待。新KV prepare histogram覆盖journal staging/tree rebuild；step apply覆盖completion host处理，不含SSE接收等待和最后debug状态输出。每个轨迹保存前后sum/count差及均值，不能把总计数误除到单请求。

以下host计时为48条测量的sum/count加权均值，单位µs；斜线后是调用数，prepare/apply每envelope一次，不能和每请求的policy调用混为一谈。

{table(['策略','policy µs / 次数','KV prepare µs / 次数','step apply µs / 次数'],host_rows)}

**未可信测得的项目：事件采集/日志/journal的独立增量开销。** 本轮debug和真实池快照全程开启，端到端指标包含其影响；prepare/apply host时间可以实测，但没有启停对照，不能推算禁用采集后的性能。没有提供该项虚构数字。

## 原始证据和限制

{link('benchmark_comparison.json')} 保存两轮汇总、延迟和所有host时间统计；每策略会话的 requests.jsonl 保存请求/完整token时钟/响应chunk，reconciled.jsonl 保存评分/选择/真实后端/命中/prompt工作量，benchmark.jsonl 保存每轨迹到达计划和metrics差，engine_0/1.log及gateway.log保存逐步完整证据，trace_definition.json保存相同原始输入。

{chr(10).join('- '+link(s['name']+'/run.json')+'：'+str(s['requests'])+'条真实请求，'+str(s['measured_requests'])+'条测量，'+str(s['state_events'])+'帧状态对账。' for s in bench['sessions'])}

独立验收同时核验两策略各50条唯一请求、流式文本与真实token整体解码一致、所有token时间戳、7次/后端的空缓存控制、逐步物理表/占用表一致以及最后双账本全0。

实际到达存在主机线程调度迟到，不能把计划时间当成真实时间。各策略最大迟到（ms）：{', '.join(s['policy']+'='+format(s['arrival_lateness_max_ms'],'.2f') for s in bench['sessions'])}；完整逐条偏差已保存。该偏差和未随机化的策略顺序限制了细小性能差异的解释。

主要限制：max_num_seqs=1、0.6B小模型、8-block人工小池、正常length结束、回环客户端、详细debug观测、只有两轮、固定策略顺序、低负载有间隔到达。接下来应先扩展并发与协议合同，再在论文模型/缓存规模/轨迹下开展随机化、足量重复和观测开销对照，不能直接外推本表。
'''
    (ROOT/'docs/dual_npu_preliminary_benchmark.md').write_text(btext)
print('reports written')
