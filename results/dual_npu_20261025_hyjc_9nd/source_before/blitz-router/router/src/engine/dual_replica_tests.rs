// CPU fixtures exercise production policy, protocol and completion consumption.
use crate::engine::kv_journal::KvJournal;
use crate::scheduler::policies::dsl_runtime::{capture_observations, apply_default_after, prefill_tokens};
fn journal(ms:Vec<Value>,last:u64)->KvJournal {
    serde_json::from_value(json!({"version":"kv-journal-v1","observation_kind":"scheduler_cache_metadata","num_blocks":16,"mutation_seq":last,"mutations":ms})).unwrap()
}
fn reset(seq:u64)->Value {json!({"mutation_seq":seq,"type":"cache_reset","group_id":0,"block_size":128,"reason":if seq==1 {"initial_empty"} else {"explicit_reset"}})}
fn insert(seq:u64,bid:u64,n:usize,seed:u32)->Value {json!({"mutation_seq":seq,"type":"cache_insert","group_id":0,"block_size":128,"block_id":bid,"cache_hash":format!("{:064x}",seed+n as u32),"logical_index":n-1,"prefix_token_ids":vec![seed;n*128]})}
async fn populate(ctx:&Arc<Mutex<ScheduleContext>>,epoch:&str,seed:Option<u32>) {
    let ms=match seed {Some(seed)=>vec![reset(1),insert(2,1,1,seed),insert(3,2,2,seed)],None=>vec![reset(1)]};
    let j=journal(ms.clone(),ms.len() as u64);let mut c=ctx.lock().await;
    let (s,t)=c.kv_journal.prepare(&j,epoch,128).unwrap();c.kv_journal=s;c.block_hash=t;
}
async fn decision(name:&str,p:&[Arc<Mutex<ScheduleContext>>],id:u64,seed:u32,expected:usize)->Entry {
    let (e,_rx)=entry(id,vec![seed;313]);let obs=capture_observations(&e,p).await;
    let rows:Vec<_>=obs.iter().map(|o|json!({"engine":o.idx,"enabled":o.routing_enabled,"hits":o.hit_blocks*128,"bs":o.bs,"waiting":o.waiting,"prefill":o.queued_tokens,"all_tokens":o.all_tokens,"work":prefill_tokens(&e.request,o),"score":prefill_tokens(&e.request,o)*(o.bs+1)})).collect();
    let actual=LmetricQ::schedule(&e,p,&mut ()).await;println!("DUAL_CPU {}",json!({"scenario":name,"candidates":rows,"expected":expected,"actual":actual,"predicted_hits":e.block_hash_state.pred_hit_tokens()}));
    assert_eq!(actual,Some(expected));e
}
#[tokio::test]
async fn dual_cpu_a_only_cache() {
    let p=vec![context(),context()];populate(&p[0],"A",Some(7)).await;populate(&p[1],"B",None).await;
    let e=decision("A_only",&p,101,7,0).await;assert_eq!(e.block_hash_state.pred_hit_tokens(),256);
    assert_eq!(ledger(&*p[1].lock().await),json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));
}
#[tokio::test]
async fn dual_cpu_b_only_cache() {
    let p=vec![context(),context()];populate(&p[0],"A",None).await;populate(&p[1],"B",Some(7)).await;
    let e=decision("B_only",&p,102,7,1).await;assert_eq!(e.block_hash_state.pred_hit_tokens(),256);
}
#[tokio::test]
async fn dual_cpu_cache_load_conflict() {
    let p=vec![context(),context()];populate(&p[0],"A",Some(7)).await;populate(&p[1],"B",None).await;
    {let mut c=p[0].lock().await;c.lmetric.bs=2;c.lmetric.waiting_reqs=1;c.lmetric.prefill_tokens=128;c.lmetric.all_tokens=600;}
    let e=decision("cache_load_conflict",&p,103,7,1).await;assert_eq!(e.block_hash_state.pred_hit_tokens(),0);
}
#[tokio::test]
async fn dual_cpu_physical_namespace_and_reset_isolation() {
    let p=vec![context(),context()];populate(&p[0],"A",Some(7)).await;populate(&p[1],"B",Some(9)).await;
    assert_ne!(p[0].lock().await.kv_journal.mapping_view(),p[1].lock().await.kv_journal.mapping_view());
    decision("same_bids_distinct_tokens_A",&p,104,7,0).await;
    decision("same_bids_distinct_tokens_B",&p,105,9,1).await;
    {let mut c=p[0].lock().await;let (s,t)=c.kv_journal.prepare(&journal(vec![reset(4)],4),"A",128).unwrap();c.kv_journal=s;c.block_hash=t;}
    let e=decision("A_reset_B_preserved",&p,106,9,1).await;assert_eq!(e.block_hash_state.pred_hit_tokens(),256);
    assert_eq!(p[1].lock().await.kv_journal.last,3);
}
struct ChannelSteps {rx:mpsc::UnboundedReceiver<EngineStepOutput>}
#[async_trait::async_trait]
impl EngineStepReceiver for ChannelSteps {
    async fn recv_step(&mut self)->Result<EngineStepOutput,EngineClientError> {self.rx.recv().await.ok_or(EngineClientError::StreamEnded)}
}
#[tokio::test]
async fn dual_cpu_failure_latches_and_admission_rechecks() {
    let p=vec![context(),context()];populate(&p[0],"A",Some(7)).await;populate(&p[1],"B",None).await;
    let (e,_)=entry(107,vec![7;313]);let obs=capture_observations(&e,&p).await;
    let bad=crate::engine::client::lmetric_test_convert(json!({"lmetric_protocol":"request-kv-v1","engine_epoch":"A","block_size":128,"kv_journal":journal(vec![reset(5)],5)}));
    let (_tx,cq)=channel(8);let (_et,er)=mpsc::unbounded_channel();
    completion_event_loop(0,Box::new(BadOnce {step:Some(bad)}),Arc::new(Mutex::new(())),cq,er,p[0].clone(),None).await;
    assert!(!p[0].lock().await.kv_journal.routable());
    assert!(!apply_default_after(&e,&p,0,&obs[0]).await);
    let e=decision("invalid_A_healthy_B",&p,108,7,1).await;assert_eq!(e.block_hash_state.pred_hit_tokens(),0);
    assert!(p[1].lock().await.kv_journal.routable());
    assert_eq!(ledger(&*p[0].lock().await),json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));
}
#[tokio::test]
async fn dual_cpu_independent_sequences_and_multiple_requests_complete() {
    use crate::engine::vllm_http::MetricSequence;
    let p=vec![context(),context()];let mut wqs=vec![];let mut sends=vec![];let mut tasks=vec![];
    let mut decoders=[MetricSequence::default(),MetricSequence::default()];
    for idx in 0..2 {
        let (tx,cq)=channel(8);wqs.push(tx);let (_et,er)=mpsc::unbounded_channel();let (stx,rx)=mpsc::unbounded_channel();sends.push(stx);
        let ctx=p[idx].clone();tasks.push(tokio::spawn(completion_event_loop(idx,Box::new(ChannelSteps{rx}),Arc::new(Mutex::new(())),cq,er,ctx,None)));
    }
    for wave in 0..2u64 {
        let mut responses=vec![];
        for idx in 0..2 {
            let rid=200+wave*2+idx as u64;let seed=7+idx as u32;
            let (e,rx)=entry(rid,vec![seed;313]);assert_eq!(LmetricQ::schedule(&e,&p,&mut ()).await,Some(idx));wqs[idx].send(e).await.unwrap();responses.push(rx);
        }
        for idx in 0..2 {
            let rid=200+wave*2+idx as u64;let seed=7+idx as u32;let seq=1+wave*4;
            let mut ms=if wave==0 {vec![reset(1),insert(2,1,1,seed),insert(3,2,2,seed)]} else {vec![]};
            let b=if wave==0 {4}else {6};
            ms.push(json!({"mutation_seq":b,"type":"request_blocks","group_id":0,"block_size":128,"request_id":rid,"block_ids":[1,2,3]}));
            ms.push(json!({"mutation_seq":b+1,"type":"request_release","group_id":0,"block_size":128,"request_id":rid,"block_ids":[]}));
            let packet=json!({"lmetric_protocol":"request-kv-v1","engine_epoch":format!("engine-{idx}"),"step_id":wave+1,"block_size":128,"kv_groups":1,"kv_events":true,"event_kind":"completion","execution_mode":"sync","latency_kind":"schedule_to_completion_host_ms","latency":1,"prefill_tokens":if wave==0 {313}else {57},"kv_journal":journal(ms,b+1),"outputs":[{"request_id":rid,"new_token_ids":[1,2,3,4,5,6,7,8],"state":"PREFILL","is_finished":true,"hit_token_cnt":if wave==0 {0}else {256},"prev_computed_tokens":0}]});
            let _=seq;decoders[idx].parse(&packet.to_string()).unwrap();
            sends[idx].send(crate::engine::client::lmetric_test_convert(packet)).unwrap();
        }
        for rx in &mut responses {
            let count=tokio::time::timeout(Duration::from_secs(3),async {let mut n=0;loop {match rx.recv().await.unwrap().unwrap(){InferStreamResponse::Intermediate {..}=>n+=1,InferStreamResponse::End {generated_text,..}=>{assert_eq!(generated_text.generated_tokens,8);return n},_=>{}}}}).await.unwrap();assert_eq!(count,8);
        }
        for ctx in &p {let c=ctx.lock().await;assert!(c.kv_journal.requests.is_empty());assert_eq!(c.kv_journal.mapping_count(),2);assert_eq!(ledger(&c),json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));}
    }
    assert_ne!(p[0].lock().await.kv_journal.epoch,p[1].lock().await.kv_journal.epoch);
    println!("DUAL_CPU {}",json!({"scenario":"multiple_requests","requests":4,"per_backend":2,"same_step_ids":true,"same_mutation_ids":true,"same_physical_ids":true,"distinct_epochs":true,"ledgers_zero":true,"responses_verified":true}));
    for t in tasks {t.abort();let _=t.await;}
}

struct DisconnectedEngine;
#[async_trait::async_trait]
impl crate::engine::client::EngineClient for DisconnectedEngine {
    async fn add_request(&mut self,_:u64,_:&crate::gateway::validation::ValidGenerateRequest)->Result<(),EngineClientError> {unreachable!()}
    async fn abort_request(&mut self,_:u64)->Result<(),EngineClientError> {unreachable!()}
    fn get_error_rx(&mut self)->mpsc::UnboundedReceiver<u64> {mpsc::unbounded_channel().1}
    fn take_step_receiver(&mut self)->Box<dyn EngineStepReceiver> {Box::new(BadOnce {step:None})}
}
#[tokio::test]
async fn dual_cpu_disconnected_idle_work_loop_exits_without_panic() {
    let ctx=context();populate(&ctx,"A",None).await;
    let q=crate::scheduler::queue::TaskAssigner::new(1,vec![ctx.clone()]);
    let t=tokio::spawn(work_event_loop(0,Box::new(DisconnectedEngine),q,ctx.clone(),None));
    let exit=tokio::time::timeout(Duration::from_secs(3),t).await.expect("closed receiver must stop the idle work loop");
    assert!(exit.is_ok(),"an expected SSE disconnection must not panic the worker");
    assert!(!ctx.lock().await.kv_journal.routable());
}
