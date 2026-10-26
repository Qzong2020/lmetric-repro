// CPU-negative and real-NPU integration drivers; events are never generated here.
#[cfg(all(test, feature = "lmetric-q", feature = "vllm-backend"))]
mod kv_journal_integration_tests {
    use super::*;
    use crate::engine::client::{EngineClientError,EngineStepOutput};
    use crate::engine::vllm_http::VllmClient;
    use crate::scheduler::kvcache::{BlockHash,BlockHashState,PrefixBlockHash};
    use crate::scheduler::policies::{lmetric::LmetricQ,policy_trait::Policy};
    use crate::scheduler::state::LMetric;
    use crate::gateway::validation::ValidGenerateRequest;
    use crate::types::GenerationParams;
    use serde_json::{json,Value};
    fn entry(id:u64,tokens:Vec<u32>)->(Entry,mpsc::UnboundedReceiver<Result<InferStreamResponse,InferError>>) {
        let (response_tx,rx)=mpsc::unbounded_channel();let n=tokens.len();
        (Entry {block_hash_state:BlockHashState::new(&tokens,128),request:ValidGenerateRequest {request_id:id,messages:vec![],inputs:String::new(),input_length:n as u32,truncate:0,decoder_input_details:false,params:GenerationParams {temperature:1.0,repetition_penalty:1.0,top_k:0,top_p:1.0,seed:0,max_new_tokens:8,stop_sequences:vec![],ignore_eos_token:false},top_n_tokens:0,input_tokens:tokens},response_tx,span:tracing::Span::none(),temp_span:None,queue_time:tokio::time::Instant::now(),batch_time:Some(tokio::time::Instant::now()),generated_token_cnt:0,prev_token_time:None,time_of_per_token:None,max_time_between_tokens:Duration::ZERO},rx)
    }
    fn context()->Arc<Mutex<ScheduleContext>> {Arc::new(Mutex::new(ScheduleContext {lmetric:LMetric::default(),block_hash:PrefixBlockHash::new(16),kv_journal:Default::default()}))}
    fn ledger(c:&ScheduleContext)->Value {json!({"bs":c.lmetric.bs,"waiting":c.lmetric.waiting_reqs,"prefill":c.lmetric.prefill_tokens,"all_tokens":c.lmetric.all_tokens})}
    struct BadOnce {step:Option<EngineStepOutput>}
    #[async_trait::async_trait]
    impl EngineStepReceiver for BadOnce {async fn recv_step(&mut self)->Result<EngineStepOutput,EngineClientError> {self.step.take().ok_or(EngineClientError::StreamEnded)}}
    #[tokio::test]
    async fn kv_journal_failure_stops_before_accounting_and_disables_admission() {
        let ctx=context();let (e,_rx)=entry(101,vec![7;313]);assert_eq!(LmetricQ::schedule(&e,&[ctx.clone()],&mut ()).await,Some(0));
        let bad=crate::engine::client::lmetric_test_convert(json!({"lmetric_protocol":"request-kv-v1","engine_epoch":"e","block_size":128,"kv_journal":{"version":"kv-journal-v1","observation_kind":"scheduler_cache_metadata","num_blocks":16,"mutation_seq":2,"mutations":[{"mutation_seq":2,"type":"cache_reset","group_id":0,"block_size":128,"reason":"explicit_reset"}]},"prefill_tokens":313}));
        let (tx,cq)=channel(8);tx.send(e).await.unwrap();let (_et,er)=mpsc::unbounded_channel();
        completion_event_loop(0,Box::new(BadOnce {step:Some(bad)}),Arc::new(Mutex::new(())),cq,er,ctx.clone(),None).await;
        assert_eq!(ledger(&*ctx.lock().await),json!({"bs":1,"waiting":1,"prefill":313,"all_tokens":313}));
        assert!(!ctx.lock().await.kv_journal.routable());
        let (e,_rx)=entry(102,vec![7;313]);assert_eq!(LmetricQ::schedule(&e,&[ctx.clone()],&mut ()).await,None);
    }
    #[tokio::test]
    async fn kv_journal_admission_rechecks_after_observation() {
        use crate::scheduler::policies::dsl_runtime::{capture_observations,apply_default_after};
        let ctx=context();let (e,_rx)=entry(101,vec![7;313]);
        let obs=capture_observations(&e,&[ctx.clone()]).await;
        ctx.lock().await.invalidate_kv("stream failed after scoring".into());
        assert!(!apply_default_after(&e,&[ctx.clone()],0,&obs[0]).await);
        assert_eq!(ledger(&*ctx.lock().await),json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));
    }

    #[tokio::test]
    async fn kv_journal_same_size_rebuild_must_invalidate_hit_observation() {
        use crate::engine::kv_journal::KvJournal;
        use crate::scheduler::policies::dsl_runtime::{capture_observations,apply_default_after};
        let ctx=context();
        let journal=|ms:Value,last:u64|serde_json::from_value::<KvJournal>(json!({"version":"kv-journal-v1","observation_kind":"scheduler_cache_metadata","num_blocks":16,"mutation_seq":last,"mutations":ms})).unwrap();
        let first=journal(json!([{"mutation_seq":1,"type":"cache_reset","group_id":0,"block_size":128,"reason":"initial_empty"},{"mutation_seq":2,"type":"cache_insert","group_id":0,"block_size":128,"block_id":1,"cache_hash":"aa","logical_index":0,"prefix_token_ids":vec![7;128]}]),2);
        {let mut c=ctx.lock().await;let (state,tree)=c.kv_journal.prepare(&first,"e",128).unwrap();c.kv_journal=state;c.block_hash=tree;}
        let (e,_rx)=entry(101,vec![7;313]);let obs=capture_observations(&e,&[ctx.clone()]).await;assert_eq!(obs[0].hit_blocks,1);
        let second=journal(json!([{"mutation_seq":3,"type":"cache_remove","group_id":0,"block_size":128,"block_id":1,"cache_hash":"aa"},{"mutation_seq":4,"type":"cache_insert","group_id":0,"block_size":128,"block_id":1,"cache_hash":"bb","logical_index":0,"prefix_token_ids":vec![9;128]}]),4);
        {let mut c=ctx.lock().await;let (state,tree)=c.kv_journal.prepare(&second,"e",128).unwrap();c.kv_journal=state;c.block_hash=tree;}
        assert!(apply_default_after(&e,&[ctx.clone()],0,&obs[0]).await);
        assert_eq!(e.block_hash_state.pred_hit_tokens(),0,"a same-size rebuilt tree must not reuse stale predicted hits");
    }

    #[tokio::test]
    async fn kv_gateway_final_frame_reports_all_actual_tokens() {
        let ctx=context();let (e,mut responses)=entry(101,vec![7;313]);
        assert_eq!(LmetricQ::schedule(&e,&[ctx.clone()],&mut ()).await,Some(0));
        let step=crate::engine::client::lmetric_test_convert(json!({"lmetric_protocol":"request-v1","block_size":128,"prefill_tokens":313,"outputs":[{"request_id":101,"new_token_ids":[1,2,3,4,5,6,7,8],"state":"PREFILL","is_finished":true,"hit_token_cnt":0}]}));
        let (tx,cq)=channel(8);tx.send(e).await.unwrap();let (_et,er)=mpsc::unbounded_channel();
        completion_event_loop(0,Box::new(BadOnce {step:Some(step)}),Arc::new(Mutex::new(())),cq,er,ctx.clone(),None).await;
        let mut tokens=0;let mut final_count=None;
        while let Ok(item)=responses.try_recv() {match item.unwrap() {
            InferStreamResponse::Intermediate {..}=>tokens+=1,
            InferStreamResponse::End {generated_text,..}=>final_count=Some(generated_text.generated_tokens),
            _=>{},
        }}
        assert_eq!(tokens,8);assert_eq!(final_count,Some(8));
        assert_eq!(ledger(&*ctx.lock().await),json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));
    }

    include!("dual_replica_tests.rs");

    struct Observed {inner:Box<dyn EngineStepReceiver>,ctx:Arc<Mutex<ScheduleContext>>,last:Option<u64>}
    #[async_trait::async_trait]
    impl EngineStepReceiver for Observed {
      async fn recv_step(&mut self)->Result<EngineStepOutput,EngineClientError> {
       if let Some(seq)=self.last {
        let c=self.ctx.lock().await;
        println!("KV_APPLIED {}",json!({"step_id":seq,"mutation_seq":c.kv_journal.last,"mappings":c.kv_journal.mapping_count(),"physical":c.kv_journal.mapping_view(),"requests":c.kv_journal.requests,"ledger":ledger(&c),"valid":c.kv_journal.routable()}));
       }
       let step=self.inner.recv_step().await?;
       assert_eq!(step.lmetric_protocol.as_deref(),Some("request-kv-v1"));self.last=Some(step.step_id);
       println!("KV_RECEIVED {}",serde_json::to_string(&step).unwrap());Ok(step)
      }
    }
    async fn idle(ctx:&Arc<Mutex<ScheduleContext>>) {
      tokio::time::timeout(Duration::from_secs(20),async {
       loop {let c=ctx.lock().await;if !c.kv_journal.routable(){std::panic!("journal invalid: {:?}",c.kv_journal.failed);}
        let done=c.lmetric.bs==0 && c.kv_journal.requests.is_empty();drop(c);
        if done {return;}tokio::time::sleep(Duration::from_millis(10)).await;
       }
      }).await.expect("request drain timeout");
    }
    #[tokio::test]
    #[ignore="requires bounded real single-card KV engine and fixture"]
    async fn kv_journal_real_npu_cache_lifecycle() {
      let url=std::env::var("LMETRIC_KV_URL").unwrap();
      let fixture:Value=serde_json::from_str(&std::fs::read_to_string(std::env::var("LMETRIC_KV_FIXTURE").unwrap()).unwrap()).unwrap();
      let receiver=crate::engine::client::lmetric_kv_connected_receiver(&url).await.unwrap();
      println!("KV_REAL_SUBSCRIBED_BEFORE_REQUEST");
      let ctx=context();let (tx,cq)=channel(32);let (_et,er)=mpsc::unbounded_channel();let taskctx=ctx.clone();
      let task=tokio::spawn(async move {completion_event_loop(0,Box::new(Observed {inner:receiver,ctx:taskctx.clone(),last:None}),Arc::new(Mutex::new(())),cq,er,taskctx,None).await;});
      let mut client=VllmClient::new(&url,"ascend-small");let _errors=client.get_error_rx();let http=reqwest::Client::new();
      let mut results=vec![];
      for (i,&prompt_index) in [0usize,0,1,2,3,4,0,0].iter().enumerate() {
        if i==7 {
         let before=ctx.lock().await.kv_journal.last;
         let response=http.post(format!("{url}/reset_prefix_cache")).send().await.unwrap();
         let status=response.status();let body=response.text().await.unwrap();println!("KV_REAL_RESET {}",json!({"status":status.as_u16(),"body":body}));assert!(status.is_success());
         tokio::time::timeout(Duration::from_secs(10),async {loop {
           let c=ctx.lock().await;assert!(c.kv_journal.routable());let reset=c.kv_journal.last>before && c.kv_journal.mapping_count()==0;drop(c);
           if reset {break;}tokio::time::sleep(Duration::from_millis(10)).await;
         }}).await.expect("idle reset must arrive over SSE without a new request");
        }
        let tokens:Vec<u32>=serde_json::from_value(fixture["prompts"][prompt_index]["tokens"].clone()).unwrap();
        let hashes=BlockHashState::new(&tokens,128).get_hashes().to_vec();
        let expected_before=ctx.lock().await.block_hash.get(&hashes)*128;
        if i==1 {assert_eq!(expected_before,256,"first repeated request must have real predicted hits");}
        if i==6 || i==7 {assert_eq!(expected_before,0,"evicted/reset prefix must not be predicted cached");}
        let raw_path=std::env::var("LMETRIC_KV_SSE_FILE").unwrap();
        let start_line=std::fs::read_to_string(&raw_path).unwrap_or_default().lines().count();
        let rid=202610250100+i as u64;let (e,_rx)=entry(rid,tokens);
        assert_eq!(LmetricQ::schedule(&e,&[ctx.clone()],&mut ()).await,Some(0));let predicted=e.block_hash_state.pred_hit_tokens();
        let pending=client.add_request(rid,&e.request).await;tx.send(e).await.unwrap();
        let response=tokio::time::timeout(Duration::from_secs(30),pending).await.unwrap().unwrap().unwrap();
        assert!(response.status().is_success());let body:Value=response.json().await.unwrap();
        idle(&ctx).await;assert_eq!(ledger(&*ctx.lock().await),json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));
        let frames:Vec<Value>=std::fs::read_to_string(&raw_path).unwrap().lines().map(|s|serde_json::from_str(s).unwrap()).collect();
        let first=frames.iter().flat_map(|f|f["outputs"].as_array().unwrap()).find(|o|o["request_id"]==rid && o["state"]=="PREFILL").expect("real first output missing");
        let actual=first["hit_token_cnt"].as_u64().unwrap();
        assert_eq!(actual,predicted as u64,"real backend hit must match current KV view");
        let work:u64=frames[start_line..].iter().map(|f|f["prefill_tokens"].as_u64().unwrap()).sum();
        assert_eq!(work,313-actual,"actual prompt work excludes cache hits");
        let c=ctx.lock().await;let row=json!({"actual_hits":actual,"prefill_work":work,"index":i,"request_id":rid,"prompt":prompt_index,"predicted_hits":predicted,"physical":c.kv_journal.mapping_view(),"mutation_seq":c.kv_journal.last,"ledger":ledger(&c),"response":body});drop(c);
        println!("KV_REAL_REQUEST {}",row);results.push(row);
      }
      tokio::time::sleep(Duration::from_millis(300)).await;
      let c=ctx.lock().await;assert!(c.kv_journal.routable());assert_eq!(c.kv_journal.capacity,8);assert!(c.kv_journal.requests.is_empty());
      println!("KV_REAL_RESULT {}",json!({"requests":results.len(),"ledger":ledger(&c),"last_mutation":c.kv_journal.last,"physical":c.kv_journal.mapping_view(),"valid":true}));drop(c);
      task.abort();let _=task.await;
    }
}
