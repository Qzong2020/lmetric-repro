// SIMULATED EVENTS ONLY. Injected as a child of task_assignment in a /tmp copy.
// Runs the unchanged production completion_event_loop, types, policy and cache.
#[cfg(test)]
mod ascend_request_tests {
    use super::*;
    use crate::engine::client::{EngineClientError, EngineStepOutput};
    use crate::scheduler::kvcache::{BlockHashState, PrefixBlockHash};
    use crate::scheduler::policies::{lmetric::LmetricQ, policy_trait::Policy};
    use crate::scheduler::state::LMetric;
    use crate::gateway::validation::ValidGenerateRequest;
    use crate::types::GenerationParams;
    use serde_json::{json, Value};
    use std::collections::VecDeque;

    struct Replay { events: VecDeque<EngineStepOutput> }
    #[async_trait::async_trait]
    impl EngineStepReceiver for Replay {
        async fn recv_step(&mut self) -> Result<EngineStepOutput, EngineClientError> {
            self.events.pop_front().ok_or(EngineClientError::StreamEnded)
        }
    }
    fn step(v: Value) -> EngineStepOutput {
        // Execute the real HTTP DTO and real client conversion.
        crate::engine::client::lmetric_test_convert(v)
    }
    fn status(state: &str, finished: bool, hit: u64, tokens: Value) -> Value {
        json!({"request_id":101,"state":state,"is_finished":finished,
               "hit_token_cnt":hit,"new_token_ids":tokens})
    }
    fn entry() -> (Entry, mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>) {
        let tokens: Vec<u32> = (0..64).collect();
        let (response_tx, rx) = mpsc::unbounded_channel();
        (Entry {
            block_hash_state: BlockHashState::new(&tokens, 16),
            request: ValidGenerateRequest {
                request_id:101, messages:vec![], inputs:String::new(), input_length:64,
                truncate:0, decoder_input_details:false,
                params:GenerationParams { temperature:1.0, repetition_penalty:1.0,
                    top_k:0, top_p:1.0, seed:0, max_new_tokens:64,
                    stop_sequences:vec![], ignore_eos_token:false },
                top_n_tokens:0, input_tokens:tokens,
            },
            response_tx, span:tracing::Span::none(), temp_span:None,
            queue_time:tokio::time::Instant::now(), batch_time:Some(tokio::time::Instant::now()),
            generated_token_cnt:0, prev_token_time:None, time_of_per_token:None,
            max_time_between_tokens:Duration::ZERO,
        }, rx)
    }
    async fn replay(pred: usize, events: Vec<Value>, disconnect: bool) -> Value {
        let (e, rx) = entry();
        let ctx = Arc::new(Mutex::new(ScheduleContext {
            lmetric:LMetric::default(), block_hash:PrefixBlockHash::new(128),
        }));
        if pred > 0 {
            ctx.lock().await.block_hash.insert(&e.block_hash_state.get_hashes()[..pred],
                                              (1..=pred as u64).collect());
        }
        assert_eq!(LmetricQ::schedule(&e, &[ctx.clone()], &mut ()).await, Some(0));
        if disconnect { drop(rx); } else { // Keep receiver alive across replay.
            return replay_inner(e, ctx, events, Some(rx)).await;
        }
        replay_inner(e, ctx, events, None).await
    }
    async fn replay_inner(e: Entry, ctx: Arc<Mutex<ScheduleContext>>, events: Vec<Value>,
        _rx: Option<mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>>) -> Value {
        let (tx, cq) = channel(8); tx.send(e).await.unwrap();
        let (_err_tx, err_rx) = mpsc::unbounded_channel();
        completion_event_loop(0, Box::new(Replay { events:events.into_iter().map(step).collect() }),
            Arc::new(Mutex::new(())), cq, err_rx, ctx.clone(), None).await;
        let c = ctx.lock().await;
        json!({"bs":c.lmetric.bs,"waiting":c.lmetric.waiting_reqs,
               "prefill":c.lmetric.prefill_tokens,"all_tokens":c.lmetric.all_tokens})
    }
    fn record(name: &str, v: &Value) {
        println!("AUDIT_RESULT {}", json!({"case":name,"simulated":true,"result":v}));
    }

    fn v1(mut event: Value) -> Value {
        event["lmetric_protocol"] = json!("request-v1");
        event["block_size"] = json!(16);
        event
    }
    fn drained(v: &Value) { assert_eq!(*v,json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0})); }

    #[tokio::test]
    async fn ascend_request_prefill_prediction_matrix() {
        for (pred,actual) in [(0,0),(2,32),(1,32),(2,16),(4,48)] {
            for chunk in [false,true] {
                let work=64-actual;
                let mut es=vec![];
                if chunk { es.push(v1(json!({"prefill_tokens":work/2}))); }
                es.push(v1(json!({"prefill_tokens":if chunk {work-work/2} else {work},
                    "outputs":[status("PREFILL",false,actual,json!([9]))]})));
                let mid=replay(pred,es.clone(),false).await;
                assert_eq!(mid["prefill"],0);assert_eq!(mid["waiting"],0);
                es.push(v1(json!({"outputs":[status("DECODE",true,actual,json!([10]))]})));
                let end=replay(pred,es,false).await;drained(&end);
                record(&format!("request_v1_pred_{}_actual_{}_chunk_{}",pred*16,actual,chunk),&end);
            }
        }
    }
    #[tokio::test]
    async fn ascend_request_first_output_finish_matrix() {
        for (pred,actual) in [(0,0),(0,32),(2,16),(1,32),(4,48)] {
            let end=replay(pred,vec![v1(json!({"prefill_tokens":64-actual,
                "outputs":[status("PREFILL",true,actual,json!([9]))]}))],false).await;
            drained(&end);record(&format!("request_v1_first_finish_{}_{}",pred*16,actual),&end);
        }
    }
    #[tokio::test]
    async fn ascend_request_legacy_regression_still_reproduced() {
        let v=replay(2,vec![json!({"prefill_tokens":64,
            "outputs":[status("PREFILL",false,32,json!([9]))]})],false).await;
        assert_eq!(v["prefill"],-32);record("legacy_negative_ledger_preserved",&v);
    }
    #[tokio::test]
    async fn ascend_request_multi_token_finish() {
        let v=replay(0,vec![v1(json!({"prefill_tokens":64,
            "outputs":[status("PREFILL",false,0,json!([9,10]))]})),
            v1(json!({"outputs":[status("DECODE",false,0,json!([11,12,13]))]})),
            v1(json!({"outputs":[status("DECODE",true,0,json!([14]))]}))],false).await;
        drained(&v);record("multi_token_no_finish_residue",&v);
    }
    #[test]
    fn ascend_request_128_tail_and_multi_block_append() {
        for size in [16,32,128] {
            for n in [0,1,15,16,48,127,128,129,255,256,300] {
                let tokens:Vec<u32>=(0..n).collect();
                let mut h=BlockHashState::new(&tokens,size);
                assert_eq!(h.get_hashes().len(),tokens.len()/size);
                let extra:Vec<u32>=(n..n+300).collect();h.append_tokens(&extra);
                let all:Vec<u32>=(0..n+300).collect();
                let reference=BlockHashState::new(&all,size);
                assert_eq!(h.get_hashes(),reference.get_hashes());
            }
        }
    }
    #[test]
    fn ascend_request_128_physical_mapping_uses_local_hash() {
        let all:Vec<u32>=(0..256).collect();let mut h=BlockHashState::new(&all,128);
        h.set_bids(vec![7,8]);
        let hashes=h.get_onto_hashes(&vec![7,8]).unwrap().to_vec();
        let mut tree=PrefixBlockHash::new(128);tree.insert(&hashes,vec![7,8]);
        assert_eq!(tree.get(&hashes),2);
        assert_eq!(BlockHashState::new(&(0..48).collect(),128).get_hashes().len(),0);
    }
    #[tokio::test]
    async fn ascend_request_mismatched_block_size_rejected_before_accounting() {
        let mut event=v1(json!({"prefill_tokens":64,
            "outputs":[status("PREFILL",false,0,json!([9]))]}));
        event["block_size"]=json!(128);
        let v=replay(0,vec![event],false).await;
        assert_eq!(v,json!({"bs":1,"waiting":1,"prefill":64,"all_tokens":64}));
        record("block_size_mismatch_no_event_applied",&v);
    }
    #[test]
    fn ascend_request_sequence_validation() {
        use crate::engine::vllm_http::MetricSequence;
        fn packet(seq:u64)->Value {json!({"lmetric_protocol":"request-v1","engine_epoch":"synthetic",
            "step_id":seq,"latency":12,"latency_kind":"schedule_to_completion_host_ms",
            "execution_mode":"async","kv_events":false,"block_size":128,"kv_groups":1,"prefill_tokens":16,"outputs":[]})}
        let duplicate=packet(1);
        let gap=packet(3);
        let mut changed=packet(2);changed["engine_epoch"]=json!("other");
        let mut missing=packet(2);missing.as_object_mut().unwrap().remove("prefill_tokens");
        let mut invalid=packet(2);invalid["outputs"]=json!([{"request_id":101}]);
        let mut groups=packet(2);groups["kv_groups"]=json!(2);
        for bad in [duplicate,gap,changed,missing,invalid,groups] {
            let mut sequence=MetricSequence::default();
            assert!(sequence.parse(&packet(1).to_string()).is_ok());
            assert!(sequence.parse(&bad.to_string()).is_err());
            assert!(sequence.failed());
            assert!(sequence.parse(&packet(2).to_string()).is_err());
        }
        let mut fresh=MetricSequence::default();
        assert!(fresh.parse(&packet(1).to_string()).is_ok());
        assert!(fresh.parse(&packet(2).to_string()).is_ok());
        assert!(MetricSequence::default().parse("{}").is_ok());
    }
    struct Limited { inner:Box<dyn EngineStepReceiver>, left:usize }
    #[async_trait::async_trait]
    impl EngineStepReceiver for Limited {
        async fn recv_step(&mut self)->Result<EngineStepOutput,EngineClientError> {
            if self.left==0 {return Err(EngineClientError::StreamEnded);}
            let step=self.inner.recv_step().await?;self.left-=1;
            assert_eq!(step.lmetric_protocol.as_deref(),Some("request-v1"));
            assert_eq!(step.step_id,4-self.left as u64);
            Ok(step)
        }
    }
    #[tokio::test]
    #[ignore = "requires scripts/run_ascend_minimal_tests.py synthetic server"]
    async fn ascend_request_real_http_sse_to_completion_loop() {
        use crate::engine::client::{EngineClient,VllmEngineClient};
        use crate::engine::vllm_http::VllmClient;
        let url=std::env::var("LMETRIC_SYNTHETIC_URL").expect("run scripts/run_ascend_minimal_tests.py");
        let mut client=VllmEngineClient::new(VllmClient::new(&url,"SYNTHETIC-NO-MODEL"));
        let _errors=client.get_error_rx();
        let receiver=client.take_step_receiver();
        let (mut e,rx)=entry();
        e.request.input_length=384;
        e.request.input_tokens=(0..384).collect();
        e.block_hash_state=BlockHashState::new(&e.request.input_tokens,128);
        let ctx=Arc::new(Mutex::new(ScheduleContext {
            lmetric:LMetric::default(),block_hash:PrefixBlockHash::new(128)}));
        ctx.lock().await.block_hash.insert(&e.block_hash_state.get_hashes()[..1],vec![1]);
        assert_eq!(LmetricQ::schedule(&e,&[ctx.clone()],&mut ()).await,Some(0));
        let (tx,cq)=channel(8);tx.send(e).await.unwrap();
        let (_err_tx,err_rx)=mpsc::unbounded_channel();
        let trigger=tokio::spawn(async move {
            let response=reqwest::Client::new().post(format!("{url}/test/run")).send().await.unwrap();
            assert_eq!(response.status(),200);
        });
        tokio::time::timeout(Duration::from_secs(15),completion_event_loop(0,
            Box::new(Limited {inner:receiver,left:4}),Arc::new(Mutex::new(())),
            cq,err_rx,ctx.clone(),None)).await.expect("synthetic SSE timeout");
        trigger.await.unwrap();
        let c=ctx.lock().await;
        let v=json!({"bs":c.lmetric.bs,"waiting":c.lmetric.waiting_reqs,
            "prefill":c.lmetric.prefill_tokens,"all_tokens":c.lmetric.all_tokens});
        drained(&v);record("REAL_HTTP_SSE_SYNTHETIC_to_Rust_completion_loop",&v);
        drop(rx);
    }
}
