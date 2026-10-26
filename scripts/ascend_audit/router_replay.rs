// SIMULATED EVENTS ONLY. Injected as a child of task_assignment in a /tmp copy.
// Runs the unchanged production completion_event_loop, types, policy and cache.
#[cfg(test)]
mod ascend_audit_tests {
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
        crate::engine::client::ascend_audit_convert(v)
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
    #[test]
    fn ascend_audit_serde() {
        use crate::engine::vllm_http::VllmMetric;
        let v: VllmMetric = serde_json::from_value(json!({"unknown":7})).unwrap();
        assert_eq!(v.prefill_tokens,0); assert_eq!(v.step_id,0); assert!(v.op_exec_log.is_none());
        let v: VllmMetric = serde_json::from_value(json!({"outputs":[{"request_id":101,"ttft":1}]})).unwrap();
        assert_eq!(v.outputs[0].state,"DECODE");
        assert!(serde_json::from_value::<VllmMetric>(json!({"outputs":[{}]})).is_err());
        assert!(serde_json::from_value::<VllmMetric>(json!({"outputs":[{"request_id":"cmpl-101-0-abcd"}]})).is_err());
        let full: Value = serde_json::from_str(include_str!("__FULL_JSON__")).unwrap();
        let m: VllmMetric = serde_json::from_value(full.clone()).unwrap();
        let roundtrip = serde_json::to_value(m).unwrap();
        assert_eq!(roundtrip["cur_used_block_ids"],full["cur_used_block_ids"]);
        let converted = step(full); assert_eq!(converted.latency,12);
        assert_eq!(converted.outputs[0].hit_token_cnt,32);
        record("serde_and_mapping",&json!({"passed":true}));
    }
    #[tokio::test]
    async fn ascend_audit_prefill_arithmetic() {
        for (pred,actual) in [(0,0),(2,32),(1,32),(2,16)] {
            let v = replay(pred,vec![json!({"prefill_tokens":64,"outputs":[status("PREFILL",false,actual,json!([9]))]})],false).await;
            assert_eq!(v["prefill"],-(actual as i64));
            assert_eq!(v["waiting"],0);
            record(&format!("yaullm_prefill_pred_{}_actual_{}",pred*16,actual),&v);
        }
    }
    #[tokio::test]
    async fn ascend_audit_chunk_and_finish() {
        let v = replay(0,vec![json!({"prefill_tokens":32}),
            json!({"prefill_tokens":32,"outputs":[status("PREFILL",false,0,json!([9]))]}),
            json!({"outputs":[status("DECODE",false,0,json!([10]))]}),
            json!({"outputs":[status("DECODE",true,0,json!([11]))]})],false).await;
        assert_eq!(v,json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));
        record("chunk_prefill_decode_finish",&v);
    }
    #[tokio::test]
    async fn ascend_audit_first_output_finish() {
        let v = replay(2,vec![json!({"prefill_tokens":32,"outputs":[status("PREFILL",true,16,json!([9]))]})],false).await;
        // Correction is skipped on a first output that also finishes: 32 admitted - 32 reported.
        assert_eq!(v["prefill"],0); record("finish_skips_hit_correction",&v);
        let v = replay(2,vec![json!({"prefill_tokens":48,"outputs":[status("PREFILL",true,16,json!([9]))]})],false).await;
        assert_eq!(v["prefill"],-16); record("compute_only_finish_hit_mismatch",&v);
    }
    #[tokio::test]
    async fn ascend_audit_preemption_only() {
        let v = replay(0,vec![json!({"prefill_tokens":32}),json!({"preempted_ids":[101]})],false).await;
        assert_eq!(v,json!({"bs":1,"waiting":1,"prefill":32,"all_tokens":64}));
        record("preemption_has_no_ledger_rebuild",&v);
    }
    #[tokio::test]
    async fn ascend_audit_cancel_ack() {
        let v = replay(0,vec![json!({"prefill_tokens":64,"outputs":[status("PREFILL",false,0,json!([9]))]}),
                              json!({"aborted_requests":[101]})],true).await;
        assert_eq!(v,json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));
        record("cancel_after_first_output_then_ack",&v);
    }
    #[tokio::test]
    #[should_panic(expected="not captured in exception context")]
    async fn ascend_audit_unsolicited_abort_ack_panics() {
        replay(0,vec![json!({"aborted_requests":[101]})],false).await;
    }
    #[tokio::test]
    async fn ascend_audit_multi_token_finish_residue() {
        let v=replay(0,vec![json!({"prefill_tokens":64,"outputs":[status("PREFILL",false,0,json!([9,10]))]}),
            json!({"outputs":[status("DECODE",true,0,json!([11]))]})],false).await;
        assert_eq!(v["all_tokens"],1); record("multi_token_event_finish_residue",&v);
    }
    #[tokio::test]
    async fn ascend_audit_duplicate_decode() {
        let v = replay(0,vec![json!({"prefill_tokens":64,"step_id":1,"outputs":[status("PREFILL",false,0,json!([9]))]}),
            json!({"step_id":2,"outputs":[status("DECODE",false,0,json!([10]))]}),
            json!({"step_id":2,"outputs":[status("DECODE",false,0,json!([10]))]})],false).await;
        assert_eq!(v["all_tokens"],67); record("duplicate_decode_is_applied_twice",&v);
    }
    #[tokio::test]
    #[should_panic]
    async fn ascend_audit_duplicate_prefill_panics() {
        replay(0,vec![json!({"prefill_tokens":64,"step_id":1,"outputs":[status("PREFILL",false,0,json!([9]))]}),
            json!({"prefill_tokens":64,"step_id":1,"outputs":[status("PREFILL",false,0,json!([9]))]})],false).await;
    }
    #[tokio::test]
    async fn ascend_audit_kv_remove_before_insert() {
        let (e,rx)=entry();
        let ctx=Arc::new(Mutex::new(ScheduleContext { lmetric:LMetric::default(),
            block_hash:PrefixBlockHash::new(128) }));
        let old=BlockHashState::new(&(100..164).collect(),16);
        ctx.lock().await.block_hash.insert(&old.get_hashes()[..1],vec![1]);
        assert_eq!(LmetricQ::schedule(&e,&[ctx.clone()],&mut ()).await,Some(0));
        let v=replay_inner(e,ctx.clone(),vec![json!({"prefill_tokens":64,
            "outputs":[status("PREFILL",false,0,json!([9]))],"evicted_block_ids":[1],
            "cur_used_block_ids":{"101":[1,2,3,4]},"new_block_hashes_ids":{"101":[1,2,3,4]}})],Some(rx)).await;
        assert_eq!(v["prefill"],0);
        assert_eq!(ctx.lock().await.block_hash.get(old.get_hashes()),0);
        let expected=BlockHashState::new(&(0..64).collect(),16);
        assert_eq!(ctx.lock().await.block_hash.get(expected.get_hashes()),4);
        record("kv_remove_before_insert_full_loop",&v);
    }
    #[test]
    #[should_panic(expected="nblock_canary")]
    fn ascend_audit_radix_parent_removal_panics() {
        let mut cache=PrefixBlockHash::new(128);
        let a=BlockHashState::new(&(0..64).collect(),16);
        cache.insert(&a.get_hashes()[..2],vec![1,2]);
        cache.remove(vec![1]);
    }
    #[test]
    fn ascend_audit_kv_reuse_order() {
        let mut cache=PrefixBlockHash::new(128);
        let a=BlockHashState::new(&(0..64).collect(),16);
        let b=BlockHashState::new(&(100..164).collect(),16);
        cache.insert(&a.get_hashes()[..1],vec![7]);
        cache.remove(vec![7]); cache.insert(&b.get_hashes()[..1],vec![7]);
        assert_eq!(cache.get(a.get_hashes()),0); assert_eq!(cache.get(b.get_hashes()),1);
        cache.remove(vec![7]); assert_eq!(cache.get(b.get_hashes()),0);
        record("kv_reuse_final_mapping",&json!({"passed":true}));
    }
    #[test]
    #[should_panic]
    fn ascend_audit_non16_partial_hash_panics() {
        // Production constructor uses a literal 16 to detect partial blocks.
        let mut h=BlockHashState::new(&(0..48).collect(),128);
        h.set_bids(vec![1]);
        h.get_onto_hashes(&vec![1]).unwrap();
        // A 48-token input has zero full blocks at the Ascend default of 128.
        // The production constructor incorrectly exposes one hash.
        assert_eq!(h.get_hashes().len(),0);
    }
}
