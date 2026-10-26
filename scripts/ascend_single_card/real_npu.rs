// Real model acceptance driver. Linked only into an isolated copy of the Router.
#[cfg(test)]
mod real_npu_validation {
    use super::*;
    use crate::engine::client::{EngineClientError,EngineStepOutput};
    use crate::engine::vllm_http::VllmClient;
    use crate::engine::client::{EngineClient,VllmEngineClient};
    use crate::scheduler::kvcache::{BlockHashState,PrefixBlockHash};
    use crate::scheduler::policies::{lmetric::LmetricQ,policy_trait::Policy};
    use crate::scheduler::state::LMetric;
    use crate::gateway::validation::ValidGenerateRequest;
    use crate::types::GenerationParams;
    use serde_json::{json,Value};
    fn ledger(c:&ScheduleContext)->Value {json!({"bs":c.lmetric.bs,"waiting":c.lmetric.waiting_reqs,"prefill":c.lmetric.prefill_tokens,"all_tokens":c.lmetric.all_tokens})}
    struct Observed { inner:Box<dyn EngineStepReceiver>,ctx:Arc<Mutex<ScheduleContext>>,finished:bool,last:Option<u64>,events:Arc<Mutex<Vec<EngineStepOutput>>> }
    #[async_trait::async_trait]
    impl EngineStepReceiver for Observed {
      async fn recv_step(&mut self)->Result<EngineStepOutput,EngineClientError> {
       if let Some(seq)=self.last {println!("REAL_APPLIED {}",json!({"step_id":seq,"ledger":ledger(&*self.ctx.lock().await)}));}
       let was_finished=self.finished;
       let step=if was_finished {
         match tokio::time::timeout(Duration::from_secs(2),self.inner.recv_step()).await {
          Ok(r)=>r.expect("post-finish SSE failed; stop session"),
          Err(_)=>{println!("REAL_DRAINED_IDLE");return Err(EngineClientError::StreamEnded);}
         }
       } else {self.inner.recv_step().await.expect("real SSE receiver failed; stop session")};
       if was_finished {assert_eq!(step.prefill_tokens,0);assert!(step.outputs.is_empty());}
       assert_eq!(step.lmetric_protocol.as_deref(),Some("request-v1"));
       assert_eq!(step.lmetric_block_size,Some(128));
       let mut events=self.events.lock().await;
       assert_eq!(step.step_id,events.len() as u64+1);
       for o in &step.outputs {assert_eq!(o.request_id,2026102501);}
       self.finished |= step.outputs.iter().any(|o|o.is_finished);self.last=Some(step.step_id);
       println!("REAL_RECEIVED {}",serde_json::to_string(&step).unwrap());events.push(step.clone());Ok(step)
      }
    }
    #[tokio::test]
    #[ignore="requires real NPU engine and isolated driver"]
    async fn real_npu_sse_completion() {
      let url=std::env::var("LMETRIC_REAL_URL").unwrap();
      let data:Value=serde_json::from_str(&std::fs::read_to_string(std::env::var("LMETRIC_REAL_PROMPT").unwrap()).unwrap()).unwrap();
      let tokens:Vec<u32>=serde_json::from_value(data["tokens"].clone()).unwrap();
      let n=tokens.len();let (response_tx,_rx)=mpsc::unbounded_channel();
      let e=Entry {block_hash_state:BlockHashState::new(&tokens,128),request:ValidGenerateRequest {request_id:2026102501,messages:vec![],inputs:data["text"].as_str().unwrap().into(),input_length:n as u32,truncate:0,decoder_input_details:false,params:GenerationParams {temperature:1.0,repetition_penalty:1.0,top_k:0,top_p:1.0,seed:0,max_new_tokens:8,stop_sequences:vec![],ignore_eos_token:false},top_n_tokens:0,input_tokens:tokens},response_tx,span:tracing::Span::none(),temp_span:None,queue_time:tokio::time::Instant::now(),batch_time:Some(tokio::time::Instant::now()),generated_token_cnt:0,prev_token_time:None,time_of_per_token:None,max_time_between_tokens:Duration::ZERO};
      let ctx=Arc::new(Mutex::new(ScheduleContext {lmetric:LMetric::default(),block_hash:PrefixBlockHash::new(128)}));
      assert_eq!(LmetricQ::schedule(&e,&[ctx.clone()],&mut ()).await,Some(0));
      println!("REAL_ADMITTED {}",ledger(&*ctx.lock().await));
      let mut client=VllmClient::new(&url,"ascend-small");let _errors=client.get_error_rx();
      let mut engine=VllmEngineClient::new(client.clone());let receiver=engine.take_step_receiver();
      let (tx,cq)=channel(8);let (_err_tx,err_rx)=mpsc::unbounded_channel();
      let events=Arc::new(Mutex::new(vec![]));let receiver=Observed {inner:receiver,ctx:ctx.clone(),finished:false,last:None,events:events.clone()};
      let loop_ctx=ctx.clone();
      let task=tokio::spawn(async move {completion_event_loop(0,Box::new(receiver),Arc::new(Mutex::new(())),cq,err_rx,loop_ctx,None).await;});
      let handshake=std::path::PathBuf::from(std::env::var("LMETRIC_HANDSHAKE_FILE").unwrap());
      tokio::time::timeout(Duration::from_secs(30),async {while !handshake.exists() {tokio::time::sleep(Duration::from_millis(20)).await;}}).await.expect("SSE must connect before POST");
      println!("REAL_SUBSCRIBED_BEFORE_POST");
      let http=client.add_request(2026102501,&e.request).await;tx.send(e).await.unwrap();
      let response=http.await.unwrap().unwrap();
      let status=response.status();let body:Value=response.json().await.unwrap();println!("REAL_HTTP {}",body);assert_eq!(status,200);
      tokio::time::timeout(Duration::from_secs(90),task).await.expect("real completion timeout").unwrap();
      let final_ledger=ledger(&*ctx.lock().await);assert_eq!(final_ledger,json!({"bs":0,"waiting":0,"prefill":0,"all_tokens":0}));
      let es=events.lock().await;assert_eq!(es.iter().map(|e|e.prefill_tokens).sum::<usize>(),n);
      let output_tokens:Vec<u32>=es.iter().flat_map(|e|e.outputs.iter().flat_map(|o|o.new_token_ids.clone())).collect();
      assert_eq!(output_tokens.len(),8);assert_eq!(body["usage"]["prompt_tokens"],n);assert_eq!(body["usage"]["completion_tokens"],8);
      assert!(es.iter().any(|e|e.prefill_tokens>0 && e.outputs.is_empty()));
      println!("REAL_RESULT {}",json!({"npu":true,"events":es.len(),"prompt_tokens":n,"completion_tokens":output_tokens,"ledger":final_ledger}));
    }
}
