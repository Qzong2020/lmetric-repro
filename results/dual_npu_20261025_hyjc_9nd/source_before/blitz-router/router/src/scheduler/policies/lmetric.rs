//! `lmetric-q` — the policy named after the lmetric paper.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! Select min by prefill_tokens(req, sctx) · (sctx.bs + 1)
//! ```
//!
//! The `(bs + 1)` term ensures idle replicas (`bs == 0`) are still
//! distinguishable by their prefill load alone — without `+1` an idle
//! replica would always tie at score 0 regardless of queued prefill.

use policy_dsl::policy;

policy! {
    name: LmetricQ,
    gctx: (),
    body: { select_min_by(&root_target(&observations), |o| prefill_tokens(req, o) * (o.bs + 1)) },
}

#[cfg(test)]
mod tests {
    use super::LmetricQ;
    use crate::gateway::validation::ValidGenerateRequest;
    use crate::scheduler::kvcache::{BlockHash, BlockHashState, PrefixBlockHash};
    use crate::scheduler::policies::{policy_trait::Policy, Entry};
    use crate::scheduler::state::LMetric;
    use crate::types::GenerationParams;
    use crate::ScheduleContext;
    use std::{sync::Arc, time::Duration};
    use tokio::sync::{mpsc, Mutex};

    fn entry(id: u64) -> Entry {
        let tokens: Vec<u32> = (0..64).collect();
        let (response_tx, _) = mpsc::unbounded_channel();
        Entry {
            block_hash_state: BlockHashState::new(&tokens, 16),
            request: ValidGenerateRequest {
                request_id: id, messages: vec![], inputs: String::new(),
                input_length: 64, truncate: 0, decoder_input_details: false,
                params: GenerationParams {
                    temperature: 1.0, repetition_penalty: 1.0, top_k: 0,
                    top_p: 1.0, seed: 0, max_new_tokens: 8,
                    stop_sequences: vec![], ignore_eos_token: false,
                },
                top_n_tokens: 0, input_tokens: tokens,
            },
            response_tx, span: tracing::Span::none(), temp_span: None,
            queue_time: tokio::time::Instant::now(), batch_time: None,
            generated_token_cnt: 0, prev_token_time: None, time_of_per_token: None,
            max_time_between_tokens: Duration::ZERO,
        }
    }

    fn pool(n: usize) -> Vec<Arc<Mutex<ScheduleContext>>> {
        (0..n).map(|_| Arc::new(Mutex::new(ScheduleContext {
            kv_journal: Default::default(), lmetric: LMetric::default(), block_hash: PrefixBlockHash::new(1024),
        }))).collect()
    }

    #[tokio::test]
    async fn balances_cache_affinity_against_pending_work_and_active_requests() {
        let req = entry(0);
        let p = pool(2);
        {
            let mut cached = p[0].lock().await;
            cached.block_hash.insert(&req.block_hash_state.get_hashes()[..3], vec![1, 2, 3]);
        }
        assert_eq!(LmetricQ::schedule(&req, &p, &mut ()).await, Some(0));
        assert_eq!(req.block_hash_state.pred_hit_tokens(), 48);
        let cached = p[0].lock().await;
        assert_eq!(cached.lmetric.bs, 1);
        assert_eq!(cached.lmetric.waiting_reqs, 1);
        assert_eq!(cached.lmetric.prefill_tokens, 16);
        drop(cached);
        // Cached but busy: (128 + 16) * (8 + 1) > (0 + 64) * (0 + 1).
        {
            let mut cached = p[0].lock().await;
            cached.lmetric.bs = 8;
            cached.lmetric.prefill_tokens = 128;
        }
        assert_eq!(LmetricQ::schedule(&entry(1), &p, &mut ()).await, Some(1));
    }

    #[tokio::test]
    async fn admission_updates_prevent_burst_concentration() {
        let p = pool(4);
        let mut counts = [0; 4];
        for id in 0..100 {
            counts[LmetricQ::schedule(&entry(id), &p, &mut ()).await.unwrap()] += 1;
        }
        assert_eq!(counts, [25; 4]);
    }

    #[tokio::test]
    async fn no_replicas_returns_none() {
        assert_eq!(LmetricQ::schedule(&entry(0), &[], &mut ()).await, None);
    }
}
