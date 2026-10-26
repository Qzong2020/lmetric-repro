//! CPU conformance checks against arXiv:2603.15202v3, Figures 6 and 17.
//! The reference expressions use Q and R independently of DSL helpers.

use super::{lmetric::LmetricQ, policy_trait::Policy, vllm::JShortestWeightQ, Entry};
use crate::gateway::validation::ValidGenerateRequest;
use crate::scheduler::kvcache::{BlockHash, BlockHashState, PrefixBlockHash};
use crate::scheduler::state::LMetric;
use crate::types::GenerationParams;
use crate::ScheduleContext;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use std::{sync::Arc, time::Duration};
use tokio::sync::{mpsc, Mutex};

#[derive(Clone, Debug, Serialize)]
struct State {
    queued: usize,
    running: usize,
    prefill: usize,
    hits: usize,
}

fn entry(id: u64, length: usize) -> Entry {
    let tokens: Vec<u32> = (0..length).map(|i| i as u32 + id as u32 * 8192).collect();
    let (response_tx, _) = mpsc::unbounded_channel();
    Entry {
        block_hash_state: BlockHashState::new(&tokens, 16),
        request: ValidGenerateRequest {
            request_id: id, messages: vec![], inputs: String::new(),
            input_length: length as u32, truncate: 0, decoder_input_details: false,
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

fn pool(req: &Entry, states: &[State]) -> Vec<Arc<Mutex<ScheduleContext>>> {
    states.iter().map(|s| {
        let mut metric = LMetric::default();
        metric.bs = s.queued + s.running;
        metric.waiting_reqs = s.queued;
        metric.prefill_tokens = s.prefill as isize;
        metric.all_tokens = metric.bs * 128;
        let mut cache = PrefixBlockHash::new(req.request.input_tokens.len() / 16 + 2);
        if s.hits > 0 {
            cache.insert(&req.block_hash_state.get_hashes()[..s.hits],
                (1..=s.hits as u64).collect());
        }
        Arc::new(Mutex::new(ScheduleContext { kv_journal: Default::default(), lmetric: metric, block_hash: cache }))
    }).collect()
}

async fn verify_bookkeeping(req: &Entry, states: &[State], pool: &[Arc<Mutex<ScheduleContext>>], chosen: usize) {
    for (i, (s, ctx)) in states.iter().zip(pool).enumerate() {
        let actual = ctx.lock().await;
        let added = usize::from(i == chosen);
        assert_eq!(actual.lmetric.bs, s.queued + s.running + added);
        assert_eq!(actual.lmetric.waiting_reqs, s.queued + added);
        assert_eq!(actual.lmetric.prefill_tokens,
            (s.prefill + added * (req.request.input_tokens.len() - s.hits * 16)) as isize);
        assert_eq!(actual.lmetric.all_tokens,
            (s.queued + s.running) * 128 + added * req.request.input_tokens.len());
    }
    assert_eq!(req.block_hash_state.pred_hit_tokens(), states[chosen].hits * 16);
    // The first engine feedback must be usable and report zero correction when
    // it agrees with the snapshot used by the decision.
}

#[tokio::test]
async fn figure6_waiting_and_running_are_distinct() {
    let req = entry(0, 64);
    let states = vec![
        State { queued: 2, running: 0, prefill: 128, hits: 0 },
        State { queued: 0, running: 9, prefill: 0, hits: 0 },
    ];
    let chosen = JShortestWeightQ::schedule(&req, &pool(&req, &states), &mut ()).await;
    assert_eq!(chosen, Some(0), "Fig.6: scores are 4*2+0=8 and 4*0+9=9");
}

#[tokio::test]
async fn independent_paper_oracles_match_real_policies() {
    let mut rng = StdRng::seed_from_u64(260315202);
    let mut lm_mismatch = 0;
    let mut vllm_mismatch = 0;
    let mut ties = 0;
    let mut first_mismatch = None;
    let cases = 5000;
    for case in 0..cases {
        let length = if case < 8 { [1, 15, 16, 17, 31, 32, 4095, 4096][case] }
                     else { rng.gen_range(1..=4096) };
        let n = if case < 5 { [1, 2, 4, 8, 16][case] } else { rng.gen_range(1..=16) };
        let states: Vec<State> = (0..n).map(|_| {
            let queued = rng.gen_range(0..32);
            State { queued, running: rng.gen_range(0..64),
                prefill: if queued == 0 { 0 } else { rng.gen_range(0..20000) },
                hits: rng.gen_range(0..=length / 16) }
        }).collect();
        // Fig.17: hypothetically add the candidate to P-token and BS before
        // comparing instances. No production helper or Observation is used.
        let lm_scores: Vec<u128> = states.iter().map(|s|
            (s.prefill as u128 + length as u128 - s.hits as u128 * 16)
                * (s.queued as u128 + s.running as u128 + 1)).collect();
        // Fig.6(a): Q-BS and R-BS carry weights four and one respectively.
        let v_scores: Vec<u128> = states.iter().map(|s|
            4 * s.queued as u128 + s.running as u128).collect();
        for is_lmetric in [true, false] {
            let mut req = entry(case as u64, length);
            let contexts = pool(&req, &states);
            let chosen = if is_lmetric {
                LmetricQ::schedule(&req, &contexts, &mut ()).await.unwrap()
            } else {
                JShortestWeightQ::schedule(&req, &contexts, &mut ()).await.unwrap()
            };
            verify_bookkeeping(&req, &states, &contexts, chosen).await;
            assert_eq!(req.block_hash_state.set_real_token_hits_get_diff((states[chosen].hits * 16) as u64), 0);
            let scores = if is_lmetric { &lm_scores } else { &v_scores };
            let minimum = *scores.iter().min().unwrap();
            if scores.iter().filter(|&&v| v == minimum).count() > 1 { ties += 1; }
            if scores[chosen] != minimum {
                if is_lmetric { lm_mismatch += 1; } else { vllm_mismatch += 1; }
                first_mismatch.get_or_insert(serde_json::json!({
                    "case": case, "lmetric": is_lmetric, "chosen": chosen,
                    "scores": scores, "states": states,
                }));
            }
        }
    }
    let summary = serde_json::json!({
        "seed": 260315202, "states": cases, "decisions": cases * 2,
        "lmetric_mismatches": lm_mismatch, "vllm_mismatches": vllm_mismatch,
        "tie_cases": ties, "first_mismatch": first_mismatch,
        "bookkeeping_and_matching_hit_feedback_checked": true,
        "scope": "CPU policy conformance, not model latency reproduction",
    });
    if let Ok(dir) = std::env::var("PAPER_CPU_RESULTS") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join("formula_differential.json"),
            serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    }
    println!("PAPER_CPU_FORMULA_RESULT {summary}");
    assert_eq!(lm_mismatch, 0, "Figure 17 mismatch");
    assert_eq!(vllm_mismatch, 0, "Figure 6 mismatch");
}

#[tokio::test]
#[ignore = "requires verified public traces and real-tokenizer output from paper_trace_check"]
async fn real_trace_tokens_match_paper_oracles() {
    use std::io::{BufRead, BufReader};
    let root = std::path::PathBuf::from(std::env::var("PAPER_CPU_RESULTS").expect("PAPER_CPU_RESULTS"));
    let mut rng = StdRng::seed_from_u64(260315202);
    let mut decisions = 0;
    let mut requests = 0;
    let mut ties = 0;
    let mut prefix_hit_blocks = 0;
    for letter in ["A", "B"] {
        let path = root.join(format!("replay-{letter}/tokenized.jsonl"));
        let capacity: usize = BufReader::new(std::fs::File::open(&path).unwrap()).lines()
            .map(|line| {
                let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
                row["hash_ids"].as_array().unwrap().len()
            }).sum::<usize>() + 1;
        let mut cache = PrefixBlockHash::new(capacity);
        let mut anonymous_prefixes = std::collections::HashMap::new();
        let mut next_prefix = 1u64;
        let file = std::fs::File::open(&path).unwrap();
        for line in BufReader::new(file).lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            let tokens: Vec<u32> = serde_json::from_value(row["input_tokens"].clone()).unwrap();
            let length = tokens.len();
            assert_eq!(length, row["input_length"].as_u64().unwrap() as usize);
            let external_hashes = row["hash_ids"].as_array().unwrap();
            let mut parent = 0u64;
            let mut expected_hits = 0;
            for hash in external_hashes.iter().take(length / 16) {
                if let Some(&node) = anonymous_prefixes.get(&(parent, hash.as_u64().unwrap())) {
                    expected_hits += 1;
                    parent = node;
                } else { break; }
            }
            let block_state = BlockHashState::new(&tokens, 16);
            assert_eq!(cache.get(block_state.get_hashes()), expected_hits,
                "anonymous prefix vs actual token hash mismatch: trace={letter}, request={requests}");
            prefix_hit_blocks += expected_hits;
            parent = 0;
            let bids: Vec<u64> = external_hashes.iter().take(length / 16).map(|hash| {
                let key = (parent, hash.as_u64().unwrap());
                parent = *anonymous_prefixes.entry(key).or_insert_with(|| {
                    let node = next_prefix;
                    next_prefix += 1;
                    node
                });
                parent
            }).collect();
            if !bids.is_empty() { cache.insert(block_state.get_hashes(), bids); }
            // Sixteen controlled replica snapshots. These are not claimed to
            // be runtime state measured on H20 or NPU engines.
            let states: Vec<State> = (0..16).map(|_| {
                let queued = rng.gen_range(0..32);
                State { queued, running: rng.gen_range(0..64),
                    prefill: if queued == 0 { 0 } else { rng.gen_range(0..20000) },
                    hits: rng.gen_range(0..=length / 16) }
            }).collect();
            for is_lmetric in [true, false] {
                let mut req = entry(requests, 1);
                req.block_hash_state = BlockHashState::new(&tokens, 16);
                req.request.input_length = length as u32;
                req.request.input_tokens = tokens.clone();
                let contexts = pool(&req, &states);
                let scores: Vec<u128> = states.iter().map(|s| if is_lmetric {
                    (s.prefill as u128 + length as u128 - 16 * s.hits as u128)
                        * (s.queued as u128 + s.running as u128 + 1)
                } else { 4 * s.queued as u128 + s.running as u128 }).collect();
                let minimum = *scores.iter().min().unwrap();
                let chosen = if is_lmetric {
                    LmetricQ::schedule(&req, &contexts, &mut ()).await.unwrap()
                } else { JShortestWeightQ::schedule(&req, &contexts, &mut ()).await.unwrap() };
                assert_eq!(scores[chosen], minimum, "trace={letter} index={requests} lmetric={is_lmetric}");
                verify_bookkeeping(&req, &states, &contexts, chosen).await;
                assert_eq!(req.block_hash_state.set_real_token_hits_get_diff((states[chosen].hits * 16) as u64), 0);
                ties += usize::from(scores.iter().filter(|&&s| s == minimum).count() > 1);
                decisions += 1;
            }
            requests += 1;
        }
    }
    let summary = serde_json::json!({"requests": requests, "decisions": decisions,
        "replicas":16,"mismatches":0,"tie_cases":ties,
        "cross_request_prefix_queries":requests,"cross_request_prefix_mismatches":0,
        "cross_request_prefix_hit_blocks":prefix_hit_blocks,
        "scope":"real trace tokens, controlled replica snapshots, real Rust policies; no inference"});
    std::fs::write(root.join("real_trace_differential.json"), serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    println!("PAPER_REAL_TRACE_RESULT {summary}");
}
