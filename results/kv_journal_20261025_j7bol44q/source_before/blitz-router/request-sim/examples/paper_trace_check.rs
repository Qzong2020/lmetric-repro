//! Validate the actual anonymous-trace inflater with the real Qwen tokenizer.
//! Usage: paper_trace_check TRACE TOKENIZER_DIR OUTPUT_DIR [LIMIT]
use request_sim::dataset::{BailianDataset, LLMTrace, PromptPayload};
use request_sim::token_sampler::TokenSampler;
use serde_json::{json, Value};
use std::{collections::HashMap, fs::File, io::{BufRead, BufReader, BufWriter, Write}, path::Path};
use tokenizers::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 4, "TRACE TOKENIZER_DIR OUTPUT_DIR [LIMIT]");
    let limit: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out).unwrap();
    let tokenizer = Tokenizer::from_file(Path::new(&args[2]).join("tokenizer.json")).unwrap();
    let sampler = TokenSampler::new(tokenizer.clone(),
        Path::new(&args[2]).join("tokenizer_config.json").to_string_lossy().into(), 2, 16, 16);
    let mut dataset = BailianDataset::new();
    dataset.load(&args[1]);
    let mut tokenized = BufWriter::new(File::create(out.join("tokenized.jsonl")).unwrap());
    let mut known_blocks: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut reverse_blocks: HashMap<Vec<u32>, u64> = HashMap::new();
    let mut length_errors = 0;
    let mut block_errors = 0;
    let mut alias_errors = 0;
    let mut repeated_blocks = 0;
    let mut timestamp_errors = 0;
    let mut first_failure = None;
    let mut checked = 0;
    let mut partial_requests = 0;
    for (i, line) in BufReader::new(File::open(&args[1]).unwrap()).lines().take(limit).enumerate() {
        let original: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let (payload, input_length, output_length) = dataset.inflate(i, &sampler);
        let text = match payload { PromptPayload::Content(s) => s, _ => panic!("expected pre-rendered content") };
        // The trace hashes already describe a rendered prompt. No chat template
        // is applied in this check, matching ChatRenderer::None in the router.
        let encoded = tokenizer.encode(text, false).unwrap();
        let ids = encoded.get_ids();
        let expected_time = (original["timestamp"].as_f64().unwrap() * 1000.0) as u64;
        if dataset.timestamp(i) != expected_time { timestamp_errors += 1; }
        if ids.len() != input_length as usize {
            length_errors += 1;
            first_failure.get_or_insert(json!({"index":i,"kind":"length","expected":input_length,"actual":ids.len()}));
        } else {
            let hashes = original["hash_ids"].as_array().unwrap();
            for (hash, block) in hashes.iter().zip(ids.chunks_exact(16)) {
                let hash = hash.as_u64().unwrap();
                if let Some(previous) = known_blocks.get(&hash) {
                    repeated_blocks += 1;
                    if previous != block {
                        block_errors += 1;
                        first_failure.get_or_insert(json!({"index":i,"kind":"shared-block","hash":hash}));
                    }
                } else { known_blocks.insert(hash, block.to_vec()); }
                if let Some(previous_hash) = reverse_blocks.get(block) {
                    if *previous_hash != hash { alias_errors += 1; }
                } else { reverse_blocks.insert(block.to_vec(), hash); }
            }
        }
        partial_requests += usize::from(input_length % 16 != 0);
        writeln!(tokenized, "{}", json!({"index":i,"timestamp_ms":expected_time,
            "input_length":input_length,"output_length":output_length,
            "hash_ids":original["hash_ids"],"input_tokens":ids})).unwrap();
        checked += 1;
    }
    tokenized.flush().unwrap();
    let summary = json!({"trace":args[1],"total_dataset_requests":dataset.len(),
        "checked_requests":checked,"partial_block_requests":partial_requests,
        "repeated_full_blocks_checked":repeated_blocks,"unique_full_blocks":known_blocks.len(),
        "length_errors":length_errors,"shared_block_errors":block_errors,
        "distinct_hash_alias_errors":alias_errors,"timestamp_errors":timestamp_errors,
        "first_failure":first_failure,
        "scope":"actual request-sim inflation and real tokenizer; no model inference"});
    std::fs::write(out.join("summary.json"), serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    println!("PAPER_TRACE_RESULT {summary}");
    assert_eq!(length_errors + block_errors + alias_errors + timestamp_errors, 0,
        "trace replay is not token-faithful; inspect summary.json");
}
