use crate::{scheduler::kvcache::BackendBlockHash, gateway::validation::ValidGenerateRequest};
use axum::body::Bytes;
use eventsource_client as es;
use futures::{stream::FusedStream, StreamExt};
use nohash_hasher::IntMap;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{future::Future, pin::Pin};
use thiserror::Error;
use tokio::{sync::mpsc, task::JoinHandle};

#[derive(Debug, Error)]
pub(crate) enum VllmClientError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("JSON serialization/deserialization failed: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("API error: {0}")]
    ApiError(String),
}

/// Don't organize data in a struct
/// which cause false sharing
pub struct VllmClient {
    client: Client,
    base_url: String,
    model_name: String,
    // Notify frontend to return error code to client
    error_req_ids_tx: mpsc::UnboundedSender<u64>,
    error_req_idx_rx: Option<mpsc::UnboundedReceiver<u64>>,
}

impl Clone for VllmClient {
    fn clone(&self) -> Self {
        assert!(self.error_req_idx_rx.is_none());
        VllmClient {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            model_name: self.model_name.clone(),
            error_req_ids_tx: self.error_req_ids_tx.clone(),
            error_req_idx_rx: None,
        }
    }
}

impl VllmClient {
    pub fn new(base_url: &str, model_name: &str) -> Self {
        // NOTE: we shouldn't set timeout,
        //       since requests in non-stream mode are all long connections
        let client = Client::builder()
            .pool_max_idle_per_host(16)
            .build()
            .expect("Failed to create HTTP client");
        let (tx, rx) = mpsc::unbounded_channel();

        Self {
            client,
            base_url: base_url.to_string(),
            model_name: model_name.to_string(),
            error_req_ids_tx: tx,
            error_req_idx_rx: Some(rx),
        }
    }

    /// event loop assigns task to this vllm instance
    /// Sends pre-tokenized token IDs to /v1/completions — tokenization happens
    /// exactly once in the router, the engine receives token IDs directly.
    pub(crate) async fn add_request(
        &self,
        id: u64,
        request: &ValidGenerateRequest,
    ) -> JoinHandle<Result<Response, VllmClientError>> {
        let request2vllm = Request2Vllm {
            prompt: request.input_tokens.clone(),
            model: self.model_name.clone(),
            stream: false,
            max_tokens: Some(request.params.max_new_tokens),
            min_tokens: Some(request.params.max_new_tokens),
            temperature: None,
        };
        self.send_request("/v1/completions", id, json!(request2vllm)).await
    }

    /// XXX: just a work around!
    pub fn get_error_rx(&mut self) -> mpsc::UnboundedReceiver<u64> {
        self.error_req_idx_rx.take().expect(
            format!(
                "Call get_error_rx multiple times on client binded to {}",
                self.base_url.as_str()
            )
            .as_str(),
        )
    }

    async fn send_request(
        &self,
        endpoint: &str,
        id: u64,
        body: serde_json::Value,
    ) -> JoinHandle<Result<Response, VllmClientError>> {
        let url = format!("{}/{}", self.base_url, endpoint.trim_start_matches('/'));

        let request = self.client.post(&url).header("X-Request-Id", id).json(&body);
        let error_tx = self.error_req_ids_tx.clone();

        tokio::spawn(async move {
            tracing::debug!("Request_{id} about to POST to completions ...");
            match request.send().await {
                Ok(response) => {
                    if !response.status().is_success() {
                        let error_msg =
                            response.text().await.unwrap_or_else(|_| "Unknown error".to_string());

                        let _ = error_tx.send(id).map_err(|e| {
                            tracing::error!(
                                "Request_{id} response fail to send error, due to error_tx {e}"
                            )
                        });
                        tracing::error!("Request_{id} POST error: {}", error_msg);
                        return Err(VllmClientError::ApiError(error_msg));
                    }
                    Ok(response)
                }
                Err(e) => {
                    let _ = error_tx.send(id).map_err(|e| {
                        tracing::error!(
                            "Request_{id} response fail to send error, due to error_tx {e}"
                        )
                    });
                    tracing::error!("Request_{id} POST error: {}", e);
                    Err(VllmClientError::RequestError(e))
                }
            }
        })
    }

    pub async fn init_sse_client(&self) -> Result<Box<dyn es::Client>, es::Error> {
        let dummy_body = json!({
            "prompt": "Once upon a time",
            "max_tokens": 50,
            "stream": true
        })
        .to_string();

        let url = format!("{}/v1/metrics", self.base_url);

        let client = es::ClientBuilder::for_url(url.as_str())?.body(dummy_body).build();

        Ok(Box::new(client))
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct VllmMetric {
    #[serde(default)]
    pub lmetric_protocol: Option<String>,
    #[serde(default)]
    pub engine_epoch: Option<String>,
    pub kv_journal: Option<super::kv_journal::KvJournal>,
    #[serde(default)]
    pub kv_events: Option<bool>,
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub kv_groups: Option<usize>,
    #[serde(default)]
    pub prefill_tokens: usize,
    #[serde(default)]
    pub prefill_token_budget: usize,
    #[serde(default)]
    pub latency: u64,
    #[serde(default)]
    pub outputs: Vec<VllmRequestStatus>,
    #[serde(default)]
    pub new_block_hashes: Vec<BackendBlockHash>,
    #[serde(default)]
    pub evicted_block_hashes: Vec<BackendBlockHash>,
    #[serde(default)]
    pub evicted_block_ids: Vec<u64>,
    #[serde(default)]
    pub cur_used_block_ids: IntMap<u64, Vec<u64>>,
    #[serde(default)]
    pub new_block_hashes_ids: IntMap<u64, Vec<u64>>,
    pub op_exec_log: Option<String>,
    #[serde(default)]
    pub preempted_ids: Vec<u64>,
    #[serde(default)]
    pub aborted_requests: Vec<u64>,
    #[serde(default)]
    pub step_id: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct VllmRequestStatus {
    pub request_id: u64,
    #[serde(default)]
    pub new_token_ids: Vec<u32>,
    #[serde(default = "default_state")]
    pub state: String,
    #[serde(default)]
    pub is_finished: bool,
    #[serde(default)]
    pub hit_token_cnt: u64,
    /// Per-request KV cache size (in tokens) at the START of this step —
    /// i.e. tokens already prefill-computed before the engine ran this step.
    /// Source: `output.num_computed_tokens` in
    /// `vllm/v1/engine/__init__.py::simplified_engine_core_outputs_to_dict`.
    /// Used by the latency simulator to reconstruct `BatchForPredictor`.
    #[serde(default)]
    pub prev_computed_tokens: u32,
}

fn default_state() -> String {
    "DECODE".to_string()
}

#[allow(unused)]
#[deprecated]
fn parse_response(resp: Result<Bytes, reqwest::Error>) {
    match resp {
        Ok(bytes) => {
            if let Ok(jstr) = std::str::from_utf8(&bytes) {
                match parse_json_inner(jstr) {
                    Ok(generations) => {
                        println!("Generations: {:?}", generations);
                    }
                    Err(e) => {
                        println!("Json error {} with raw string: {}", e, jstr);
                        // tracing::error!("Json error {} with raw string: {}", e, jstr);
                    }
                }
            } else {
                println!("Invalid response from vllm: Utf8Error");
                // tracing::error!("Invalid response from vllm: Utf8Error");
            }
        }
        Err(e) => {
            println!("Error from vllm: {}!", e);
            // tracing::error!("Error from vllm: {}!", e);
        }
    }
}

#[allow(unused)]
#[deprecated]
async fn parse_json(
    jstr: String,
    id: u64,
    mut stream: Pin<Box<dyn FusedStream<Item = Result<Bytes, reqwest::Error>>>>,
) -> Pin<
    Box<
        dyn Future<
            Output = Option<(
                (u64, Pin<Box<dyn FusedStream<Item = Result<Bytes, reqwest::Error>>>>),
                Option<String>,
            )>,
        >,
    >,
> {
    Box::pin(async move {
        match parse_json_inner(&jstr) {
            Ok(generation) => {
                match generation {
                    Some((_chat_id, Some(content))) => Some(((id, stream), Some(content))),
                    Some((_chat_id, None)) => Some(((id, stream), None)),
                    // the close of sse
                    None => None,
                }
            }
            Err(ref e) if e.is_eof() => {
                let bytes_snd: Bytes = stream.next().await.unwrap().expect("Not axum::Bytes");
                let jstr_snd = jstr + std::str::from_utf8(&bytes_snd).unwrap();
                let fut = parse_json(jstr_snd, id, stream).await;
                fut.await
            }
            Err(e) => {
                println!("Json error {} with raw string: {}", e, jstr);
                None
                // tracing::error!("Json error {} with raw string: {}", e, jstr);
            }
        }
    })
}

fn parse_json_inner(jstr: &str) -> Result<Option<(String, Option<String>)>, serde_json::Error> {
    let mut generation = None;
    for line in jstr.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(idx) = line.find('{') {
            let jvalue: serde_json::Value = serde_json::from_str(&line[idx..])?;
            generation = Some((jvalue["id"].to_string(), {
                let v = &jvalue["choices"][0]["finish_reason"];
                match v {
                    serde_json::Value::Null => {
                        Some(jvalue["choices"][0]["delta"]["content"].to_string())
                    }
                    _ => None,
                }
            }));
        } else {
            continue;
        };
    }
    Ok(generation)
}

#[derive(Serialize, Clone)]
struct Request2Vllm {
    prompt: Vec<u32>,
    model: String,
    stream: bool,
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

/// New request-v1 sessions are contiguous and cannot silently reconnect.
/// Legacy yaullm JSON keeps its existing permissive parsing behavior.
#[derive(Default)]
pub(super) struct MetricSequence {
    epoch: Option<String>,
    last: u64,
    mutation_last: u64,
    kv_mode: bool,
    failed: bool,
}

impl MetricSequence {
    pub(super) fn versioned(&self) -> bool { self.epoch.is_some() }

    pub(super) fn failed(&self) -> bool { self.failed }

    pub(super) fn invalidate(&mut self) { self.failed = true; }

    pub(super) fn parse(&mut self, data: &str) -> Result<VllmMetric, String> {
        if self.failed { return Err("LMetric session invalid; restart required".into()); }
        let result = self.parse_inner(data);
        if result.is_err() && (self.versioned() || data.contains("lmetric_protocol")) {
            self.failed = true;
        }
        result
    }

    fn parse_inner(&mut self, data: &str) -> Result<VllmMetric, String> {
        if data.len() > 20 * 1024 * 1024 { return Err("oversized SSE envelope".into()); }
        let raw: serde_json::Value = serde_json::from_str(data).map_err(|e| e.to_string())?;
        let metric: VllmMetric = serde_json::from_value(raw.clone()).map_err(|e| e.to_string())?;
        match metric.lmetric_protocol.as_deref() {
            None if self.epoch.is_none() => return Ok(metric),
            Some("request-v1" | "request-kv-v1") => {},
            _ => return Err("Unsupported/missing LMetric protocol".into()),
        }
        let kv_mode = metric.lmetric_protocol.as_deref() == Some("request-kv-v1");
        if self.versioned() && kv_mode != self.kv_mode { return Err("KV capability changed".into()); }
        if kv_mode && !cfg!(all(feature="lmetric-q",feature="radixtree-blockhash",not(feature="simulator"))) {
            return Err("KV journal requires lmetric-q/radixtree without simulator".into());
        }
        let mut mutation_last = self.mutation_last;
        if kv_mode {
            let j = metric.kv_journal.as_ref().ok_or("missing KV journal")?;
            if j.version != "kv-journal-v1" || j.observation_kind != "scheduler_cache_metadata"
                || !(2..=1_000_000).contains(&j.num_blocks) || j.mutations.len() > 8192 {
                return Err("invalid KV journal capabilities/bounds".into());
            }
            for m in &j.mutations {
                if m.mutation_seq != mutation_last + 1 || m.group_id != 0
                    || Some(m.block_size) != metric.block_size {
                    return Err("KV mutation sequence or group/block mismatch".into());
                }
                mutation_last = m.mutation_seq;
            }
            if mutation_last != j.mutation_seq { return Err("KV watermark mismatch".into()); }
            if !matches!(raw["event_kind"].as_str(),Some("completion"|"control")) {
                return Err("invalid KV envelope kind".into());
            }
        } else if metric.kv_journal.is_some() { return Err("unexpected KV journal".into()); }
        for field in ["step_id", "latency", "outputs", "prefill_tokens"] {
            if raw.get(field).is_none() { return Err(format!("Missing request-v1 field: {field}")); }
        }
        if metric.kv_groups != Some(1) || metric.block_size.unwrap_or(0) == 0
            || metric.kv_events != Some(kv_mode)
            || raw["latency_kind"] != "schedule_to_completion_host_ms"
            || !matches!(raw["execution_mode"].as_str(), Some("sync" | "async")) {
            return Err("Invalid request-v1 capabilities/timing semantics".into());
        }
        if !metric.evicted_block_ids.is_empty() || !metric.cur_used_block_ids.is_empty()
            || !metric.new_block_hashes_ids.is_empty() || !metric.new_block_hashes.is_empty()
            || !metric.evicted_block_hashes.is_empty() || !metric.preempted_ids.is_empty()
            || !metric.aborted_requests.is_empty() {
            return Err("request-v1 does not support KV/preempt/abort events".into());
        }
        let epoch = metric.engine_epoch.as_deref().filter(|s| !s.is_empty())
            .ok_or("Missing request-v1 engine_epoch")?;
        if self.epoch.as_deref().map_or(false, |e| e != epoch) || metric.step_id != self.last + 1 {
            return Err("LMetric epoch changed or step sequence duplicated/gapped".into());
        }
        let mut seen = std::collections::HashSet::new();
        for (i, status) in metric.outputs.iter().enumerate() {
            for field in ["request_id", "new_token_ids", "state", "is_finished", "hit_token_cnt", "prev_computed_tokens"] {
                if raw["outputs"][i].get(field).is_none() {
                    return Err(format!("Missing request-v1 output field: {field}"));
                }
            }
            if !matches!(status.state.as_str(), "PREFILL" | "DECODE")
                || !seen.insert(status.request_id)
                || (status.new_token_ids.is_empty() && !status.is_finished) {
                return Err("Invalid request-v1 request output".into());
            }
        }
        self.epoch = Some(epoch.to_owned());
        self.last = metric.step_id;
        self.kv_mode = kv_mode;
        self.mutation_last = mutation_last;
        Ok(metric)
    }
}
