from pathlib import Path
import re
ROOT=Path('/root/lmetric-repro/blitz-router')
def edit(path,old,new,count=1):
 p=ROOT/path;s=p.read_text();assert s.count(old)==count,(path,s.count(old),old[:80]);p.write_text(s.replace(old,new))
edit('router/src/engine/mod.rs','pub(crate) mod vllm_http;','pub(crate) mod vllm_http;\npub mod kv_journal;')
edit('router/src/engine/vllm_http.rs','    pub engine_epoch: Option<String>,','    pub engine_epoch: Option<String>,\n    pub kv_journal: Option<super::kv_journal::KvJournal>,')
edit('router/src/engine/client.rs','pub struct EngineStepOutput {','pub struct EngineStepOutput {\n    pub engine_epoch: Option<String>,\n    pub kv_journal: Option<super::kv_journal::KvJournal>,')
edit('router/src/engine/client.rs','            lmetric_protocol: m.lmetric_protocol,','            engine_epoch: m.engine_epoch,\n            kv_journal: m.kv_journal,\n            lmetric_protocol: m.lmetric_protocol,')
# Add absent optional fields to all other literal construction sites.
for path in (ROOT/'router/src').rglob('*.rs'):
 s=path.read_text()
 s2=re.sub(r'(EngineStepOutput\s*\{\s*)(?=lmetric_protocol:)',r'\1engine_epoch: None, kv_journal: None,\n            ',s)
 if s2!=s:path.write_text(s2)
edit('router/src/scheduler/state.rs','    pub block_hash: PrefixBlockHash,','    pub block_hash: PrefixBlockHash,\n    pub kv_journal: crate::engine::kv_journal::KvJournalState,')
p=ROOT/'router/src/scheduler/state.rs';p.write_text(p.read_text()+'''
impl ScheduleContext {
    pub fn invalidate_kv(&mut self, reason: String) {
        self.kv_journal.invalidate(reason);
        self.block_hash = PrefixBlockHash::new(self.kv_journal.capacity.max(2));
    }
}
''')
# Preserve all existing constructor fields and user modifications.
for path in (ROOT/'router/src').rglob('*.rs'):
 s=path.read_text();s2=re.sub(r'(ScheduleContext\s*\{\s*)(?=(?:lmetric|block_hash):)',r'\1kv_journal: Default::default(), ',s)
 if s2!=s:path.write_text(s2)
edit('router/src/scheduler/policies/dsl_runtime.rs','    pub idx: usize,','    pub idx: usize,\n    pub routing_enabled: bool,')
edit('router/src/scheduler/policies/dsl_runtime.rs','        Observation {\n            idx,','        Observation {\n            routing_enabled: sctx.kv_journal.routable(),\n            idx,')
edit('router/src/scheduler/policies/dsl_runtime.rs','    observations.iter().collect()','    observations.iter().filter(|o| o.routing_enabled).collect()')
edit('router/src/scheduler/policies/dsl_runtime.rs','''    cached: &Observation,
) {
    let request = &entry.request;
    let ScheduleContext { lmetric, block_hash } =
        &mut *all_sctx[chosen].lock().await;''','''    cached: &Observation,
) -> bool {
    let request = &entry.request;
    let mut context = all_sctx[chosen].lock().await;
    if !context.kv_journal.routable() { return false; }
    let ScheduleContext { lmetric, block_hash, .. } = &mut *context;''')
edit('router/src/scheduler/policies/dsl_runtime.rs','''        all_tokens_inc: request.input_tokens.len(),
    };
}''','''        all_tokens_inc: request.input_tokens.len(),
    };
    true
}''')
edit('policy-dsl/src/lower.rs','''                        apply_default_after(entry, all_sctx, idx, &observations[idx]).await;''','''                        if !apply_default_after(entry, all_sctx, idx, &observations[idx]).await {
                            return None;
                        }''')
edit('router/src/scheduler/kvcache.rs','        assert!(backend_bids.is_empty() == false);','''        if backend_bids.is_empty() {
            self.block_indices.clear();
            self.sync_nblks = 0;
            return;
        }''')
# Protocol validation keeps legacy/request-v1 behavior and uses a separate KV version.
edit('router/src/engine/vllm_http.rs','    last: u64,\n    failed: bool,','    last: u64,\n    mutation_last: u64,\n    kv_mode: bool,\n    failed: bool,')
edit('router/src/engine/vllm_http.rs','''        let raw: serde_json::Value = serde_json::from_str(data).map_err(|e| e.to_string())?;''','''        if data.len() > 20 * 1024 * 1024 { return Err("oversized SSE envelope".into()); }
        let raw: serde_json::Value = serde_json::from_str(data).map_err(|e| e.to_string())?;''')
edit('router/src/engine/vllm_http.rs','            Some("request-v1") => {},','            Some("request-v1" | "request-kv-v1") => {},')
edit('router/src/engine/vllm_http.rs','''        for field in ["step_id", "latency", "outputs", "prefill_tokens"] {''','''        let kv_mode = metric.lmetric_protocol.as_deref() == Some("request-kv-v1");
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
        for field in ["step_id", "latency", "outputs", "prefill_tokens"] {''')
edit('router/src/engine/vllm_http.rs','            || metric.kv_events != Some(false)','            || metric.kv_events != Some(kv_mode)')
edit('router/src/engine/vllm_http.rs','        self.last = metric.step_id;','        self.last = metric.step_id;\n        self.kv_mode = kv_mode;\n        self.mutation_last = mutation_last;')
# Extend existing request accounting semantics, not the legacy yaullm path.
p=ROOT/'router/src/engine/colocation.rs';s=p.read_text().replace('m.lmetric_protocol.as_deref() == Some("request-v1")','matches!(m.lmetric_protocol.as_deref(), Some("request-v1" | "request-kv-v1"))');p.write_text(s)
edit('router/src/engine/colocation.rs','''                    tracing::error!(engine = replica_index, error = %e, "RECV_STEP_ERROR");''','''                    tracing::error!(engine = replica_index, error = %e, "RECV_STEP_ERROR");
                    let mut context = schedule_context.lock().await;
                    if context.kv_journal.active() || std::env::var("LMETRIC_REQUIRE_KV_JOURNAL").as_deref()==Ok("1") {
                        context.invalidate_kv(e.to_string());
                        break;
                    }
                    drop(context);''')
edit('router/src/engine/colocation.rs','''                tracing::error!(engine = replica_index, "LMetric block-size mismatch; stop consumer and restart session");
                break;''','''                tracing::error!(engine = replica_index, "LMetric block-size mismatch; stop consumer and restart session");
                schedule_context.lock().await.invalidate_kv("block-size mismatch".into());
                break;''')
edit('router/src/engine/colocation.rs','''            tracing::trace!(engine = replica_index, step = ?m, "STEP_RECEIVED");''','''            // Stage journal before applying request accounting. No partial cache
            // view or untrusted physical ID becomes visible to a policy.
            let prepared_kv = if let Some(journal) = &m.kv_journal {
                let mut context = schedule_context.lock().await;
                match context.kv_journal.prepare(journal, m.engine_epoch.as_deref().unwrap_or(""), m.lmetric_block_size.unwrap_or(0)) {
                    Ok(prepared) => Some(prepared),
                    Err(reason) => {
                        tracing::error!(engine=replica_index, %reason, "KV_JOURNAL_INVALID");
                        context.invalidate_kv(reason);
                        break;
                    }
                }
            } else {
                let mut context = schedule_context.lock().await;
                if context.kv_journal.active() { context.invalidate_kv("KV capability disappeared".into()); break; }
                None
            };
            tracing::trace!(engine = replica_index, step = ?m, "STEP_RECEIVED");''')
edit('router/src/engine/colocation.rs','''            sctx.block_hash.remove(m.evicted_block_ids.clone());''','''            if let Some((state, tree)) = prepared_kv {
                for mutation in &m.kv_journal.as_ref().unwrap().mutations {
                    if matches!(mutation.kind.as_str(), "request_blocks" | "request_release") {
                        let rid = mutation.request_id.unwrap();
                        if let Some(entry) = entries.get_mut(&rid).or_else(|| except_context.entries.get_mut(&rid)) {
                            entry.block_hash_state.set_bids(mutation.block_ids.clone());
                        }
                    }
                }
                sctx.kv_journal = state;
                sctx.block_hash = tree;
            }
            sctx.block_hash.remove(m.evicted_block_ids.clone());''')
