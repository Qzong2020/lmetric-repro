from pathlib import Path
R=Path('/root/lmetric-repro/blitz-router/router/src')
p=R/'engine/colocation.rs';s=p.read_text();old='unreachable!("engine#{replica_index} WQ channel closed by receiver at CQ");';assert s.count(old)==1
s=s.replace(old,'tracing::info!(engine = replica_index, "WORK_LOOP_STOPPED");\n                    break;')
s=s.replace('            // Adds newly posted requests\n','            let apply_started = std::time::Instant::now();\n            // Adds newly posted requests\n',1)
s=s.replace('            let prepared_kv = if let Some(journal) = &m.kv_journal {','            let kv_started = std::time::Instant::now();\n            let prepared_kv = if let Some(journal) = &m.kv_journal {',1)
s=s.replace('            tracing::trace!(engine = replica_index, step = ?m, "STEP_RECEIVED");','            metrics::histogram!("blitz_kv_prepare_duration_seconds", kv_started.elapsed().as_secs_f64());\n            tracing::trace!(engine = replica_index, step = ?m, "STEP_RECEIVED");',1)
needle='            // Invariant checks at event loop boundary\n';assert s.count(needle)==1
s=s.replace(needle,'''            metrics::histogram!("blitz_step_apply_duration_seconds", apply_started.elapsed().as_secs_f64());
            if tracing::enabled!(target: "cache_tracking", tracing::Level::DEBUG) {
                let c = schedule_context.lock().await;
                let state = serde_json::json!({"engine":replica_index,"step_id":m.step_id,
                    "engine_epoch":c.kv_journal.epoch,"mutation_seq":c.kv_journal.last,
                    "physical":c.kv_journal.mapping_view(),"requests":c.kv_journal.requests,
                    "valid":c.kv_journal.routable(),"active_entries":entries.len(),
                    "exception_entries":except_context.entries.len(),
                    "bs":c.lmetric.bs,"waiting":c.lmetric.waiting_reqs,
                    "prefill":c.lmetric.prefill_tokens,"all_tokens":c.lmetric.all_tokens});
                tracing::debug!(target: "cache_tracking", state = %state, "KV_STATE");
            }

'''+needle)
p.write_text(s)
p=R/'scheduler/policies/dsl_runtime.rs';s=p.read_text();needle='        Observation {\n            routing_enabled:';assert s.count(needle)==1
s=s.replace(needle,'''        tracing::debug!(target: "scheduling", request_id = entry.request.request_id,
            engine = idx, routing_enabled = sctx.kv_journal.routable(),
            bs = sctx.lmetric.bs, waiting = sctx.lmetric.waiting_reqs,
            prefill = sctx.lmetric.prefill_tokens, all_tokens = sctx.lmetric.all_tokens,
            hit_tokens = hit_blocks * block_size, cache_epoch = epoch,
            lmetric_score = (sctx.lmetric.prefill_tokens.max(0) as usize
                + entry.request.input_tokens.len().saturating_sub(hit_blocks * block_size))
                * (sctx.lmetric.bs + 1),
            shortest_weight_score = 3 * sctx.lmetric.waiting_reqs + sctx.lmetric.bs,
            "CANDIDATE");
        Observation {
            routing_enabled:''')
p.write_text(s)
p=R/'engine/vllm_http.rs';s=p.read_text();old='if kv_mode && !cfg!(all(feature="lmetric-q",feature="radixtree-blockhash",not(feature="simulator"))) {';assert s.count(old)==1
s=s.replace(old,'if kv_mode && !cfg!(all(any(feature="lmetric-q",feature="join-shortest-weight-q"),feature="radixtree-blockhash",not(feature="simulator"))) {').replace('KV journal requires lmetric-q/radixtree without simulator','KV journal requires lmetric-q or join-shortest-weight-q/radixtree without simulator');p.write_text(s)
