from pathlib import Path
ROOT=Path('/root/lmetric-repro/engines/vllm')
def edit(path,old,new,count=1):
 p=ROOT/path;s=p.read_text();
 if new in s:return
 assert s.count(old)==count,(path,s.count(old),old[:70]);p.write_text(s.replace(old,new))
edit('vllm/v1/core/block_pool.py','        self.metrics_collector = metrics_collector\n','        self.metrics_collector = metrics_collector\n        self.lmetric_journal = None\n')
edit('vllm/v1/core/block_pool.py','            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)\n','''            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if self.lmetric_journal is not None:
                index = num_cached_blocks + i
                self.lmetric_journal.insert(
                    blk.block_id, block_hash, kv_cache_group_id, block_size,
                    index, request.all_token_ids[: (index + 1) * block_size],
                )
''')
edit('vllm/v1/core/block_pool.py','        block.reset_hash()\n\n        if self.enable_kv_cache_events:',"""        if self.lmetric_journal is not None:
            self.lmetric_journal.remove(
                block.block_id, get_block_hash(block_hash), get_group_id(block_hash)
            )
        block.reset_hash()

        if self.enable_kv_cache_events:""")
edit('vllm/v1/core/block_pool.py','        logger.info("Successfully reset prefix cache")','''        if self.lmetric_journal is not None:
            self.lmetric_journal.reset()
        logger.info("Successfully reset prefix cache")''')
edit('vllm/v1/core/kv_cache_manager.py','        # P/D: delay caching blocks if we have to recv from','''        # Observe successful full occupancy independently of cache registration.
        journal = self.block_pool.lmetric_journal
        if journal is not None:
            journal.request_blocks(
                request.request_id, self.get_block_ids(request.request_id)
            )

        # P/D: delay caching blocks if we have to recv from''')
edit('vllm/v1/core/kv_cache_manager.py','        self.coordinator.free(request.request_id)\n','''        self.coordinator.free(request.request_id)
        if self.block_pool.lmetric_journal is not None:
            self.block_pool.lmetric_journal.release(request.request_id)
''')
edit('vllm/v1/engine/lmetric_events.py','    p = config.parallel_config\n','''    from vllm.v1.engine.lmetric_kv_journal import kv_enabled

    p = config.parallel_config
    if kv_enabled() and (
        p.tensor_parallel_size != 1 or not config.model_config.enforce_eager
        or not config.cache_config.enable_prefix_caching
    ):
        raise ValueError("LMetric KV journal requires single-card eager prefix cache")
''')
edit('vllm/v1/engine/lmetric_events.py','        self.epoch = uuid.uuid4().hex\n','''        self.epoch = uuid.uuid4().hex
        self.kv_journal = None
        from vllm.v1.engine.lmetric_kv_journal import KVJournal, kv_enabled

        if kv_enabled():
            spec = groups[0].kv_cache_spec
            pool = scheduler.kv_cache_manager.block_pool
            if (type(spec).__name__ != "FullAttentionSpec"
                    or getattr(spec, "sliding_window", None) is not None
                    or getattr(spec, "attention_chunk_size", None) is not None
                    or len(pool.cached_block_hash_to_block) != 0):
                raise ValueError("LMetric KV requires a fresh single FullAttention pool")
            self.kv_journal = KVJournal(self.epoch, block_size, pool.num_gpu_blocks)
            pool.lmetric_journal = self.kv_journal
''')
edit('vllm/v1/engine/lmetric_events.py','        self.seq = 0\n','        self.protocol = "request-kv-v1" if self.kv_journal else PROTOCOL\n        self.seq = 0\n')
edit('vllm/v1/engine/lmetric_events.py','                            kv_events=False, protocol=PROTOCOL)','                            kv_events=self.kv_journal is not None, protocol=self.protocol)')
edit('vllm/v1/engine/lmetric_events.py','            if req.client_index != 0 or req.mm_features or req.prompt_token_ids is None:','''            if self.kv_journal is not None and getattr(req, "cache_salt", None):
                raise RuntimeError("LMetric KV excludes cache_salt")
            if req.client_index != 0 or req.mm_features or req.prompt_token_ids is None:''')
edit('vllm/v1/engine/lmetric_events.py','        eco.lmetric_event = dict(lmetric_protocol=PROTOCOL, engine_epoch=self.epoch,','        eco.lmetric_event = dict(lmetric_protocol=self.protocol, engine_epoch=self.epoch,')
edit('vllm/v1/engine/lmetric_events.py','                                kv_events=False, block_size=self.runtime["scheduler_block_size"],','                                kv_events=self.kv_journal is not None, block_size=self.runtime["scheduler_block_size"],')
edit('vllm/v1/engine/lmetric_events.py','        return core_outputs\n','''        if self.kv_journal is not None:
            eco.lmetric_event["kv_journal"] = self.kv_journal.drain()
            eco.lmetric_event["event_kind"] = "completion"
        return core_outputs

    def control_output(self, make_output):
        """Export idle control mutations without fabricating a forward step."""
        self.seq += 1
        event = dict(
            lmetric_protocol=self.protocol, engine_epoch=self.epoch,
            step_id=self.seq, latency=0,
            latency_kind="schedule_to_completion_host_ms",
            execution_mode="async" if self.runtime["async_scheduling"] else "sync",
            kv_events=True, block_size=self.runtime["scheduler_block_size"],
            kv_groups=1, prefill_tokens=0, outputs=[], event_kind="control",
            kv_journal=self.kv_journal.drain(),
        )
        return make_output(lmetric_event=event)
''')
edit('vllm/v1/engine/lmetric_events.py','        for status in event["outputs"]:\n','''        for mutation in event.get("kv_journal", {}).get("mutations", []):
            if "request_id" in mutation:
                rid = mutation["request_id"]
                if rid not in self.identities:
                    self.failed = "Unknown journal request identity"
                    raise RuntimeError(self.failed)
                mutation["request_id"] = self.identities[rid]
        for status in event["outputs"]:
''')
edit('vllm/v1/engine/core.py','        self.lmetric_tracker = None\n','''        from vllm.v1.engine.lmetric_kv_journal import kv_enabled

        if kv_enabled() and not enabled():
            raise ValueError("KV journal requires VLLM_LMETRIC_EVENTS=1")
        self.lmetric_tracker = None
''')
edit('vllm/v1/engine/core.py','''        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )''','''        tracker = self.lmetric_tracker
        journal = getattr(tracker, "kv_journal", None)
        if journal is not None and (
            self.scheduler.has_unfinished_requests() or self.batch_queue
            or reset_running_requests or reset_connector
        ):
            raise ValueError("LMetric KV reset requires a fully drained engine")
        result = self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )
        if journal is not None and result:
            self.output_queue.put_nowait((0, tracker.control_output(EngineCoreOutputs)))
        return result''')
edit('vllm/v1/engine/async_llm.py','''            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")''','''            except Exception as e:
                if lmetric_hub is not None:
                    lmetric_hub.failed = "Engine output failed; restart LMetric session"
                logger.exception("AsyncLLM output_handler failed.")''')
