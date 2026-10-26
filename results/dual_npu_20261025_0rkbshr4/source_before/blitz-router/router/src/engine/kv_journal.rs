//! Scheduler metadata journal. Backend hashes are identities, never local keys.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::scheduler::kvcache::{BlockHash, BlockHashState, PrefixBlockHash};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvJournal {
    pub version: String,
    pub observation_kind: String,
    pub num_blocks: usize,
    pub mutation_seq: u64,
    pub mutations: Vec<KvMutation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvMutation {
    pub mutation_seq: u64,
    #[serde(rename="type")]
    pub kind: String,
    pub group_id: usize,
    pub block_size: usize,
    pub block_id: Option<u64>,
    pub cache_hash: Option<String>,
    pub logical_index: Option<usize>,
    #[serde(default)]
    pub prefix_token_ids: Vec<u32>,
    pub request_id: Option<u64>,
    #[serde(default)]
    pub block_ids: Vec<u64>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug)]
struct Mapping { backend: String, prefix: Vec<u64> }

#[derive(Clone, Debug, Default)]
pub(crate) struct KvJournalState {
    pub epoch: Option<String>,
    pub last: u64,
    pub capacity: usize,
    pub failed: Option<String>,
    maps: HashMap<u64, Mapping>,
    pub requests: HashMap<u64, Vec<u64>>,
}

impl KvJournalState {
    pub fn active(&self) -> bool { self.epoch.is_some() }
    pub fn routable(&self) -> bool { self.failed.is_none() }
    pub fn invalidate(&mut self, reason: String) { self.failed.get_or_insert(reason); }
    pub fn mapping_count(&self) -> usize { self.maps.len() }
    pub fn mapping_view(&self) -> std::collections::BTreeMap<u64, String> {
        self.maps.iter().map(|(bid,m)|(*bid,m.backend.clone())).collect()
    }

    /// Validate and stage a whole transaction before touching the routing view.
    pub fn prepare(&self, journal: &KvJournal, epoch: &str, block_size: usize)
        -> Result<(Self, PrefixBlockHash), String> {
        if let Some(reason)=&self.failed { return Err(reason.clone()); }
        if journal.version!="kv-journal-v1" || journal.observation_kind!="scheduler_cache_metadata"
            || !(2..=1_000_000).contains(&journal.num_blocks) || journal.mutations.len()>8192 {
            return Err("unsupported journal capabilities or bounds".into());
        }
        if self.epoch.as_deref().map_or(false,|e|e!=epoch) || epoch.is_empty()
            || (self.capacity!=0 && self.capacity!=journal.num_blocks) {
            return Err("KV epoch or physical capacity changed".into());
        }
        let mut next=self.clone();
        if next.epoch.is_none() {
            let first=journal.mutations.first().ok_or("missing initial empty baseline")?;
            if first.kind!="cache_reset" || first.reason.as_deref()!=Some("initial_empty") || first.mutation_seq!=1 {
                return Err("KV stream must start with an initial empty baseline".into());
            }
        }
        next.epoch=Some(epoch.into());next.capacity=journal.num_blocks;
        for m in &journal.mutations {
            if m.mutation_seq!=next.last+1 || m.group_id!=0 || m.block_size!=block_size || block_size==0 {
                return Err("KV mutation duplicated/gapped, group or block size mismatch".into());
            }
            let valid_bid=|bid:u64|bid>0 && bid<(journal.num_blocks as u64);
            match m.kind.as_str() {
                "cache_reset" => {
                    if !next.requests.is_empty() { return Err("reset while requests hold blocks".into()); }
                    if !matches!(m.reason.as_deref(),Some("initial_empty"|"explicit_reset"))
                        || (m.reason.as_deref()==Some("initial_empty") && m.mutation_seq!=1) {
                        return Err("invalid reset reason".into());
                    }
                    next.maps.clear();
                }
                "cache_insert" => {
                    let bid=m.block_id.filter(|b|valid_bid(*b)).ok_or("invalid insert ID")?;
                    let hash=m.cache_hash.as_ref().filter(|s| !s.is_empty() && s.len()<=128 && s.len()%2==0 && s.bytes().all(|b|b.is_ascii_hexdigit())).ok_or("invalid backend hash")?;
                    let logical=m.logical_index.ok_or("missing logical position")?;
                    if logical>=8192/block_size || m.prefix_token_ids.len()!=(logical+1)*block_size
                        || m.prefix_token_ids.len()>8192 || next.maps.contains_key(&bid) {
                        return Err("invalid prefix proof or physical ID reuse without removal".into());
                    }
                    let hashes=BlockHashState::new(&m.prefix_token_ids,block_size).get_hashes().to_vec();
                    next.maps.insert(bid,Mapping {backend:hash.clone(),prefix:hashes});
                }
                "cache_remove" => {
                    let bid=m.block_id.filter(|b|valid_bid(*b)).ok_or("invalid remove ID")?;
                    let old=next.maps.get(&bid).ok_or("unknown physical mapping removed")?;
                    if m.cache_hash.as_deref()!=Some(old.backend.as_str()) { return Err("old physical mapping identity mismatch".into()); }
                    next.maps.remove(&bid);
                }
                "request_blocks" => {
                    let rid=m.request_id.ok_or("missing occupancy request ID")?;
                    if m.block_ids.len()>8192/block_size+1 || m.block_ids.iter().any(|b|!valid_bid(*b)) {
                        return Err("invalid full occupancy list".into());
                    }
                    let unique:std::collections::HashSet<_>=m.block_ids.iter().collect();
                    if unique.len()!=m.block_ids.len() || next.requests.len()>=1024 && !next.requests.contains_key(&rid) {
                        return Err("duplicate occupancy or active request bound".into());
                    }
                    next.requests.insert(rid,m.block_ids.clone());
                }
                "request_release" => {
                    let rid=m.request_id.ok_or("missing release request ID")?;
                    if !m.block_ids.is_empty() || next.requests.remove(&rid).is_none() {
                        return Err("unknown request release".into());
                    }
                }
                _ => return Err("unknown KV mutation type".into()),
            }
            next.last=m.mutation_seq;
        }
        if next.last!=journal.mutation_seq {return Err("KV watermark mismatch".into());}
        // Rebuild from the bounded authoritative physical registry. Orphaned
        // descendants remain in that registry, but cannot yield a false hit.
        let mut copies:HashMap<u64,Vec<u64>>=HashMap::new();
        for (&bid,m) in &next.maps { copies.entry(*m.prefix.last().unwrap()).or_default().push(bid); }
        let mut tree=PrefixBlockHash::new(next.capacity);
        for (&bid,m) in &next.maps {
            let path:Option<Vec<u64>>=m.prefix.iter().enumerate().map(|(i,h)| {
                if i+1==m.prefix.len() {Some(bid)} else {copies.get(h).map(|ids|ids[0])}
            }).collect();
            if let Some(path)=path {tree.insert(&m.prefix,path);}
        }
        Ok((next,tree))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json,Value};
    fn m(seq:u64,kind:&str)->Value {json!({"mutation_seq":seq,"type":kind,"group_id":0,"block_size":128})}
    fn reset(seq:u64,reason:&str)->Value {let mut v=m(seq,"cache_reset");v["reason"]=json!(reason);v}
    fn insert(seq:u64,bid:u64,n:usize,seed:u32)->Value {
        let mut v=m(seq,"cache_insert");v["block_id"]=json!(bid);v["cache_hash"]=json!(format!("{:064x}",seed as u64+n as u64));
        v["logical_index"]=json!(n-1);v["prefix_token_ids"]=json!(vec![seed;n*128]);v
    }
    fn remove(seq:u64,bid:u64,hash:&str)->Value {let mut v=m(seq,"cache_remove");v["block_id"]=json!(bid);v["cache_hash"]=json!(hash);v}
    fn j(ms:Vec<Value>,last:u64)->KvJournal {serde_json::from_value(json!({"version":"kv-journal-v1","observation_kind":"scheduler_cache_metadata","num_blocks":16,"mutation_seq":last,"mutations":ms})).unwrap()}
    #[test]
    fn kv_journal_duplicate_gap_epoch_and_atomic_failure() {
        let (state,_)=KvJournalState::default().prepare(&j(vec![reset(1,"initial_empty"),insert(2,1,1,7)],2),"e",128).unwrap();
        for bad in [j(vec![insert(2,2,1,8)],2),j(vec![insert(4,2,1,8)],4),j(vec![insert(3,1,1,8)],3)] {
            assert!(state.prepare(&bad,"e",128).is_err());assert_eq!(state.last,2);assert_eq!(state.mapping_count(),1);
        }
        assert!(state.prepare(&j(vec![],2),"other",128).is_err());
    }
    #[test]
    fn kv_journal_parent_only_eviction_orphan_restore_alias_reuse_and_reset() {
        let mut state=KvJournalState::default();
        let (s,tree)=state.prepare(&j(vec![reset(1,"initial_empty"),insert(2,1,1,7),insert(3,2,2,7),insert(4,3,1,7)],4),"e",128).unwrap();state=s;
        let hashes=BlockHashState::new(&vec![7;256],128).get_hashes().to_vec();assert_eq!(tree.get(&hashes),2);
        let h=format!("{:064x}",8);
        let (s,tree)=state.prepare(&j(vec![remove(5,1,&h)],5),"e",128).unwrap();state=s;assert_eq!(tree.get(&hashes),2);
        let (s,tree)=state.prepare(&j(vec![remove(6,3,&h)],6),"e",128).unwrap();state=s;assert_eq!(tree.get(&hashes),0);assert_eq!(state.mapping_count(),1);
        let (s,tree)=state.prepare(&j(vec![insert(7,1,1,7)],7),"e",128).unwrap();state=s;assert_eq!(tree.get(&hashes),2);
        let (s,tree)=state.prepare(&j(vec![remove(8,1,&h),insert(9,1,1,9)],9),"e",128).unwrap();state=s;assert_eq!(tree.get(&hashes),0);
        let (s,tree)=state.prepare(&j(vec![reset(10,"explicit_reset")],10),"e",128).unwrap();assert_eq!(s.mapping_count(),0);assert_eq!(tree.get(&hashes),0);
    }
    #[test]
    fn kv_journal_request_full_occupancy_release_is_not_eviction() {
        let mut occupancy=m(3,"request_blocks");occupancy["request_id"]=json!(101);occupancy["block_ids"]=json!([1,2]);
        let mut release=m(4,"request_release");release["request_id"]=json!(101);release["block_ids"]=json!([]);
        let (state,_)=KvJournalState::default().prepare(&j(vec![reset(1,"initial_empty"),insert(2,1,1,7),occupancy,release],4),"e",128).unwrap();
        assert!(state.requests.is_empty());assert_eq!(state.mapping_count(),1);
    }
}

#[cfg(test)]
mod protocol_tests {
    use serde_json::json;
    #[test]
    fn kv_journal_sse_sequence_rejects_duplicate_gap_downgrade_and_epoch() {
        use crate::engine::vllm_http::MetricSequence;
        let packet=|step:u64,seq:u64|json!({"lmetric_protocol":"request-kv-v1","engine_epoch":"e","step_id":step,"latency":0,"outputs":[],"prefill_tokens":0,"latency_kind":"schedule_to_completion_host_ms","execution_mode":"sync","event_kind":"control","kv_events":true,"block_size":128,"kv_groups":1,"kv_journal":{"version":"kv-journal-v1","observation_kind":"scheduler_cache_metadata","num_blocks":16,"mutation_seq":seq,"mutations":[{"mutation_seq":seq,"type":"cache_reset","group_id":0,"block_size":128,"reason":"initial_empty"}]}});
        let first=packet(1,1);
        let mut epoch=packet(2,2);epoch["engine_epoch"]=json!("other");
        let mut downgrade=packet(2,2);downgrade["lmetric_protocol"]=json!("request-v1");
        let mut group=packet(2,2);group["kv_journal"]["mutations"][0]["group_id"]=json!(1);
        for bad in [packet(1,1),packet(2,1),packet(2,3),epoch,downgrade,group] {
            let mut decoder=MetricSequence::default();assert!(decoder.parse(&first.to_string()).is_ok());
            assert!(decoder.parse(&bad.to_string()).is_err());assert!(decoder.failed());
            assert!(decoder.parse(&packet(2,2).to_string()).is_err());
        }
    }
}
