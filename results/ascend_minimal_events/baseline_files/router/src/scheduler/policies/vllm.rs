//! Scheduling policies derived from the vLLM baseline.
//!
//! Currently one policy: `join-shortest-weight-q`. Figure 6(a) of
//! arXiv:2603.15202v3 uses 4 * queued + running. Here `bs` includes both
//! queued and running requests, so the equivalent expression is
//! 3 * waiting + bs. Without "weight" the name
//! would be ambiguous with other "shortest"-style policies (`least-bs-q`,
//! `least-waiting-q`) that drop one term entirely.
//!
//! Spec-form DSL (`docs/dsl/policies.md` §2):
//!
//! ```text
//! Select min by 3 · sctx.waiting + sctx.bs
//! ```
//!
//! ## vLLM origin
//!
//! Matches the paper's Figure 6(a), https://arxiv.org/html/2603.15202v3#S4.F6.
//! State invariant: bs = waiting + running. Applying weight four to waiting
//! AND adding bs would count waiting once more, yielding 5 * queued + running.

use policy_dsl::policy;

policy! {
    name: JShortestWeightQ,
    gctx: (),
    body: { select_min_by(&root_target(&observations), |o| 3 * o.waiting + o.bs) },
}
