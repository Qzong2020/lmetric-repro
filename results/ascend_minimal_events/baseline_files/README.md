# BlitzScale Router

Distributed LLM inference router written in Rust. Routes client requests to backend **yaullm** engines (a patched vLLM) over HTTP+SSE, manages KV cache state via RadixTree prefix matching, and dynamically scales replicas.

## Build

Install the system build dependencies first:

```bash
# openEuler
sudo dnf install -y gcc gcc-c++ make cmake perl pkgconf openssl-devel

# Debian / Ubuntu
sudo apt-get install -y build-essential cmake perl pkg-config libssl-dev
```

The pinned Rust toolchain is selected automatically by rustup. Cargo is
configured to prefer dependencies compatible with that toolchain when resolving
a new lockfile. If an existing lockfile selects packages requiring a newer Rust
version, run `cargo update` once to resolve compatible versions.

```bash
# Default build (lmetric scheduling policy)
cargo build -p router --features lmetric-q

# Pick any other policy by swapping the feature
cargo build -p router --features bounded-most-hit-q
cargo build -p router --features aibrix-q
```

See `router/Cargo.toml` for the full list of policy feature flags.

## Run

End-to-end tests (router + yaullm + request-sim orchestration) live in the [MetricsTestRunner](https://github.com/blitz-serving/MetricsTestRunner) repo, not here.

## Docs

Full docs: <https://blitz-serving.github.io/blitzscale-doc/>

## Project Structure

```
blitz-router/
├── router/             # Rust router (~14,000 LOC)
├── radixtree/          # Patricia trie crate (production + verified L0 + lowering bench ladder)
├── policy-dsl/         # Scheduling-policy DSL proc-macro
├── request-sim/        # Request simulator (git submodule)
└── docs/               # Documentation
```

## Related Projects

- **[yaullm](https://github.com/blitz-serving/yaullm)** — Patched vLLM engine with step-level SSE metrics

## Acknowledgements

These projects incorporate code from **Text Generation Inference (TGI)**
<https://github.com/huggingface/text-generation-inference>
Copyright 2022-present Hugging Face Inc.
Licensed under the **Apache License, Version 2.0**.
The full license text is provided in `LICENSES/Apache-2.0.txt`.
Modification details in `NOTICE` and individual source files.
