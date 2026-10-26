#!/usr/bin/env python3
"""Fetch only pinned public traces and tokenizer files; never model weights."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    config = json.loads((ROOT / "configs/paper_cpu/inputs.json").read_text())
    for item in config["files"]:
        path = ROOT / item["path"]
        if path.exists() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]:
            print("verified", item["path"], flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        for attempt in range(3):
            try:
                request = urllib.request.Request(item["url"], headers={"User-Agent": "paper-cpu-reproduction"})
                with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                assert partial.stat().st_size == item["bytes"]
                assert digest(partial) == item["sha256"]
                partial.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
        print("downloaded and verified", item["path"], flush=True)


if __name__ == "__main__":
    main()
