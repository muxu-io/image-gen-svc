#!/usr/bin/env python3
"""Download default image-gen-svc model checkpoints into /models.

Intended to run inside a one-shot container with the image-gen-models
named volume mounted at /models. Reads the bundled default registry; for
each entry, downloads either:
  - a single file (`url` + optional `sha256`) → streamed to `path`, idempotent
    via existence check, sha256 verified when `--verify` is passed; or
  - a Hugging Face snapshot (`repo_id`) → fetched via `snapshot_download` into
    `path` (a directory). HF's content cache makes this resumable, so we call
    it unconditionally and let the library short-circuit when complete."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from importlib import resources
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = int(r.headers.get("content-length") or 0)
        downloaded = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                pct = 100.0 * downloaded / total
                print(
                    f"  {dest.name}: {pct:5.1f}% ({downloaded/1e9:.2f}/{total/1e9:.2f} GB)",
                    end="\r",
                    flush=True,
                )
    print()
    tmp.replace(dest)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify sha256 if registered")
    args = ap.parse_args()

    raw = yaml.safe_load(
        resources.files("image_gen_svc").joinpath("default_registry.yml").read_text()
    )

    rc = 0
    for model_id, body in (raw.get("models") or {}).items():
        path = Path(body["path"])
        repo_id = body.get("repo_id")
        url = body.get("url")
        sha = body.get("sha256")

        if repo_id:
            print(f"[pull] {model_id}: hf:{repo_id} → {path}")
            path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(path),
                allow_patterns=body.get("allow_patterns"),
            )
            print(f"[done] {model_id}")
            continue

        if not url:
            print(f"[skip] {model_id}: no url or repo_id")
            continue

        if path.exists():
            if args.verify and sha:
                actual = sha256_of(path)
                if actual != sha:
                    print(f"[fail] {model_id}: sha256 mismatch ({actual} vs {sha})")
                    rc = 1
                    continue
            print(f"[ok]   {model_id}: already present at {path}")
            continue

        print(f"[pull] {model_id}: {url} → {path}")
        stream_download(url, path)
        print(f"[done] {model_id}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
