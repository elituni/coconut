"""Download pinned ProsQA splits from the official Coconut repository."""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Pin the source so a later upstream edit does not silently change an experiment.
REVISION = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
BASE_URL = f"https://raw.githubusercontent.com/facebookresearch/coconut/{REVISION}"
FILES = ("prosqa_train.json", "prosqa_valid.json")


def download(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in (*FILES, "download_manifest.json", "UPSTREAM_LICENSE")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Target data files already exist; use a new output directory.")
    # Validate everything before writing; do not leave truncated JSON after a failed request.
    payloads = {}
    manifest = {}
    for name in FILES:
        url = f"{BASE_URL}/data/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        examples = json.loads(content)
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"Invalid or empty dataset from {url}")
        for item in examples:
            if not isinstance(item, dict) or not all(
                key in item for key in ("question", "steps", "answer")
            ):
                raise ValueError(f"Invalid example schema from {url}")
        payloads[name] = content
        manifest[name] = {
            "url": url,
            "sha256": hashlib.sha256(content).hexdigest(),
            "examples": len(examples),
        }
    with urllib.request.urlopen(f"{BASE_URL}/LICENSE", timeout=60) as response:
        payloads["UPSTREAM_LICENSE"] = response.read()
    for name, content in payloads.items():
        (output / name).write_bytes(content)
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        download(args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
