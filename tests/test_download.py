import importlib.util
import io
import json
from pathlib import Path

import pytest


def downloader():
    path = Path(__file__).parents[1] / "scripts/download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_download_and_no_overwrite(tmp_path, monkeypatch, examples):
    module = downloader()
    urls = []

    def response(url, **kwargs):
        urls.append(url)
        return io.BytesIO(
            b"MIT License" if url.endswith("LICENSE") else json.dumps(examples).encode()
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", response)
    module.download(tmp_path)
    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert manifest["prosqa_train.json"]["examples"] == 3
    assert all(module.REVISION in url for url in urls)
    assert (tmp_path / "UPSTREAM_LICENSE").read_text() == "MIT License"
    with pytest.raises(FileExistsError):
        module.download(tmp_path)


def test_invalid_download_does_not_write_partial_files(tmp_path, monkeypatch):
    module = downloader()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"[]"))
    with pytest.raises(ValueError, match="empty"):
        module.download(tmp_path)
    assert not list(tmp_path.iterdir())
