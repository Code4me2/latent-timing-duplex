import argparse
import json

from media_intake.cli import _batch
from media_intake.models import YouTubeSource


def test_batch_isolates_source_failures(tmp_path, capsys):
    config = {
        "schema": "media-intake-sources/v1",
        "policies": {
            "media": {"language": "en-US"},
            "captions": {"language": "en-US"},
        },
        "sources": [
            {"source_id": "youtube-kWPvdAZGd3A",
             "url": "https://www.youtube.com/watch?v=kWPvdAZGd3A", "language": "en-US"},
            {"source_id": "youtube-dQw4w9WgXcQ",
             "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "language": "en-US"},
        ],
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(config))

    class Client:
        def inspect_metadata(self, source: YouTubeSource):
            if source.media_id == "kWPvdAZGd3A":
                raise RuntimeError("isolated failure")
            return {"source_id": source.source_id, "status": "ok"}

    args = argparse.Namespace(sources=path, output_dir=None, stages="metadata",
                              adopt_existing=False)
    assert _batch(Client(), args) == 1
    report = json.loads(capsys.readouterr().out)
    assert [row["status"] for row in report["sources"]] == ["failed", "ok"]
