# media-intake-local

A small, extractable Python package for policy-driven YouTube metadata inspection,
media acquisition, caption acquisition, and coordinated batch intake. The
**distribution name is intentionally local** (`media-intake-local`); no package
index availability is claimed. The import and command names are `media_intake`
and `media-intake`.

The package is independent of the parent research application. It has no NumPy
or PyYAML dependency and does not import `latent_timing_duplex`. Copying this
directory elsewhere is sufficient to build, test, or install it.

## Install and reproduce the Python environment

Core import has no third-party runtime dependency:

```bash
cd packages/media-intake
uv sync --locked --extra dev
uv run pytest
uv build
# or: python -m pip install .
```

Install the optional upstream Python tool in an isolated environment with:

```bash
uv sync --locked --extra tooling --extra dev
```

`uv.lock` pins **Python packages only**. It does not and cannot pin system
binaries. The wrapper invokes the independently maintained [yt-dlp](https://github.com/yt-dlp/yt-dlp)
and [FFmpeg](https://ffmpeg.org/) projects; neither is bundled or relicensed.
Media acquisition requires `yt-dlp`, `ffmpeg`, and `ffprobe` on `PATH`.
Metadata and caption stages require only `yt-dlp`. Current yt-dlp YouTube support
may also require a supported JavaScript runtime (for example Deno, Node.js,
Bun, or QuickJS) and the yt-dlp EJS challenge components; follow yt-dlp's
current installation documentation. The optional extra installs yt-dlp's
Python distribution, not FFmpeg or a JavaScript runtime.

Receipts record the external versions actually executed. Such observations are
provenance, not reproducible pins. The versions used during development were
`yt-dlp 2026.08.19` and FFmpeg/ffprobe `n9.0.1`; other supported versions may
work, and extractor behavior can change upstream.

## Python API: stages are independent

```python
from pathlib import Path
from media_intake import (
    CaptionPolicy, MediaFormatPolicy, MediaIntake, YouTubeSource,
)

source = YouTubeSource(
    "youtube-dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "en-US",
)
client = MediaIntake()

# Metadata only: no media, ffmpeg, or ffprobe.
summary = client.inspect_metadata(source)

# Media only: no caption request.
media = client.acquire_media(
    source, Path("/private/media"),
    MediaFormatPolicy(maximum_height=1080, language="en-US",
                      video_codec="h264", audio_codec="aac", container="mp4"),
)

# Captions only: no MP4 and no ffprobe required.
captions = client.acquire_captions(
    source, Path("/private/media"), CaptionPolicy(language="en-US")
)

# Coordinated metadata + media + captions with one metadata snapshot.
record = client.acquire(
    source, Path("/private/media"),
    MediaFormatPolicy(language="en-US"), CaptionPolicy(language="en-US"),
)
```

`MediaIntake(yt_dlp=..., ffprobe=...)` accepts small injected adapters. The
built-in adapters accept an injected `Runner`, enabling deterministic offline
tests without monkeypatching subprocess internals.

## CLI

Each stage is explicit and does not force the others:

```bash
media-intake metadata --source-id youtube-dQw4w9WgXcQ \
  --url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' --language en-US

media-intake media --source-id youtube-dQw4w9WgXcQ \
  --url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' --language en-US \
  --output-dir /private/media --maximum-height 1080 \
  --video-codec h264 --audio-codec aac --container mp4

media-intake captions --source-id youtube-dQw4w9WgXcQ \
  --url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' --language en-US \
  --output-dir /private/media

media-intake acquire --source-id youtube-dQw4w9WgXcQ \
  --url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' --language en-US \
  --output-dir /private/media

media-intake batch --sources examples/sources.json \
  --output-dir /private/media --stages media,captions
media-intake batch --sources examples/sources.json --stages metadata
```

The example source file is self-contained and contains no local/private paths or
media. Replace it with sources you are entitled to process. Permission checks,
repository-location restrictions, dataset lineage, and downstream eligibility
are intentionally host-application responsibilities; this generic package does
not infer or fabricate them.

## Validated policy, language, and captions

`MediaFormatPolicy` accepts only bounded resolution and known combinations:

* H.264 + AAC in MP4 or MKV;
* VP9 + Opus in WebM or MKV;
* AV1 + Opus in WebM or MKV.

The yt-dlp format expression is assembled internally; callers cannot inject an
arbitrary expression. Audio must be marked `original`, match the exact requested
language, match the codec/container policy, and not be marked dubbed. Missing
matches fail closed. ffprobe verifies the requested container family, one
expected video stream, one expected audio stream, resolution, codecs, and
duration. (ffprobe reports WebM and Matroska through the same demuxer alias.)

`CaptionPolicy` separately requests a same-language publisher track and/or an
explicitly original automatic track (`*-orig`). Publisher tracks are not
necessarily human-authored. Plain automatic tracks are not silently treated as
original. Automatic translations are disabled by default and are requested only
through `automatic_translations=(...)` or repeated CLI
`--automatic-translation LANG` flags.

Every requested caption has a distinct status:

* `downloaded`: timed WebVTT was fetched and validated;
* `unavailable`: metadata advertised no matching track;
* `failed`: a selected track failed to download or validate.

A failed caption makes the stage `partial`; it is not rewritten as unavailable.
Caption attempts are independent of media reuse, so a rerun refreshes captions
even when a verified video already exists.

## Private artifacts and provenance

Treat full metadata, signed URLs, complete captions, and media as private. By
default outputs include:

```text
<root>/<video-id>.<container>
<root>/<video-id>.info.json
<root>/<video-id>.media-receipt.json
<root>/attempts/<uuid>/youtube-<video-id>/
  source.info.json
  media-record.json / captions-record.json / record.json
  ffprobe.json
  captions/<origin>/*.vtt
  *.log
```

Attempts use generated names and new-only JSON writes, preserving successful and
failed history. New media and its newly generated sidecar are downloaded into a
private attempt staging directory, fully validated, and then published without
overwriting existing evidence. A sidecar left at the root without media is never
reused; quarantine it before retrying. SHA-256 receipts validate reuse of media
and the original yt-dlp sidecar. Existing unreceipted media plus sidecar require
explicit `adopt_existing=True` or `--adopt-existing`; adoption validates all
selected formats, the container, and the probe. For adopted files, unknown
original download time remains `null`. Receipts and stage records include the
normalized source and requested policies, and receipt reuse requires the schema
for the calling compatibility surface. `acquisition_time` is the wrapper's
download completion time for a new acquisition, not a source recording or upload
time. `upload_date`/platform timestamps come from metadata, while `checked_at`
and `verified_at` are later local inspection times.

Files and new directories receive private modes where supported, without
changing the process umask. Identifiers and paths are constrained, output
symlinks (including generated-path components and lock files) are rejected,
subprocesses use `shell=False`, and network/process timeouts and retries are
bounded. Both standalone and compatibility CLIs use the same per-root lock,
which supports POSIX `flock` and Windows `msvcrt.locking`.

## Migrating the parent application's old CLI

The parent repository still supports:

```bash
python -m latent_timing_duplex.data.podcasts --sources configs/podcast_sources.json
python -m latent_timing_duplex.data.podcast_download \
  --sources configs/podcast_sources.json --output-dir "$HOME/Videos/latent-timing-duplex"
```

Those thin wrappers translate the old `podcast-sources/v1` configuration,
preserve legacy root artifact names, `acquisition-runs/` layout and
`podcast-*-receipt/v1` integrity checks, and add project-only permission and
pilot-eligibility fields. New generic configurations use
`media-intake-sources/v1` as shown in `examples/sources.json`.

## Known limits

* Only canonical public YouTube watch URLs are accepted; authentication and
  access-control bypass are out of scope.
* “Original” depends on yt-dlp/platform metadata and should be audited by ear.
* ffprobe checks stream/container metadata and duration, not every decoded frame.
* SHA-256 detects local changes; it is not a signature or proof of legal origin.
* Stereo does not establish speaker separation, identity, natural pause timing,
  or suitability for any dataset or pilot.
* Live media acquisition is integration behavior of upstream network tools. The
  package's automated tests use injected offline adapters and do not claim to
  test a live new-download branch.

See `NOTICE` for attribution and `LICENSE` for Apache-2.0 terms.
