# Podcast acquisition and labeling preparation

The three sources in `configs/podcast_sources.json` are engineering-demo inputs,
**not approved pilot data**. The user confirmed acquisition permissions in chat;
this does not assert broader redistribution rights. The locked
[labeling protocol](DATA_LABELING_PROTOCOL.md) remains unchanged.

| Video ID | Publisher/channel | Episode | Duration | Reported upload date |
|---|---|---|---|---|
| kWPvdAZGd3A | Standard Capital | Founders Should Avoid Bad Investors | 19:35 | 2026-10-13 |
| BHiYomkfoGQ | Latent Space | Robots That Predict the Future: Dyna-2 & Physical AI — Jason Ma, Dyna Robotics | 1:21:05 | 2026-10-13 |
| DprKDXRlubw | The Ezra Klein Show | What Americans Need to Understand About China | 1:42:37 | 2026-07-14 |

All advertise original-English stereo audio and dubbed alternatives. Stereo
is not evidence of speaker separation. The initial caption acquisition found
original-English **automatic captions** for all three, but no publisher-provided
caption tracks. These are machine-generated transcripts, not gold labels.

## Requirements and metadata-only inspection

Install the workspace with `uv sync --locked --extra dev` and provide `yt-dlp`,
`ffmpeg`, and `ffprobe` on PATH. No GPU, model download, API credentials, or ASR
service is needed. The reusable implementation and its standalone installation,
Python API, per-stage CLI, format policy, dependency origins, and limits are in
[`packages/media-intake/README.md`](../packages/media-intake/README.md). Its CLI
lock supports POSIX and Windows; these project wrappers retain their old command
names and configuration. Tool versions are recorded per run; keep yt-dlp current
if YouTube extraction changes. Current yt-dlp YouTube extraction may additionally
need a supported JavaScript runtime and EJS challenge components, as documented
upstream. The Python lock does not pin yt-dlp/FFmpeg system binaries.

The original metadata-only command remains available and never downloads media:

```bash
.venv/bin/python -m latent_timing_duplex.data.podcasts \
  --sources configs/podcast_sources.json
```

## Repeatable acquisition

From the repo root:

```bash
.venv/bin/python -m latent_timing_duplex.data.podcast_download \
  --sources configs/podcast_sources.json \
  --output-dir "$HOME/Videos/latent-timing-duplex"
```

Every configured source must have `rights_status: "user_confirmed"`. Add new
sources only after verifying their intended use. Keep permission documentation
privately; the config records confirmation, not an independent legal judgment.

The command:

1. Fetches and saves current full yt-dlp metadata in a new run directory.
2. Selects H.264 video up to 1080p and explicitly original, requested-language
   AAC audio, merging to MP4 without re-encoding. Missing matching formats fail
   closed rather than substituting a dub. Original video timing is retained.
3. Checks the result with ffprobe (container, streams, codecs, resolution and
   duration), hashes the MP4 and original acquisition metadata, and writes a
   media receipt containing the normalized source and format policy.
4. Separately downloads available matching publisher and explicitly original
   automatic captions as **timestamped WebVTT transcripts**. It does not
   substitute auto-translations or run ASR when captions are unavailable.
5. Writes a per-source `record.json` and a batch `run.json`, with tool versions,
   provenance, paths, SHA-256 hashes, and explicit caption statuses.

### Metadata retained

- Source URL and video ID; title and description.
- Channel/publisher name, ID and URL; uploader name, ID and URL.
- Platform-reported upload date, timestamp and release timestamp when available.
- Duration, chapters, categories and tags when available.
- Original audio selection, media streams, full format inventory, reported
  license, availability, and metadata-refresh/verification times.
- Caption origin (`publisher` or `automatic`), language key, and artifact hash.

Null values remain null; upload date is not assumed to be recording date.
Channel/uploader is publishing provenance, not speaker identity.
Publisher-provided captions are not necessarily human-authored. Preserve raw
VTT cue timing and overlapping/rolling cues; downstream conversion to plain
text must not treat repeated rolling cues as new utterances. A transcript
annotator can open a VTT in a text editor or a compatible media player.

### Layout (all outside Git)

```text
~/Videos/latent-timing-duplex/
  <video-id>.mp4
  <video-id>.info.json              # immutable original acquisition metadata
  <video-id>.media-receipt.json     # MP4 + original metadata checksums
  acquisition-runs/<unique-run>/
    sources.json                   # config snapshot, including permission note
    run.json                       # batch status and record paths
    youtube-<video-id>/
      source.info.json             # refreshed full metadata
      ffprobe.json
      download.log
      intake-record.json           # generic package record
      record.json                  # host policy + curated provenance and hashes
      publisher/<video-id>.<language>.vtt   # when available
      automatic/<video-id>.<language>.vtt   # when available
```

The first manual acquisition also left `download.log`, `download-archive.txt`,
and `captions/automatic/` at the root. The repeatable command does not depend on
that archive; the run records identify its verified artifacts explicitly.

### Retries, failures and integrity

- Local yt-dlp configuration and playlist traversal are disabled.
- Network requests have timeouts and retries. New media and sidecar outputs are
  produced in a private run staging directory and validated before publication.
- Existing receipted media is rehashed and checked, not downloaded again.
- Captions and metadata are refreshed in separate immutable run directories,
  even when media already exists. A video download archive cannot hide a failed
  caption fetch. Previous successful caption files are retained.
- Missing captions are `unavailable`, not a download error. Failed caption
  requests are `failed`, make the source `partial`, and cause exit code 1.
  An error for one episode does not stop processing the other sources.
- Changed receipted media/metadata or a missing receipted MP4 fails rather than
  silently overwriting evidence. Inspect the run record and logs. Quarantine
  damaged artifacts before explicitly reacquiring them.
- If interrupted before a media receipt is created but after both the MP4 and
  sidecar exist, rerun with `--adopt-existing`. This also imports an earlier
  yt-dlp download **only with matching `.info.json` establishing the complete
  selected video/audio policy**. It verifies rather than redownloads. A root
  sidecar without media is never reused and must be quarantined before retry.
  Existing receipts and their compatibility schemas are still enforced.
  Adoption time is recorded; an unknown original download time is not invented.
- Output roots inside Git working trees are rejected. A per-root file lock
  prevents concurrent downloader processes; new files use private permissions.

ffprobe checks container/stream metadata and duration, not a complete frame-by-
frame decode or speaker audition. SHA-256 receipts detect local accidental
changes, but are **not authenticated signatures or proof of origin**. Imported
sidecars require trust in the local acquisition process. This is a private,
local intake tool, not an adversarial multi-user storage service. New paths and
files receive scoped private modes where supported; the library does not change
the calling process's umask.

Full metadata can contain signed media URLs; transcripts contain complete
future dialogue. Keep **all acquisition outputs coordinator-private**. Neither
transcripts, complete episodes, nor source mappings belong in blinded prefix
annotation packages. The downloader does not bypass login or access controls.

## Next implementation slices

1. **Channel audit:** listen to each channel over multiple speaker turns; inspect
   edits, ads, music, overlap and clipping. Request isolated microphone tracks
   when possible. Diarization identifies *who spoke when* but does not isolate
   simultaneous speakers. Masking mixed audio by diarization is not valid
   speaker separation. Model-separated audio requires separate provenance and
   validation; it is not automatically eligible for protocol v1.
2. **Candidate preparation:** for approved speaker-separated recordings, audit
   VAD, assign user/partner roles, preserve the original timing reference, and
   connect to `experiments/silence_graft/build_manifest.py` and `render_clips.py`.
   All extracts from an episode must stay in the same lineage/split; reconcile
   related source lineages before splitting. Chunks are not new conversations.
3. **Collection UI:** consume existing blinded assignment packages, collect
   prefix-only action/confidence/quality and acceptable onset, and export strict
   responses for `compile_annotations.py`. Translate onset delays relative to
   the checkpoint into private elapsed-time coordinates server-side; do not
   display hidden numeric pause duration. Lock all required prefixes before
   authenticated reveal release. Do not expose transcripts or video in the
   existing audio-only pilot.

## Evidence limits

Three source episodes cannot meet the locked 60–90-item, at-least-30-lineage
reliability gate. They are useful for ingestion/UI smoke tests only. Do not
lower the gate, invent lineages, or call demo annotations a passing pilot.
Edited podcast pauses may not preserve natural conversational timing; source
quality needs review independently of successful downloading.

No channel separation, VAD run, annotation UI, or real labeling result is
implemented by the acquisition command.
