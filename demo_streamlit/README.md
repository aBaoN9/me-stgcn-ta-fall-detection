# FallScope / Streamlit Video Lab

Local research prototype requested by Gia Bao on 22 November 2026. Video upload,
playback and skeleton on the left; model selection, window scores and synchronized
charts on the right. The compact blue-and-white UI has no marketing hero or
person/run selector. Labels put the model first, then the variant: ME-STGCN-TA / A3,
TCN / Small v2, GRU / Small v2, CTR-GCN / Adapted and SkateFormer / Adapted small. Motion-SVM / Base and
ST-GCN / Base (3 channels) also appear, but inference is explicitly unavailable:
the repo has notebooks 02/03, not verified baseline checkpoint files. They are not
aliases for A0 or A3; no substitute weights, invented metrics or automatic training.
This is new code, not a restoration of the removed PoseLab.

SkateFormer is imported unchanged from Tung's PR #4, pinned commit
`20c67f57c87d1a2cfc0489bf23da45522c4c8d2c`, following local review on 23 November 2026.
It has 12 verified research-fold checkpoints (three seeds, four subjects), its own
artifact manifest and predictions. PR #4 was merged into main on 23 November 2026
at `3c6a39eaf4ee4d37e0848b94bad800aa2dcd4e3a`. All five models belong to the
accepted research benchmark; this does not replace deployment weights or promote
SkateFormer above the other models. The internal `accepted` run key is retained
for saved-session compatibility. Its fixed-threshold mean Accuracy is 84.17%,
not a new video metric.

## Start

Use Python 3.11 and an isolated environment. Do not upgrade the system CUDA or
PyTorch installation just to run this CPU demo. If a compatible CPU PyTorch is
already installed, `python -m venv --system-site-packages .venv-streamlit` can reuse
it; otherwise create a normal environment and install a suitable CPU build first.

```powershell
python -m venv .venv-streamlit
.venv-streamlit/Scripts/python -m pip install -r demo_streamlit/requirements.txt
.venv-streamlit/Scripts/python -m demo_streamlit.prepare_assets
./demo_streamlit/run.ps1
```

Open http://127.0.0.1:8501. The launcher prefers `.venv-streamlit`, binds localhost,
sets the upload limit to 50 MB and disables Streamlit usage telemetry. The one-time
asset setup downloads MediaPipe full float16 version 1 from Google's model bucket
and verifies SHA-256 `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`.
It lives in ignored `.runtime/fallscope-assets`, not Git. Inference does not fetch
models or send video to external services. Google Fonts is optional; there is a
local font fallback. MediaPipe/FFmpeg retain their own dependency/license terms.

## Use

1. Select one-person MP4/MOV/AVI/WEBM: at most 50 MB, 60 seconds, 3,600 frames,
   Full HD pixel count, 1-60 FPS and at least two frames. Unsupported, corrupt,
   truncated or detectably variable-timestamp media fails visibly. Convert to a
   constant-FPS MP4 if necessary; timestamp validation is bounded, not a complete
   media-forensics check. No webcam/live-stream mode yet.
2. Select A3, TCN, GRU, CTR-GCN adapted or SkateFormer adapted small directly, without selecting a contributor
   or experiment run. Advanced settings choose a seed,
   held-out-subject fold and a trailing window of 1-6 seconds (default 3 seconds).
   This is one **research-fold checkpoint**, not an ensemble, final deployment
   model or a guarantee that the uploaded person is held out.
3. Click **Phan tich video**. A session-local background job shows loading, pose,
   inference and preview progress; **Huy phan tich** requests cooperative cancellation.
   The main UI stays usable. Changing the input/model/settings or clearing the upload
   cancels the old job and prevents stale publication. Only one active job per session;
   another can start once cancellation finishes. Workers do not access Streamlit APIs.
   Processing completes first; then the H.264 preview,
   skeleton, Fall score, pose coverage, forward latency, score chart and torso-angle
   chart follow the browser video's `currentTime`. Pause freezes; seek rebuilds
   the charts up to that time. Playback speed and skeleton toggle are available.
   Preview strips audio and preserves the accepted constant frame timing.
4. The event strip explicitly summarizes all precomputed regions with score >=0.5;
   this is not a clinical event detector. Warm-up and insufficient-pose windows
   have **no prediction**, not a default ADL decision. If no windows qualify, the
   app reports that fact. Torso angle is geometry, not model attention.
5. The **Vung can xem lai** list has a thumbnail, start/end time and peak score for
   each high-score region. Click a card to pause and seek just inside its start
   (1 ms avoids browser timestamp rounding to the preceding window). The list
   summarizes the entire precomputed video, not a causal warning feed. Thumbnails
   come from the original video at the region start and are not new model evidence.
6. The session summary separates valid windows, high-score duration, warm-up,
   unavailable time after warm-up and processing time. Download printable HTML
   (browser Print / Save as PDF), numeric CSV and JSON with window scores,
   timings, events, input hash and checkpoint identity. All three exclude video,
   thumbnails and landmarks. Downloading does not restart playback.
   Changing video/model/run/seed/fold/window
   clears the previous result; inference never starts just because a selector
   changes. Use **Xoa video / ket qua** to clear the upload/result for this session.

## Opt-In Local History

- Nothing is written to history automatically. **Luu phien tren may nay** stores
  the report, original filename, score trace and checkpoint metadata in ignored
  `.runtime/fallscope/history.sqlite3`. The default excludes video, thumbnails,
  poses and torso-angle tracks. Even a report contains potentially sensitive
  names and activity scores; only save appropriate data.
- A separate unchecked checkbox explicitly opts in to storing the silent H.264
  preview, event thumbnails, poses and angles for synchronized replay. The original
  upload/audio is never included. This is local storage, not encrypted backup.
- **Lich su tren may nay** lists saved sessions across restarts. Open report-only
  records for summary/downloads, or replay records that include video. Deleting
  requires selecting a record, checking the confirmation and clicking delete.
  Removing history does not delete independently downloaded reports or backups.
- Storage is capped at 20 records / 250 MiB of logical payload, 64 MiB per record.
  It refuses new saves at the limit rather than silently evicting old records.
  SQLite uses transactions, bound parameters, integrity hashes, secure-delete and
  compaction after deletion; connections always close, including on Windows.
  This does not guarantee forensic erasure on SSDs, snapshots or OneDrive backups.
- History belongs to this local installation, not an authenticated user. Anyone
  able to reach the app can access it. Keep the localhost binding; multi-user
  identity, encrypted storage, authorization and public hosting remain out of scope.

The iframe uses self-contained HTML/CSS/JavaScript and SVG, with no external chart
runtime. Columns stack on small screens; the iframe scrolls internally to expose
the stacked monitor. Controls are labeled and keyboard accessible. JSON embedded
in HTML escapes script delimiters, and dynamic text uses textContent.

## Actual Pipeline And Limits

- MediaPipe Tasks VIDEO mode, one pose, confidence thresholds 0.35, all decoded
  frames: x/y/z/visibility plus a detected-frame mask. There is no multi-person
  identity tracking; use footage with one visible subject.
- At each approximately 0.25 seconds, take only past frames in the trailing window
  (at least 32 frames). Require >=50% pose frames and a detected endpoint. Interpolate
  only inside that window; sample 32 indices; call the original feature v2 function
  for hip centering, torso normalization, velocity, angle and displacement channels.
- Use the actual accepted research checkpoint on CPU. SHA-256/bytes, model/seed/fold, feature
  version and exact train/test identities are checked before weights-only loading.
  Each job loads a verified model without a global upload/result cache; artifacts
  are trusted, immutable inputs. Restart the server after replacing research assets.
- **Precompute then synchronized playback**, not live streaming inference. The
  models were evaluated on clips, not this sliding-window protocol. Window length,
  pose gaps and domain shift can change scores. No new Accuracy, detection-delay,
  sensitivity or false-alarm-rate claim is established by this demo.
- Fall score is a clip/window softmax, not calibrated probability or Accuracy.
  Threshold stays fixed at 0.5 for illustration. CPU forward time excludes pose
  extraction, feature work, decoding, UI and transcoding; total processing and pose
  times are reported separately. Do not compare it directly with archived GPU FPS.
- Benchmark Accuracy/F1/Sensitivity/Specificity are a separate collapsed section:
  three-seed mean/sample SD on the same 160 clips/4 subjects, not 480 independent
  videos. These numbers do not animate on an unlabeled upload. The optional local
  tuning T4 archive is not shipped in this update and remains PARTIAL: weights/raw
  OOF are missing, so it is not offered in the application
  selector. The evidence loader still rejects inference for that run, with no
  fallback to another run's model. No training or deployment-asset changes occur.
- Private uploads/results stay per Streamlit session, never in a global data cache.
  Temporary decode/transcode files are removed after processing/cancellation.
  Cancellation is cooperative at frame/window/FFmpeg-poll boundaries, not instant
  interruption of imports/native calls. Jobs stop at the next boundary after four
  minutes; FFmpeg separately has a 120-second bound and is killed/reaped on cancel.
  The app persists only explicitly saved sessions; memory can remain until
  reset/session cleanup. Unselected history is never copied to a global data cache.
  It is a bounded local prototype, not hardened for untrusted public hosting.

## Checks

```powershell
.venv-streamlit/Scripts/python -B -X utf8 -m unittest discover -s demo_streamlit/tests -v
python -B -X utf8 scripts/check_repository.py
```

Tests cover the model-only UI, model switches, missing weights, stale-result reset, input/evidence
hash guards, causal windows/no future interpolation, warm-up/pose gaps, invalid
scores/media, event gaps and HTML payload escaping. Tests also cover report
privacy/escaping, opt-in history/save/open/delete/storage bounds,
SQLite tamper detection, background completion/failure/cancellation and FFmpeg cleanup;
the replay regression
still checks all 60 accepted research checkpoints (including 12 SkateFormer PR #4)
against one real held-out sample each.
Local smoke additionally exercises videos through the model builders and a
generated blank video with zero valid predictions. These are software checks, not
a new out-of-domain evaluation. Medical/safety deployment remains out of scope.
