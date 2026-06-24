# Re-encode backlog plan

Planning document for re-encoding the legacy "old pipeline" behavior-video backlog into the AIND file-standards format. Captures decisions, open questions, and TODOs as of this session.

## Terminology

| Name | What it is | Where it lives |
|---|---|---|
| **Raw** | Original camera capture, linear-light, bgr24/yuv420p, no transfer function. | `/home/galen.lynch/encode-testing/raw/` (3 confirmed no-gamma files: `gamma_no`, `testSide`, `testBottom`). Source raws for the 13k production videos are presumed lost / not available. |
| **Old pipeline** | Legacy encoder used by the previous behavior-video pipeline. `h264_nvenc -preset fast -crf 23 -b:v 50M`, linear-light values stored directly (no transfer function applied). Called `pipeline_encode` in the historical `benchmarks.csv`. | Production outputs at `/mnt/Data/ephys/tongue-tracking/ecephys_786867_2025-09-25_12-43-56/behavior-videos/`; ~13,000 such files exist project-wide. |
| **New online** | AIND file-standards `ONLINE_8BIT` profile in `src/aind_video_utils/encoding.py`. h264_nvenc, real-time-acquisition target. | Defined in `encoding.py`; canonical spec source. |
| **New offline_8bit** | AIND file-standards `OFFLINE_8BIT` profile. `libx264 -preset veryslow -crf 18`, applies BT.709 transfer function — output pixel values are gamma-encoded. Long-term-storage target. | Defined in `encoding.py`. |
| **Cascade / two-stage** | The actual production path: old-pipeline output → new offline_8bit re-encode. This is what we're sizing for the 13k-video re-encode backlog. | Not yet measured on real data this session. |

## Goal

Re-encode ~13,000 old-pipeline videos (5.5 Mbps h264 in linear-light yuv420p, 720×540, ~1000 fps, ~84 min each, ~2.5M frames each, ~3.3 GB each, ~43 TB total) into a faster variant of `OFFLINE_8BIT` that:

1. Is visually indistinguishable from the old-pipeline source for downstream behavioral CV (which downsamples 4×).
2. Completes in reasonable wallclock and EC2 cost.
3. Stays within the AIND file-standards encoding-profile family (libx264, BT.709 TRC) so outputs are spec-compliant.

## Decisions made this session

1. **CPU, not GPU.** NVENC engines (1–3 per GPU) don't scale like CPU cores; many parallel libx264 jobs across cores beat GPU on cost and total throughput for offline batch. Also `OFFLINE_8BIT` is libx264 by spec — reproducible across hardware.
2. **Preset is the main throughput knob, not CRF.** libx264 CRF affects fps only 15–50%; preset is 6–7×. CRF should be picked for quality/storage, preset for throughput.
3. **Faster preset is on the table.** Historical CSV (linear-vs-linear comparison) shows `fast` vs `veryslow` at matched CRF: 6.4× speedup, 0.75 VMAF loss, essentially identical compression. With the 4× downsample in Lightning Pose, this is very likely acceptable.
4. **Aggressive CRF probably defensible too.** Old-pipeline source is already lossy (~95 VMAF vs raw in linear space, lower in display space). CRF 20–22 may be the sweet spot given the upstream loss already baked in.
5. **Display-space (BT.709-gamma-encoded) is the right metric space for the cascade benchmark.** Historical CSV numbers were linear-vs-linear; not directly comparable to a cascade measurement that includes the new offline_8bit's TRC step.
6. **EC2 sizing**: family bake-off → contention test → fan out across multiple smaller spot instances rather than one bare-metal box.
7. **Sharding**: static `WORKER_ID` env-var sharding (simplest) or AWS Batch array jobs (more polished). Both viable.

## Open questions

- **Final preset/CRF**: depends on visual eyeball + cascade metrics. Candidate grid is `preset ∈ {fast, medium, slow}` × `crf ∈ {18, 20, 22}`.
- **Where are the original raw sources for the production 13k?** If they're available somewhere, we could re-encode from raw and skip the cascade altogether — would give better quality at the same offline cost. If they're truly gone, cascade is the only option.
- **Pipeline_encode bitrate mystery**: production files are 5.5 Mbps, benchmark `pipeline_encode` of `gamma_no` came in at 53 Mbps with the same flags. Either NVENC defaults shifted between when the production files were made and the benchmark, or content compresses very differently. Worth verifying on a fresh re-encode of one of the raw files before trusting cascade numbers.
- **Single instance × longer wallclock vs many instances × short wallclock**: a 1× c7i.24xlarge spot job runs ~8.5 days for the full batch; 4× runs ~2 days. Cost is roughly the same, complexity differs.
- **AWS Batch vs manual sharding**: depends on appetite for one-time setup vs simpler ad-hoc launches.

## Critical gotcha: gamma / TRC mismatch in metrics

Old-pipeline stores **linear-light** values. `OFFLINE_8BIT` outputs **BT.709-gamma-encoded** values. A naive `[ref][dist]libvmaf` comparison between raw and an offline_8bit cascade output is apples-to-oranges — the pixel values are on different scales because of the gamma curve itself, not because of encoder loss. Will yield artificially terrible numbers (~60s–70s VMAF for what's actually a near-perfect encode).

**Fix**: convert both metric inputs into a common color space (BT.709 display space recommended) before VMAF/PSNR/SSIM. Apply via the `colorspace` filter in the lavfi graph inside `get_vmaf_score`, `get_psnr_score`, `get_ssim_score`.

```
[0:v]colorspace=itrc=linear:otrc=bt709:all=bt709:format={fmt}[reference];
[1:v]colorspace=itrc=bt709:otrc=bt709:all=bt709:format={fmt}[distorted];
[distorted][reference]libvmaf=...
```

Historical `benchmarks.csv` numbers (e.g., `pipeline_encode` VMAF 95.27, `h264_slow_crf_18` VMAF 93.53) were linear-vs-linear and **will not match** display-space cascade numbers. That's expected and acceptable; the historical numbers were measuring h264-only loss in linear, which isn't the question anymore.

## TODOs

### Local — quality decision

- [ ] Update `get_vmaf_score`, `get_psnr_score`, `get_ssim_score` in `benchmarks/benchmarking.py` to take explicit input TRCs (`itrc_ref`, `itrc_dist`) and convert both to BT.709 display space before metric computation.
- [ ] Build cascade benchmark driver:
  - Inputs: `gamma_no` + `testSide` (linear raws confirmed earlier in this session).
  - Stage 1: encode each raw with old-pipeline flags (`h264_nvenc -preset fast -crf 23 -b:v 50M`).
  - Stage 2 candidates: `OFFLINE_8BIT.replace(codec_params=("-preset", P, "-crf", C))` for `(P, C) ∈ {fast, medium, slow} × {18, 20, 22}`, plus the current veryslow CRF 18 default as anchor.
  - Also measure: direct one-stage `OFFLINE_8BIT` from raw (no old-pipeline first stage) as upper-bound reference.
  - Metrics: VMAF, PSNR-Y, SSIM-Y in BT.709 display space against raw.
  - Output: `/mnt/Data/encodes/cascade_<date>/results.csv`.
- [ ] Visual eyeball: extract 3 × 15-second clips from one of the 786867 tongue-tracking videos (`-c:v copy`, no re-encode), re-encode each at the candidate (preset, crf) combos, side-by-side viewing against the old-pipeline input. No metrics for this — just human judgment on whether tongue / paw / whisker detail is preserved.
- [ ] Pick final `(preset, crf)` based on metric data + eyeball.

### EC2 — throughput decision

- [ ] Family bake-off: spin up `c8a.4xlarge`, `c8i.4xlarge`, `c8g.4xlarge` (or c7 equivalents if c8 unavailable in region). On each: confirm `x264 --version` shows expected SIMD (AVX-512 on x86, NEON/SVE on Graviton), then run 1 ffmpeg with chosen preset+CRF on one test video. Compare $/frame.
- [ ] Scaling test on winner at `8xlarge`: sweep `(jobs, threads)` = `(16, 1), (8, 2), (4, 4)` and `(32, 1)` (SMT zone). Find contention point.
- [ ] Verify scaling extrapolates with one short run on production size before committing the full batch.

### Production — batch run

- [ ] Build manifest CSV of all 13k inputs with target output paths.
- [ ] Add to `aind-transcode` CLI (or write a driver script):
  - `--manifest <path>` mode (alongside existing directory mode).
  - `--failure-log <path>` for structured per-video failure tracking.
  - `--verify` flag: probe output post-encode, compare frame count to source.
  - Atomic `.tmp` write + rename inside `transcode_video()`.
  - Optional `--worker-id N --workers M` for static sharding.
- [ ] Pick sharding mechanism: static `WORKER_ID`/`N_WORKERS` env vars set via EC2 user-data (simplest) OR AWS Batch array jobs.
- [ ] Decide staging: stream inputs from S3 via `http_input_flags` (simpler) vs pre-download to instance NVMe (faster, more setup).
- [ ] Outputs to S3 with multipart upload (atomic by default; partial uploads aren't visible).
- [ ] Launch.
- [ ] Reconcile failures from `failures.csv`, re-run idempotently.

## Files / locations to remember

- `src/aind_video_utils/encoding.py:100` — `OFFLINE_8BIT` profile definition (current `-preset veryslow -crf 18`).
- `benchmarks/benchmarking.py` — VMAF / PSNR / SSIM scoring + encode-stats helpers. Lives outside the installed package; sibling to `src/`.
- `notebooks/benchmarks.py` — historical sweep driver (fixed this session to use `sys.path` to find `benchmarking`).
- `/home/galen.lynch/encode-testing/raw/` — 3 linear raw videos for cascade benchmarking.
- `/home/galen.lynch/encode-testing/benchmarks.csv` — historical benchmark results (linear-vs-linear, not directly comparable to display-space cascade numbers).
- `/mnt/Data/encodes/` — scratch dir for this session's cascade artifacts.
- `/mnt/Data/ephys/tongue-tracking/ecephys_786867_2025-09-25_12-43-56/behavior-videos/` — example production directory (3 cameras × ~84 min × 1000 fps × 720×540, ~3.3 GB each as old-pipeline output).
