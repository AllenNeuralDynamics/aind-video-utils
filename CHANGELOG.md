# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.4.1 (2026-06-24)

### Fix

- use 'unknown' for unset keys

## v0.4.0 (2026-06-24)

### Feat

- **qc**: headless batch exposure QC module + CLI
- **benchmarks**: add PSNR and SSIM scoring helpers; rework notebook layout

### Fix

- **encoding**: probe-aware setparams; remove redundant ispace=bt709 override
- **encoding**: force range=pc in setparams to prevent yuv420p shadow/highlight crush

## v0.3.1 (2026-03-06)

### Fix

- Allow headless use (#3)

## v0.3.0 (2026-03-06)

### Feat

- add https support, and color-range QC figure

## v0.2.2 (2026-03-01)

## v0.2.1 (2026-03-01)

## v0.2.0 (2026-02-19)

### Feat

- return bit depth and range when getting luma frames

### Refactor

- refactor project to have better organization

## v0.0.0 (2024-10-03)
