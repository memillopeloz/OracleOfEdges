# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project name: **OracleOfEdges**. Executable: **`argus`** (the hundred-eyed giant). The workspace folder is still named `PuzzleSeer`.

## Build

Dependencies auto-fetched via CMake FetchContent (`cpp-httplib` v0.15.3, `nlohmann/json` v3.11.3, prebuilt ONNX Runtime 1.20.1 linux-x64). OpenCV 4.x must be installed on the system.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Targets: `argus_core` (static lib: everything but `main.cpp`), `argus` (server), `argus_bench` (benchmark). All four matchers are always compiled; there is no optional-ML build. `-DONNXRUNTIME_DIR=/path` overrides the downloaded ONNX Runtime with a local install. The MobileNetV2 model (`config/mobilenetv2.onnx`) is tracked in the repo; if it's missing, the ML matcher reports itself unavailable at runtime rather than failing the build.

## Run

The binary must be run from the `build/` directory (CMake copies `config/` and `web/` there post-build):

```bash
cd build
./argus
./argus --port 8080 --config-dir ../config --web-dir ../web
```

Required before first run: place `artwork.jpg` (or `.png`/`.bmp`) in `config/`.

VS Code: the default build task configures and builds with CMake; IntelliSense reads `build/compile_commands.json`. `config/` in the repo root is the source of truth; it gets copied into `build/config/` on each build, so edit the repo copy.

## Architecture

The server flow is: HTTP upload (piece photo + the set of matchers the user selected) → `WebHandler` → selected `Matcher` instances in parallel → results rendered as an HTML table (coordinate, confidence, time per method). The server never votes or picks a winner; the user compares results. Keep the matching logic separate from route registration.

**`Matcher` (abstract interface, [src/Matcher.hpp](src/Matcher.hpp))** — lifecycle is `prepare(artwork, grid)` once at startup (precompute per-cell crops/histograms/keypoints/embeddings, keep the grid), then `match(piece) const → MatchResult` per request. `match` runs concurrently from multiple threads and must not mutate state. `available()` returns false when a matcher can't run (e.g. ML model missing). `MatchResult` carries row, col, coord string (e.g. `"B3"`), confidence (0.0–1.0), `timeMs`, matcher name, and `error` (empty on success). Build results with `matcher_utils::makeResult`, which stamps label and elapsed time.

**`matcher_utils` ([src/matchers/MatcherUtils.hpp](src/matchers/MatcherUtils.hpp))** — shared helpers: `limitSize`, `centerCrop` (trims table background/knobs from the photo), `cellTiles` (overlapping piece-sized sub-windows of each cell, used by Histogram and ML), `cellAt` (artwork pixel → cell).

**`GridConfig` ([src/GridConfig.hpp](src/GridConfig.hpp))** — parses `grid.json`; key methods are `cellRect(row, col, artworkSize)` which slices the artwork into cells, `coordLabel(row, col)` for human-readable coordinates, and `allCells()` for iteration. Declarations in the header, implementation in `GridConfig.cpp`. At parse time: missing labels are generated, label counts are validated, and `artworkPath` is resolved against the config directory (explicit `artwork_path`, else a search for `artwork.*`).

**`WebHandler` ([src/WebHandler.cpp](src/WebHandler.cpp))** — owns artwork `cv::Mat`, `GridConfig`, and all matcher instances; prepares them in parallel at construction. `locate(piece, selectedNames)` runs the selected matchers (case-insensitive by `name()`) in parallel. `registerRoutes(svr)` adds the JSON API: `GET /api/info` (grid + matchers with availability), `GET /api/artwork` (downscaled overview JPEG, encoded once at startup), `POST /api/locate` (multipart: `piece` = PNG cut-out, one `matcher` field per selected matcher → `{results:[…]}`; 400 with `{error}` on bad input). `main.cpp` mounts `web/` at `/`.

**Web page ([web/](web/))** — static, no build step, no libraries. [cutout.js](web/cutout.js) finds the piece with no DOM dependency (background modelled as a per-channel plane fitted to a border frame → colour distance with luminance at half weight → Otsu × user sensitivity → close/open → largest blob clear of the border → fill holes), at 800 px. [app.js](web/app.js) shows the outline preview with a looser/tighter slider, uploads a ≤1024 px PNG crop with alpha = outline, and renders the results table plus an artwork overview with each matcher's cell outlined in its colour. Matcher selection persists in `localStorage`. Photos come from `<input capture>`, so plain HTTP on the LAN works (no `getUserMedia`/HTTPS needed).

## Matcher implementations

Each matcher lives in `src/matchers/` (`.hpp` + `.cpp`) and inherits from `Matcher`. Input is a `Piece` built once per request by `matcher_utils::makePiece`: the client sends a cut-out PNG whose alpha is the piece outline (a 3-channel image falls back to the photo's centre 60%). `makePiece` keeps the largest blob, straightens it so the body edges are axis-aligned (length-weighted circular mean of 4θ along the contour — `minAreaRect` is thrown off by knobs), crops, erodes the fringe, and finds `core`, the largest square fully inside the piece. Remaining ambiguity is a multiple of 90°, so matchers try 4 orientations where needed.

`grid.json`'s optional `pieces` gives the expected piece side (`GridConfig::pieceSide`); matchers use it for scale and tile size and fall back to wide searches without it.
- **TemplateMatcher** — grayscale `TM_CCOEFF_NORMED`; background outside the outline filled with the piece mean (≈ masked, but FFT-fast — real masked `matchTemplate` was ~5× slower). Scale from piece area vs. `pieceSide`; artwork reduced so a piece spans ~32 px; coarse pass at half res, top 3 refined.
- **MLMatcher** — MobileNetV2 pooled 1280-d features (exposed in memory at load by `onnx_graph::exposeClassifierInput` in [OnnxGraph.cpp](src/matchers/OnnxGraph.cpp), a minimal protobuf editor — the stock model only outputs class logits, which did markedly worse). Cosine vs. one tile per piece position. Tile embedding dominates startup (~25 s at 5000 pieces). Weakest matcher; denser tiles did not help.
- **HistogramMatcher** — masked H-S histogram (18×16 bins) correlation vs. piece-sized tiles at half-piece stride; ~20 ms.
- **KeypointMatcher** — ORB, ~150 artwork features per piece (cap 800k) spread per cell; piece ORB uses a 12-level ×1.3 pyramid to bridge photo vs. artwork scale. Brute-force Hamming in chunks of 200k (BFMatcher's per-set limit is 2^18) + ratio test + RANSAC homography of the piece centre, falling back to per-cell voting. Best-calibrated confidence (≈0 when wrong).

**Benchmark:** `tools/bench.cpp` → `argus_bench [config-dir] [samples] [pieces...]` (run from `build/`). Simulates client cut-outs (jigsaw mask, rotation, lighting, blur, noise, JPEG) from the artwork and reports per-matcher accuracy, time, and mean confidence on hits vs. misses. Fixed seed, so runs are comparable — re-run it after any matcher change.

**Page test:** `node tools/webtest.mjs <url> <photo> <outPrefix> [sensitivity]` drives headless Brave (override with `ARGUS_BROWSER`; not `BROWSER`, which this desktop sets to a Firefox stub) over the DevTools pipe through photo → outline → locate at phone size, saving screenshots and printing the results table and any JS errors. Needs a running server. Baseline (40 samples, 10587×6800 artwork, 5000 pieces): Template 60%, Keypoint 67%, Histogram 60%, ML 17%; startup ~28 s.

## Config format

```json
{
  "rows": 4,
  "cols": 5,
  "row_labels": ["A", "B", "C", "D"],
  "col_labels": ["1", "2", "3", "4", "5"],
  "artwork_path": "artwork.jpg",
  "pieces": 5000
}
```

`pieces` (optional) is the approximate piece count. `row_labels`/`col_labels` are optional (generated if absent) but, if present, must exactly match `rows`/`cols` — `GridConfig::fromFile` throws otherwise. `artwork_path` is relative to the config directory.

## Code style

- One `return` per function; track failure in a status variable where 0 = success (see `main.cpp`'s `Status` enum).
- Each file includes what it uses directly; don't rely on transitive includes.
- `#pragma once` in all headers.
