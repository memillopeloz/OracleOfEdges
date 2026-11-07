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

Targets: `argus_core` (static lib: everything but `main.cpp`), `argus` (server), `argus_bench` (benchmark). All five matchers are always compiled; there is no optional-ML build. `-DONNXRUNTIME_DIR=/path` overrides the downloaded ONNX Runtime with a local install. The MobileNetV2 model (`config/mobilenetv2.onnx`) is tracked in the repo; if it's missing, the ML matcher reports itself unavailable at runtime rather than failing the build.

## Run

The binary must be run from the `build/` directory (CMake copies `config/` and `web/` there post-build):

```bash
cd build
./argus
./argus --port 8080 --config-dir ../config --web-dir ../web
./argus --samples-dir ../samples        # also enable the temporary admin page /admin.html
```

Required before first run: place `artwork.jpg` (or `.png`/`.bmp`) in `config/`.

VS Code: the default build task configures and builds with CMake; IntelliSense reads `build/compile_commands.json`. `config/` in the repo root is the source of truth; it gets copied into `build/config/` on each build, so edit the repo copy.

## Architecture

The server flow is: HTTP upload (piece photo + the set of matchers the user selected) → `WebHandler` → selected `Matcher` instances in parallel → results rendered as an HTML table (coordinate, confidence, time per method). The server never votes or picks a winner; the user compares results. Keep the matching logic separate from route registration.

**`Matcher` (abstract interface, [src/Matcher.hpp](src/Matcher.hpp))** — lifecycle is `prepare(artwork, grid)` once at startup (precompute per-cell crops/histograms/keypoints/embeddings, keep the grid), then `match(piece) const → MatchResult` per request. `match` runs concurrently from multiple threads and must not mutate state. `available()` returns false when a matcher can't run (e.g. ML model missing). `MatchResult` carries row, col, coord string (e.g. `"B3"`), confidence (0.0–1.0), `timeMs`, matcher name, and `error` (empty on success). Build results with `matcher_utils::makeResult`, which stamps label and elapsed time.

**`matcher_utils` ([src/matchers/MatcherUtils.hpp](src/matchers/MatcherUtils.hpp))** — shared helpers: `limitSize`, `makePiece` (upload → normalized `Piece`), `gridTiles` (piece-sized tiles over the whole artwork, used by Histogram and ML when the piece count is known), `cellTiles` (overlapping sub-windows of each cell, when it isn't), `cellAt` (artwork pixel → cell), `makeResult`.

**`GridConfig` ([src/GridConfig.hpp](src/GridConfig.hpp))** — parses `grid.json`; key methods are `cellRect(row, col, artworkSize)` which slices the artwork into cells, `coordLabel(row, col)` for human-readable coordinates, and `allCells()` for iteration. Declarations in the header, implementation in `GridConfig.cpp`. At parse time: missing labels are generated, label counts are validated, and `artworkPath` is resolved against the config directory (explicit `artwork_path`, else a search for `artwork.*`).

**`WebHandler` ([src/WebHandler.cpp](src/WebHandler.cpp))** — owns artwork `cv::Mat`, `GridConfig`, and all matcher instances; prepares them in parallel at construction. `locate(piece, selectedNames)` runs the selected matchers (case-insensitive by `name()`) in parallel. `registerRoutes(svr)` adds the JSON API: `GET /api/info` (grid + matchers with availability), `GET /api/artwork` (downscaled overview JPEG, encoded once at startup), `POST /api/locate` (multipart: `piece` = PNG cut-out, one `matcher` field per selected matcher → `{results:[…]}`; 400 with `{error}` on bad input). `main.cpp` mounts `web/` at `/`.

**Admin page (temporary, [AdminRoutes.cpp](src/AdminRoutes.cpp) + [web/admin.html](web/admin.html))** — only with `--samples-dir <dir>`. Lets the user photograph sample pieces from the phone and save the original photo under its cell name (`C5.jpg`, next free `C5_2.jpg`, …) or replace an existing one; shows the same outline preview as the main page. Cell labels are validated against the grid and only listed files can be replaced. Delete both files plus the option in `main.cpp` when sample collection is over.

**Web page ([web/](web/))** — static, no build step, no libraries. [cutout.js](web/cutout.js) finds the piece with no DOM dependency, at 800 px, tuned on real photos of sepia pieces on brown cardboard/wood: robust background model from a border frame (per-channel plane + residual covariance, refit on the best 80%) → smoothed Mahalanobis distance → Otsu × user sensitivity → seed blob clear of the border → hysteresis growth through a weaker threshold (thin strands opened away; growth running along or reaching past the seed to the photo edge is a leak) → close/open → main blob → fill holes. [photo.js](web/photo.js) is the canvas side (decode with EXIF orientation, working copy, outline preview, ≤1024 px PNG cut-out with alpha = outline), shared by the page and `tools/cutsamples.mjs`. [app.js](web/app.js) is the UI: outline preview with a looser/tighter slider, results table plus an artwork overview with each matcher's cell outlined in its colour. Matcher selection persists in `localStorage`. Photos come from `<input capture>`, so plain HTTP on the LAN works (no `getUserMedia`/HTTPS needed).

## Matcher implementations

Each matcher lives in `src/matchers/` (`.hpp` + `.cpp`) and inherits from `Matcher`. Input is a `Piece` built once per request by `matcher_utils::makePiece`: the client sends a cut-out PNG whose alpha is the piece outline (a 3-channel image falls back to the photo's centre 60%). `makePiece` keeps the largest blob, straightens it so the body edges are axis-aligned (length-weighted circular mean of 4θ along the contour — `minAreaRect` is thrown off by knobs), crops, erodes the fringe, and finds `core`, the largest square fully inside the piece. Remaining ambiguity is a multiple of 90°, so matchers try 4 orientations where needed.

`grid.json`'s optional `pieces` gives the expected piece side (`GridConfig::pieceSide`); matchers use it for scale and tile size and fall back to wide searches without it.
- **TemplateMatcher** — grayscale `TM_CCOEFF_NORMED`; background outside the outline filled with the piece mean (≈ masked, but FFT-fast — real masked `matchTemplate` was ~5× slower). Outline eroded a further 2.5% of √area first (drops the dark rim). Scale from piece area vs. `pieceSide`, searched at ×1.0–1.6 (√area includes knobs and rim, so the expected scale is too small); artwork reduced so a piece spans ~32 px; coarse pass at half res, top 3 refined.
- **MLMatcher** — MobileNetV2 pooled 1280-d features (exposed in memory at load by `onnx_graph::exposeClassifierInput` in [OnnxGraph.cpp](src/matchers/OnnxGraph.cpp), a minimal protobuf editor — the stock model only outputs class logits, which did markedly worse). Cosine vs. one tile per piece position. Tile embedding dominates startup (~25 s at 5000 pieces). Weakest matcher; denser tiles did not help.
- **HistogramMatcher** — grey only (the print is sepia): per tile, resized to 64 px, a z-scored brightness histogram + rotation-invariant uniform LBP at radii 1 and 2 + gradient magnitude/contrast histogram; mean Hellinger distance vs. piece-sized tiles at half-piece stride. Confidence = margin over the best tile in another cell. ~5 ms. Replaced the H-S colour histogram on 2026-11-06 (real 1% → 3%, sim 1% → 8%); a per-block ablation found no block clearly better than the others. More ideas are to be discussed with the user.
- **KeypointMatcher** — ORB, ~150 artwork features per piece (cap 800k) spread per cell; the piece is shrunk to ~2× its artwork footprint, then ORB with an 8-level ×1.3 pyramid and FAST threshold 8 (the soft print has weak corners; 12 levels, no shrink, when the piece count is unknown). Brute-force Hamming, 8 neighbours, in chunks of 200k (BFMatcher's per-set limit is 2^18); the ratio test compares against the nearest neighbour at least a piece away (closer ones are the same spot, not rivals) + RANSAC homography of the piece centre, falling back to per-cell voting. Best-calibrated confidence (≈0 when wrong).
- **CombinedMatcher** — wraps the Template and Keypoint instances `WebHandler` already prepared (its own `prepare` only keeps the grid), runs both in parallel, returns Keypoint's cell when its confidence ≥ 0.05 (any value in 0.02–0.10 scored the same on real samples), else Template's with confidence × 0.05. Their hits overlap only partly (either right on 53% of real photos). Matchers run concurrently on one shared `Piece`: never write into its buffers (Template's erosion once did, making the bench non-repeatable).

**Benchmark:** `tools/bench.cpp` → `argus_bench [config-dir] [samples] [pieces...]` (run from `build/`). Simulates client cut-outs from the artwork the way a phone photo of this sepia print looks (irregular rectangular jigsaw mask, near-monochrome warm print whose channel weights and ink density vary per piece, blur, light grain, uneven light, table/shadow rim inside the outline, rotation, slight tilt, JPEG) and reports per-matcher accuracy, time, and mean confidence on hits vs. misses. Its constants were tuned so the accuracies track `--real`; fixed seed, so runs are comparable — re-run it after any matcher change, and confirm with `--real`. Baseline (80 samples, 10587×6800 artwork, 5040 pieces, 2026-11-07): Combined 38%, Keypoint 36%, Template 30%, Histogram 8%, ML 0%; startup ~30 s. Real (62 photos, same day): Combined 48%, Keypoint 40%, Template 38%, ML 4%, Histogram 3%. Template and Keypoint take ~3–4.5 s per piece with all matchers running, Combined slightly more. Keypoint's confident misses on real photos are the neighbouring cell (pieces on a cell boundary).

**Real samples:** `samples/` (git-ignored) holds phone photos named by true cell (`C5.jpg`, `C5_2.jpg`). `node tools/cutsamples.mjs [samplesDir] [outDir] [sensitivity]` cuts them out with the page's own code in headless Brave (default output `samples/cut/`, plus `outline/` previews), then `argus_bench config --real ../samples/cut` scores them per photo. First real run (62 photos, 2026-11-05): Keypoint 30%, Template 16%, ML 4%, Histogram 1% — far below the simulated numbers of the time, which led to the realistic simulator.

**Page test:** `node tools/webtest.mjs <url> <photo> <outPrefix> [sensitivity]` drives headless Brave through photo → outline → locate at phone size, saving screenshots and printing the results table and any JS errors. Needs a running server. Browser driving lives in [tools/cdp.mjs](tools/cdp.mjs) (DevTools over `--remote-debugging-pipe`, no npm packages; `ARGUS_BROWSER` overrides the binary — not `BROWSER`, which this desktop sets to a Firefox stub).

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

- Every function has a preceding comment: summary, then `In:`, `Out:` and `Errors:` lines. Public methods are documented at the declaration (header), file-local helpers in the `.cpp`.

- One `return` per function; track failure in a status variable where 0 = success (see `main.cpp`'s `Status` enum).
- Each file includes what it uses directly; don't rely on transitive includes.
- `#pragma once` in all headers.
