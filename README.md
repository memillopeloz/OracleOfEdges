# OracleOfEdges

A C++17 server, **Argus** (named for the hundred-eyed giant who sees everything), that accepts a photo of a puzzle piece (from a smartphone browser) 
and locates it within a reference artwork image using four different matching methods.

## Requirements

| Dependency       | Version  | Notes                                    |
|------------------|----------|------------------------------------------|
| CMake            | ≥ 3.16   |                                          |
| OpenCV           | ≥ 4.x    | core, imgproc, imgcodecs, features2d     |
| ONNX Runtime     | 1.20.1   | Downloaded automatically by CMake        |
| cpp-httplib      | 0.15.3   | Downloaded automatically by CMake        |
| nlohmann/json    | 3.11.3   | Downloaded automatically by CMake        |

Only OpenCV must be installed on the system; everything else is fetched via
CMake `FetchContent` on first configure.

## Build

```bash
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Optional: use a local ONNX Runtime install instead of the downloaded one
cmake .. -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=/path/to/onnxruntime
```

## Setup

1. Place your full puzzle artwork image as `config/artwork.jpg` (or `.png`).
2. Edit `config/grid.json` to match your puzzle's grid layout.

The MobileNetV2 model for the ML matcher ships in the repo as
`config/mobilenetv2.onnx` ([ONNX Model Zoo](https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx),
Apache-2.0). If it is missing, the server still starts and the ML matcher
reports itself as unavailable.

`config/` is copied into `build/config/` on every build, so edit the files in
the repo's `config/`, not the copies.

## Run

```bash
cd build
./argus
# or with custom options:
./argus --port 8080 --config-dir ../config --web-dir ../web
```

The server binds to `0.0.0.0:8080` — accessible from any device on the same network.

Find your laptop's local IP:
```bash
ip addr | grep 'inet '        # Linux
ipconfig                      # Windows
ifconfig | grep 'inet '       # macOS
```

Then open `http://<laptop-ip>:8080` on your smartphone.

### Using the page

1. Lay the piece face up on a plain surface that contrasts with it, with space
   all around, and tap **Take photo**.
2. Check the yellow outline. If it misses part of the piece or takes in the
   table, move the **Looser / Tighter** slider.
3. Pick the methods to run and tap **Locate piece**. The table shows each
   method's cell, confidence and time, and the artwork overview outlines each
   method's cell in its colour.

## Benchmark

`argus_bench` measures each matcher on simulated piece photos cut from your artwork:

```bash
cd build
./argus_bench                       # config/, 40 samples, piece count from grid.json
./argus_bench config 40 1000 5000   # compare piece counts
```

## Grid configuration

`config/grid.json` supports full customization:

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

- `pieces` is optional: the approximate piece count. It tells the matchers how
  large a piece is in the artwork, making them faster and more accurate.

- `row_labels` / `col_labels` are optional; if omitted they are generated
  (rows A, B, …, Z, AA, …; columns 1 … N). If given, their lengths must match `rows` / `cols`.
- `artwork_path` is optional and relative to the config directory. If omitted,
  the config directory is searched for `artwork.jpg/.jpeg/.png/.bmp/.tiff`.

## Matching methods

| # | Method           | Algorithm                | Strengths                              |
|---|------------------|--------------------------|----------------------------------------|
| 1 | Template Match   | cv::matchTemplate        | Simple, fast, good with consistent lighting |
| 2 | ML (MobileNetV2) | ONNX embedding + cosine  | Semantic understanding, lighting-robust |
| 3 | Histogram Match  | HSV histogram + CORREL   | Rotation-invariant, very fast          |
| 4 | Keypoint (ORB)   | ORB + BF matcher + ratio | Robust to rotation and scale changes   |

All four methods are always built. The user chooses which methods to run for each
photo (one or several), and the results are shown as a table with coordinate,
confidence, and timing per selected method.

## Project structure

```
OracleOfEdges/
├── config/
│   ├── grid.json
│   ├── artwork.jpg        ← place your artwork here
│   └── mobilenetv2.onnx   ← ML model (included)
├── src/
│   ├── main.cpp
│   ├── GridConfig.hpp/.cpp
│   ├── Matcher.hpp
│   ├── WebHandler.hpp/.cpp
│   └── matchers/
│       ├── MatcherUtils.hpp/.cpp
│       ├── TemplateMatcher.hpp/.cpp
│       ├── MLMatcher.hpp/.cpp
│       ├── HistogramMatcher.hpp/.cpp
│       ├── KeypointMatcher.hpp/.cpp
│       └── OnnxGraph.hpp/.cpp
├── web/                   ← the page served at /
│   ├── index.html, style.css
│   ├── app.js             ← UI, upload, results
│   └── cutout.js          ← piece outline detection
├── tools/
│   └── bench.cpp          ← argus_bench
└── CMakeLists.txt
```
