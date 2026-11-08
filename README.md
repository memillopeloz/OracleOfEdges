# OracleOfEdges

A C++17 server, **Argus** (named for the hundred-eyed giant who sees everything), that accepts a photo of a puzzle piece (from a smartphone browser) 
and locates it within a reference artwork image using four different matching methods, plus a combination of the two best.

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

Argus reads the repo's `config/` and `web/` directly (their location is compiled
in), so edits there take effect on the next start.

## Run

```bash
build/argus                     # from the repo; works from any directory
# or with custom options:
build/argus --port 8080 --config-dir /path/to/config --web-dir /path/to/web
```

To type just `argus`, link it into a directory on your `PATH`:
```bash
ln -s "$PWD/build/argus" ~/.local/bin/argus
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
   table, move the **Looser / Tighter** slider, then touch up what's left by
   hand: pick **Add** or **Erase** and paint along the edge (**Undo** steps
   back, **Reset** returns to the automatic outline).
3. Pick the methods to run and tap **Locate piece**. The table shows each
   method's cell, confidence and time, and the artwork overview outlines each
   method's cell in its colour, with a dot where it puts the piece.
4. Tap a cell on the overview to zoom in: the cell is shown at full resolution
   with each method's estimate of the piece outline drawn in its colour, so you
   can compare the artwork there with the piece in your hand.

## Benchmark

`argus_bench` measures each matcher on simulated piece photos cut from your artwork:

```bash
cd build
./argus_bench                       # 40 samples, piece count from grid.json
./argus_bench 40 1000 5000          # compare piece counts
./argus_bench /other/config 40      # another config directory
```

To score real photos, put them in `samples/` named by their true cell
(`C5.jpg`, `C5_2.jpg`, …), cut them out with the page's own code, then score.
To add or replace sample photos straight from the phone, start the server with
`--samples-dir samples` (from the repo) and open `http://<laptop-ip>:8080/admin.html`
(a temporary admin page: pick the cell, take the photo, save).

```bash
node tools/cutsamples.mjs                     # samples/ → samples/cut/
build/argus_bench --real samples/cut
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
| 3 | Histogram Match  | Grey-level + texture histograms | Rotation-invariant, very fast   |
| 4 | Keypoint (ORB)   | ORB + BF matcher + ratio | Robust to rotation and scale changes   |
| 5 | Combined         | Keypoint, else Template  | Best overall on this puzzle            |

All five methods are always built. The user chooses which methods to run for each
photo (one or several), and the results are shown as a table with coordinate,
confidence, and timing per selected method.

New to these techniques? [How the matching methods work](#how-the-matching-methods-work)
at the end of this page explains each one in plain terms.

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
│       ├── CombinedMatcher.hpp/.cpp
│       └── OnnxGraph.hpp/.cpp
├── web/                   ← the page served at /
│   ├── index.html, style.css
│   ├── app.js             ← UI, upload, results
│   ├── photo.js           ← decode, preview, editable mask, cut-out PNG
│   ├── editor.js          ← hand touch-up of the outline
│   └── cutout.js          ← piece outline detection
├── tools/
│   ├── bench.cpp          ← argus_bench
│   ├── cutsamples.mjs     ← cut out samples/ like the page
│   ├── webtest.mjs        ← end-to-end page test
│   └── cdp.mjs            ← headless browser driver
└── CMakeLists.txt
```

## How the matching methods work

Argus has to answer one question: *where in the big picture does this small
piece belong?* Each method answers it differently, so they succeed and fail on
different pieces. That's why you can run several at once and compare.

Before any method runs, Argus prepares the photo the same way for all of them:
it keeps only the piece (using the outline you confirmed), turns it so its
straight edges are level, and trims the ragged border. A piece can still be
upside down or on its side at this point, so methods that care about direction
try all four quarter-turns.

### Template

**The idea:** slide the piece over the picture like a stencil and, at every
spot, measure how well the pattern of light and dark matches.

Think of holding a small transparent photo over a large poster and moving it
around until the shapes underneath line up. Template does exactly that, at
every position, in all four orientations. To keep it fast, it first works on a
shrunken copy of the picture, then double-checks its best few guesses at a
finer size.

- **Good at:** pieces with clear shapes and edges: architecture, frames, text.
- **Struggles with:** large plain areas (sky, walls), where many spots look the same.
- **Speed:** the slowest, a couple of seconds per piece.

### ML (machine learning)

**The idea:** let a neural network "describe" the piece and every part of the
picture, then find the part whose description is closest.

The network (MobileNetV2) was trained on millions of everyday photos to
recognise things. Argus doesn't use it to name objects; it uses the network's
internal summary of an image: a list of 1,280 numbers capturing textures,
shapes and colours. At startup Argus summarises every piece-sized region of
the picture; for each photo it summarises the piece and picks the most similar
region.

- **Good at:** recognising *kinds* of content (foliage, fabric, stone) even when
  the details are blurry.
- **Struggles with:** telling apart regions that contain the same kind of thing,
  which is common in a painting with lots of similar areas. It is the weakest
  method here.
- **Speed:** fast per photo (well under a second), but it makes the server take
  longer to start.

### Histogram

**The idea:** compare the *character* of the piece's surface (how its light
and dark are spread, how busy or smooth it is), ignoring where things are.

Colour is left out on purpose: this puzzle is printed in near-monochrome sepia,
so blue sky and green leaves both come out brown. Instead Argus tallies three
things for the piece and for every piece-sized region of the picture: how the
brightness is spread (mostly mid-tones? a few bright highlights?), what the
tiny local patterns look like (flat, edges, corners, specks), and how much
detail there is compared with the overall contrast. All three ignore rotation
and survive the print being lighter, darker or flatter than the screen. The
region whose tallies look most like the piece's is the answer.

- **Good at:** being very fast, and telling smooth areas (sky, walls) from busy
  ones (foliage, crowds).
- **Struggles with:** almost everything else. Many regions of the picture have
  the same character, so it is rarely right on its own. Its confidence says
  how far ahead the winner is of the best region in any other cell.
- **Speed:** the fastest, a few thousandths of a second.

### Keypoint

**The idea:** find small distinctive spots (corners, sharp details) on the
piece, find the same kind of spots in the picture, and look for a group that
matches *with the same arrangement*.

It works like recognising a constellation: one star tells you little, but five
stars in the right pattern are unmistakable. Argus finds such spots
("keypoints") all over the picture at startup, matches the piece's spots to
them, and then checks that the matched spots sit in the same geometric
arrangement, allowing for rotation and size. Spots that agree on one location
win.

- **Good at:** detailed areas, and its confidence is the most trustworthy: when
  it is wrong it usually says so with a confidence near zero.
- **Struggles with:** soft or plain pieces that have few sharp details; then it
  may find nothing and report no result.
- **Speed:** a few seconds per piece.

### Combined

**The idea:** ask Keypoint first, and fall back to Template when Keypoint isn't
sure.

Keypoint is right less often than it would like, but it knows when it is: a
clearly non-zero confidence is usually a hit. Template always gives an answer
but can't tell a good one from a bad one. They tend to get *different* pieces
right, so taking Keypoint's answer when it is confident and Template's
otherwise gets more pieces right than either alone.

- **Good at:** the best hit rate overall on this puzzle.
- **Confidence:** Keypoint's when it answered; a fallback to Template always
  shows a low confidence (under 5%), since Template's guess is less certain.
- **Speed:** runs both side by side, so as slow as the slower of the two.

### Reading the results

- **Cell** is the grid square the method thinks the piece belongs in.
- **Confidence** (0–100%) is the method's own sense of how sure it is. Each
  method measures it differently, so compare confidence within a method, not
  across methods. A 30% from Keypoint can mean more than a 95% from Histogram.
- **When methods agree**, the answer is much more likely to be right. When they
  disagree, trust Keypoint first if its confidence is clearly above zero —
  which is exactly what Combined does for you.

