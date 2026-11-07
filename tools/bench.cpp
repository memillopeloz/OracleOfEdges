// argus_bench — measures each matcher's accuracy and speed on simulated pieces.
//
// Cuts random jigsaw-shaped pieces from the artwork and degrades them the way a
// phone photo of the physical print does (sepia toning, print tone shifts, blur,
// grain, uneven light, table/shadow caught in the outline, rotation, JPEG), then
// delivers them as the client will: a BGRA cut-out cropped to the piece, alpha =
// outline. Tuned so its accuracies track `--real` on the sample photos.
// A result counts as correct when it names the cell holding the piece's centre.
//
// Usage: argus_bench [config-dir] [samples] [pieces...]
//   config-dir  default: the repo's config/ (omit it when the next argument is a
//               number or --real)
//   samples     simulated pieces per run (default: 40)
//   pieces      piece counts to test (default: the value in grid.json; 0 = unknown)
//
//        argus_bench [config-dir] --real <cut-dir>
//   Scores real cut-outs (from tools/cutsamples.mjs) instead. Each file is named
//   by its true cell, optionally with a suffix: C5.png, C5_2.png. Prints one
//   line per photo and the same summary as the simulated run.
#include "GridConfig.hpp"
#include "WebHandler.hpp"
#include "matchers/MatcherUtils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Simulation parameters, measured on real phone photos of this puzzle (2026-11-06)
constexpr double PHOTO_SCALE       = 6.0;   // photo px per artwork px (the client then caps at 1024 px)
constexpr double BLUR_ART_PX       = 0.5;   // Gaussian blur sigma, in artwork px
constexpr double COLOUR_KEPT       = 0.2;   // fraction of the original colour surviving the sepia print
constexpr double CHANNEL_JITTER    = 0.25;  // per-piece spread of the print's channel weights
constexpr double DENSITY_VARIATION = 0.15;  // low-frequency ink density variation across a piece
constexpr double RIM_FRACTION      = 0.045; // background/shadow band inside the outline, × piece side
constexpr unsigned RANDOM_SEED     = 7;     // fixed, so runs are comparable

struct Tally {
    int    hits      = 0;
    long   totalMs   = 0;
    double confHit   = 0.0;
    double confMiss  = 0.0;
};

// Jigsaw outline: a w × h body centred at c with a knob or a blank on each side,
// each slightly off-centre and of varying size, as on the real (irregular) pieces
// In:     size — mask size, px
//         c    — body centre, px
//         w, h — body width and height, px
//         rng  — random source (advanced)
// Out:    CV_8U mask, 255 inside the piece
// Errors: none
cv::Mat jigsawMask(cv::Size size, cv::Point2f c, double w, double h, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    cv::rectangle(mask, cv::Rect2f(c.x - w / 2, c.y - h / 2, w, h), cv::Scalar(255), cv::FILLED);
    const double small = std::min(w, h);
    const cv::Point2f dirs[4] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    for (const cv::Point2f dir : dirs) {
        const bool knob = U(rng) < 0.5;
        const double halfExtent = (dir.x != 0 ? w : h) / 2;
        const cv::Point2f along(dir.y, dir.x);
        const double radius = small * (0.16 + 0.08 * U(rng));
        const cv::Point2f base = c + along * static_cast<float>((U(rng) - 0.5) * 0.3 * (dir.x != 0 ? h : w));
        const cv::Point2f knobCentre = base + dir * static_cast<float>(halfExtent + (knob ? 0.9 : -0.9) * radius);
        cv::circle(mask, knobCentre, static_cast<int>(radius), cv::Scalar(knob ? 255 : 0), cv::FILLED);
        // neck joining a knob to the body
        const cv::Point2f n1 = base + along * static_cast<float>(radius * 0.55);
        const cv::Point2f n2 = base - along * static_cast<float>(radius * 0.55);
        const cv::Point2f reach = dir * static_cast<float>(knob ? 1.0 : -1.0) * static_cast<float>(radius);
        const std::vector<cv::Point> neck = {n1 - reach * 0.2f, n1 + reach, n2 + reach, n2 - reach * 0.2f};
        if (knob)
            cv::fillConvexPoly(mask, neck, cv::Scalar(255));
    }
    return mask;
}

// Simulated client cut-out of a jigsaw piece, degraded like a phone photo of
// the sepia print (see the file header for the list of effects)
// In:     art  — full artwork, CV_8UC3 BGR
//         c    — piece centre in artwork px, at least `side` from every edge
//         side — mean body side in artwork px
//         rng  — random source (advanced)
// Out:    BGRA cut-out cropped to the outline, alpha = outline, ≤ 1024 px
// Errors: none for valid input; a centre too close to the edge may clip the
//         piece (cv::Exception if the region becomes empty)
cv::Mat simulateCutout(const cv::Mat& art, cv::Point2f c, double side, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);

    // Rectangular body: some pieces wider, some taller, some about square
    const double aspect = std::exp((U(rng) - 0.5) * 0.45);
    const double w = side * std::sqrt(aspect), h = side / std::sqrt(aspect);

    const int half = static_cast<int>(side);
    const cv::Rect region = cv::Rect(static_cast<int>(c.x) - half, static_cast<int>(c.y) - half,
                                     2 * half, 2 * half) & cv::Rect(0, 0, art.cols, art.rows);
    cv::Mat img;
    cv::resize(art(region), img, cv::Size(), PHOTO_SCALE, PHOTO_SCALE, cv::INTER_CUBIC);
    const cv::Point2f centre = (c - cv::Point2f(region.tl())) * static_cast<float>(PHOTO_SCALE);
    cv::Mat mask = jigsawMask(img.size(), centre, w * PHOTO_SCALE, h * PHOTO_SCALE, rng);

    // Sepia print: mostly grey, warm cast (red ≈ grey, green ≈ 0.82, blue ≈ 0.66)
    cv::Mat grey, grey3, colour;
    // The print's brightness doesn't follow the file's grey: each region's inks
    // weigh the channels differently (blue sky prints brown, white marble tan)
    std::normal_distribution<double> N(0.0, 1.0);
    double wb = std::max(0.0, 0.114 + CHANNEL_JITTER * N(rng));
    double wg = std::max(0.0, 0.587 + CHANNEL_JITTER * N(rng));
    double wr = std::max(0.0, 0.299 + CHANNEL_JITTER * N(rng));
    const double wsum = std::max(1e-3, wb + wg + wr);
    cv::transform(img, grey, cv::Matx13f(static_cast<float>(wb / wsum), static_cast<float>(wg / wsum),
                                         static_cast<float>(wr / wsum)));
    cv::cvtColor(grey, grey3, cv::COLOR_GRAY2BGR);
    cv::addWeighted(img, COLOUR_KEPT, grey3, 1.0 - COLOUR_KEPT, 0.0, colour);
    colour.convertTo(colour, CV_32FC3, 1.0 / 255);
    cv::pow(colour, std::exp((U(rng) - 0.5) * 0.7), colour);   // print tone curve, gamma ≈ 0.7–1.4
    colour *= 255.0;
    const double gain = 0.85 + 0.3 * U(rng), contrast = 1.1 + 0.3 * U(rng);
    const cv::Scalar mean = cv::mean(colour);
    colour = (colour - mean) * contrast + mean;
    cv::multiply(colour, cv::Scalar(gain * (0.62 + 0.08 * U(rng)), gain * (0.78 + 0.08 * U(rng)), gain), colour);

    // Uneven lighting across the piece
    cv::Mat light(img.size(), CV_32F);
    const double gx = (U(rng) - 0.5) * 0.25 / img.cols, gy = (U(rng) - 0.5) * 0.25 / img.rows;
    for (int y = 0; y < light.rows; ++y)
        for (int x = 0; x < light.cols; ++x)
            light.at<float>(y, x) = static_cast<float>(1.0 + gx * (x - centre.x) + gy * (y - centre.y));
    // and ink density drifting across the print (blotches of about a third of a piece)
    cv::Mat density(4, 4, CV_32F);
    cv::randn(density, 0.0, DENSITY_VARIATION);
    cv::resize(density, density, light.size(), 0, 0, cv::INTER_CUBIC);
    light += density;
    cv::Mat light3;
    cv::merge(std::vector<cv::Mat>{light, light, light}, light3);
    cv::multiply(colour, light3, colour);

    // Band of table and shadow caught inside the outline, thicker on one side
    const double rim = RIM_FRACTION * side * PHOTO_SCALE * (0.6 + 0.8 * U(rng));
    const double shadowAngle = U(rng) * 2 * CV_PI;
    cv::Mat outline, shifted;
    const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, rim * std::cos(shadowAngle), 0, 1, rim * std::sin(shadowAngle));
    cv::warpAffine(mask, shifted, shift, mask.size(), cv::INTER_NEAREST);
    const int grow = std::max(1, static_cast<int>(rim * 0.5));
    cv::dilate(mask | shifted, outline, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * grow + 1, 2 * grow + 1)));
    // ragged edge, as the client's outline follows the photo's noise
    cv::Mat edgeNoise(outline.size(), CV_32F), soft;
    cv::randn(edgeNoise, 0, 1);
    cv::GaussianBlur(edgeNoise, edgeNoise, cv::Size(0, 0), 0.6 * PHOTO_SCALE);
    cv::normalize(edgeNoise, edgeNoise, -1.0, 1.0, cv::NORM_MINMAX);
    outline.convertTo(soft, CV_32F, 1.0 / 255);
    cv::GaussianBlur(soft, soft, cv::Size(0, 0), 0.8 * PHOTO_SCALE);
    outline = (soft + 0.35 * edgeNoise) > 0.5;
    const cv::Scalar table(30 + 15 * U(rng), 45 + 15 * U(rng), 65 + 20 * U(rng));
    colour.setTo(table, outline & ~mask);

    // Optics: blur, grain, sensor noise
    cv::GaussianBlur(colour, colour, cv::Size(0, 0), BLUR_ART_PX * PHOTO_SCALE * (0.8 + 0.4 * U(rng)));
    // Print and paper texture at several scales: {blur sigma in photo px, strength};
    // real photos look smooth, a few grey levels of grain at most
    const double TEXTURE[3][2] = {{0.6, 8.0}, {2.0, 10.0}, {5.0, 60.0}};
    for (const auto& t : TEXTURE) {
        cv::Mat grain(img.size(), CV_32F), grain3;
        cv::randn(grain, 0, t[1]);
        cv::GaussianBlur(grain, grain, cv::Size(0, 0), t[0]);
        cv::merge(std::vector<cv::Mat>{grain * 0.75, grain * 0.88, grain}, grain3);
        colour += grain3;
    }
    cv::Mat noise(img.size(), CV_32FC3);
    cv::randn(noise, 0, 3);
    colour += noise;
    colour.convertTo(img, CV_8UC3);

    // Hand-held: any rotation, camera a few degrees off overhead (foreshortening); then JPEG
    const cv::Mat R = cv::getRotationMatrix2D(centre, U(rng) * 360.0, 1.0);
    cv::Mat P = cv::Mat::eye(3, 3, CV_64F);
    R.copyTo(P.rowRange(0, 2));
    const double tilt = 0.09 * U(rng) / img.cols, tiltAngle = U(rng) * 2 * CV_PI;
    const cv::Mat toCentre = (cv::Mat_<double>(3, 3) << 1, 0, -centre.x, 0, 1, -centre.y, 0, 0, 1);
    const cv::Mat back     = (cv::Mat_<double>(3, 3) << 1, 0,  centre.x, 0, 1,  centre.y, 0, 0, 1);
    const cv::Mat keystone = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0,
                              tilt * std::cos(tiltAngle), tilt * std::sin(tiltAngle), 1);
    P = back * keystone * toCentre * P;
    cv::warpPerspective(img,     img,     P, img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::warpPerspective(outline, outline, P, outline.size(), cv::INTER_NEAREST);
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", img, jpeg, {cv::IMWRITE_JPEG_QUALITY, 85});
    img = cv::imdecode(jpeg, cv::IMREAD_COLOR);

    cv::Mat bgra;
    cv::cvtColor(img, bgra, cv::COLOR_BGR2BGRA);
    cv::insertChannel(outline, bgra, 3);
    cv::Mat cut = bgra(cv::boundingRect(outline)).clone();
    // The client sends at most 1024 px
    return matcher_utils::limitSize(cut, 1024);
}

// Prints one row per matcher: accuracy, mean time, mean confidence on hits and
// on misses
// In:     names   — matchers, in display order
//         tallies — per-matcher totals (missing names read as zero)
//         samples — number of pieces scored (> 0)
// Out:    none (stdout)
// Errors: none
void printTallies(const std::vector<std::string>& names, std::map<std::string, Tally>& tallies,
                  int samples) {
    std::printf("  %-10s %8s %10s %14s\n", "matcher", "accuracy", "avg time", "conf hit/miss");
    for (const std::string& n : names) {
        const Tally& t = tallies[n];
        const int misses = samples - t.hits;
        std::printf("  %-10s %7d%% %7ld ms %8.2f / %.2f\n", n.c_str(), 100 * t.hits / samples,
                    t.totalMs / samples, t.hits ? t.confHit / t.hits : 0.0,
                    misses ? t.confMiss / misses : 0.0);
    }
    std::fflush(stdout);
}

// Scores every matcher on simulated pieces at random positions (fixed seed) and
// prints the tallies
// In:     art       — full artwork, CV_8UC3 BGR
//         grid      — cell layout; grid.pieces sets the simulated piece size
//                     (0 = simulate a 1000-piece puzzle, matchers search blind)
//         modelPath — ONNX model; empty = ML unavailable
//         samples   — number of simulated pieces (> 0)
// Out:    none (stdout)
// Errors: exceptions from WebHandler construction propagate
void runBench(const cv::Mat& art, GridConfig grid, const std::string& modelPath, int samples) {
    const auto t0 = std::chrono::steady_clock::now();
    const WebHandler handler(art, grid, modelPath);
    const long prepareMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - t0).count();

    // With an unknown piece count, simulate a 1000-piece puzzle
    const double side = std::sqrt(static_cast<double>(art.size().area()) / (grid.pieces > 0 ? grid.pieces : 1000));
    const std::vector<std::string> names = handler.matcherNames();
    std::map<std::string, Tally> tallies;
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_real_distribution<float> X(side, art.cols - side), Y(side, art.rows - side);

    for (int i = 0; i < samples; ++i) {
        const cv::Point2f c(X(rng), Y(rng));
        const cv::Point truth = matcher_utils::cellAt(grid, art.size(), c);
        const cv::Mat cut = simulateCutout(art, c, side, rng);
        for (const MatchResult& r : handler.locate(cut, names)) {
            Tally& t = tallies[r.matcherName];
            const bool hit = r.row == truth.y && r.col == truth.x;
            t.hits    += hit;
            t.totalMs += r.timeMs;
            (hit ? t.confHit : t.confMiss) += r.confidence;
        }
    }

    std::printf("\npieces=%d (piece ≈ %.0f px in artwork), %d samples, prepare %ld ms\n",
                grid.pieces, side, samples, prepareMs);
    printTallies(names, tallies, samples);
}

// True cell of a sample photo, from its file name ("C5" or "C5_2" → C5)
// In:     grid — cell layout
//         stem — file name without extension
// Out:    (col, row) of the matching cell; (-1, -1) if the name isn't a cell label
// Errors: none
cv::Point cellFromName(const GridConfig& grid, const std::string& stem) {
    const std::string label = stem.substr(0, stem.find('_'));
    cv::Point cell(-1, -1);
    for (const GridCell& c : grid.allCells())
        if (c.label == label)
            cell = cv::Point(c.col, c.row);
    return cell;
}

// Scores every matcher on real cut-outs (from tools/cutsamples.mjs) named by
// their true cell, printing each photo's answers (+ = correct) and the tallies
// In:     art       — full artwork, CV_8UC3 BGR
//         grid      — cell layout
//         modelPath — ONNX model; empty = ML unavailable
//         cutDir    — directory of .png cut-outs
// Out:    none (stdout)
// Errors: files not named after a cell, or unreadable, are listed as skipped;
//         a missing cutDir throws std::filesystem::filesystem_error
void runReal(const cv::Mat& art, const GridConfig& grid, const std::string& modelPath,
             const std::string& cutDir) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(cutDir))
        if (entry.is_regular_file() && entry.path().extension() == ".png")
            files.push_back(entry.path());
    std::sort(files.begin(), files.end());

    const auto t0 = std::chrono::steady_clock::now();
    const WebHandler handler(art, grid, modelPath);
    const long prepareMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - t0).count();
    const std::vector<std::string> names = handler.matcherNames();
    std::map<std::string, Tally> tallies;

    std::printf("\n  %-8s %-5s", "photo", "truth");
    for (const std::string& n : names)
        std::printf(" %-10s", n.c_str());
    std::printf("\n");

    int scored = 0;
    for (const fs::path& file : files) {
        const cv::Point truth = cellFromName(grid, file.stem().string());
        const cv::Mat cutout = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
        if (truth.x < 0 || cutout.empty()) {
            std::printf("  %-8s skipped (%s)\n", file.stem().string().c_str(),
                        truth.x < 0 ? "name is not a cell label" : "unreadable");
        } else {
            ++scored;
            std::printf("  %-8s %-5s", file.stem().string().c_str(),
                        grid.coordLabel(truth.y, truth.x).c_str());
            for (const MatchResult& r : handler.locate(cutout, names)) {
                Tally& t = tallies[r.matcherName];
                const bool hit = r.row == truth.y && r.col == truth.x;
                t.hits    += hit;
                t.totalMs += r.timeMs;
                (hit ? t.confHit : t.confMiss) += r.confidence;
                const std::string answer = r.error.empty() ? r.coord : "-";
                std::printf(" %s %-8s", hit ? "+" : " ", answer.c_str());
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    std::printf("\nreal cut-outs from %s: %d scored, prepare %ld ms\n", cutDir.c_str(), scored, prepareMs);
    if (scored > 0)
        printTallies(names, tallies, scored);
}

} // namespace

// Entry point; see the usage in the file header
// In:     argv — [config-dir] [samples [pieces...] | --real <cut-dir>]; config-dir
//                 is taken as given unless argv[1] is a number or --real
// Out:    0 on success
// Errors: 1 on any exception (bad number, unreadable grid.json, missing cut
//         directory), 2 if the artwork can't be read; message on stderr
int main(int argc, char* argv[]) {
    int status = 0;
    try {
        // The config directory is optional: a number or --real starts the rest
        const bool hasDir = argc > 1 && std::string(argv[1]) != "--real"
                         && std::string(argv[1]).find_first_not_of("0123456789") != std::string::npos;
        const int first = hasDir ? 2 : 1;   // index of the first argument after config-dir
        const std::string configDir = hasDir ? argv[1] : std::string(ARGUS_SOURCE_DIR) + "/config";
        const bool real = argc > first + 1 && std::string(argv[first]) == "--real";
        const int samples = argc > first && !real ? std::stoi(argv[first]) : 40;

        const GridConfig grid = GridConfig::fromFile((fs::path(configDir) / "grid.json").string());
        const cv::Mat art = cv::imread(grid.artworkPath, cv::IMREAD_COLOR);
        const std::string model = (fs::path(configDir) / "mobilenetv2.onnx").string();
        const std::string modelPath = fs::exists(model) ? model : "";

        std::vector<int> pieceCounts;
        for (int i = first + 1; i < argc && !real; ++i)
            pieceCounts.push_back(std::stoi(argv[i]));
        if (pieceCounts.empty())
            pieceCounts.push_back(grid.pieces);

        if (art.empty()) {
            std::fprintf(stderr, "[error] Cannot read artwork: %s\n", grid.artworkPath.c_str());
            status = 2;
        } else if (real) {
            runReal(art, grid, modelPath, argv[first + 1]);
        } else {
            for (int pieces : pieceCounts) {
                GridConfig variant = grid;
                variant.pieces = pieces;
                runBench(art, variant, modelPath, samples);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        status = 1;
    }
    return status;
}
