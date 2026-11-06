// argus_bench — measures each matcher's accuracy and speed on simulated pieces.
//
// Cuts random jigsaw-shaped pieces from the artwork and degrades them the way a
// phone photo would (rotation, lighting, blur, noise, JPEG), then delivers them
// as the client will: a BGRA cut-out cropped to the piece, alpha = outline.
// A result counts as correct when it names the cell holding the piece's centre.
//
// Usage: argus_bench [config-dir] [samples] [pieces...]
//   config-dir  default: config
//   samples     simulated pieces per run (default: 40)
//   pieces      piece counts to test (default: the value in grid.json; 0 = unknown)
//
//        argus_bench <config-dir> --real <cut-dir>
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

constexpr int    PHOTO_PX      = 700;   // simulated phone resolution of the piece region
constexpr double KNOB_RADIUS   = 0.17;  // jigsaw knob radius, fraction of body side
constexpr unsigned RANDOM_SEED = 7;     // fixed, so runs are comparable

struct Tally {
    int    hits      = 0;
    long   totalMs   = 0;
    double confHit   = 0.0;
    double confMiss  = 0.0;
};

// Simulated client cut-out (BGRA) of a jigsaw piece with body side `side` centred at c
cv::Mat simulateCutout(const cv::Mat& art, cv::Point2f c, double side, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const int half = static_cast<int>(side);
    const cv::Rect region = cv::Rect(static_cast<int>(c.x) - half, static_cast<int>(c.y) - half,
                                     2 * half, 2 * half) & cv::Rect(0, 0, art.cols, art.rows);
    const double up = static_cast<double>(PHOTO_PX) / region.width;
    cv::Mat img;
    cv::resize(art(region), img, cv::Size(), up, up, cv::INTER_CUBIC);

    // Square body with a knob or a blank on each side
    const cv::Point2f centre = (c - cv::Point2f(region.tl())) * static_cast<float>(up);
    const double body = side * up;
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
    cv::rectangle(mask, cv::Rect2f(centre.x - body / 2, centre.y - body / 2, body, body),
                  cv::Scalar(255), cv::FILLED);
    for (const cv::Point2f dir : {cv::Point2f(0, -1), cv::Point2f(1, 0), cv::Point2f(0, 1), cv::Point2f(-1, 0)}) {
        const bool knob = U(rng) < 0.5;
        cv::circle(mask, centre + dir * static_cast<float>(body * (knob ? 0.62 : 0.38)),
                   static_cast<int>(body * KNOB_RADIUS), cv::Scalar(knob ? 255 : 0), cv::FILLED);
    }

    // Photo degradations
    const cv::Mat R = cv::getRotationMatrix2D(centre, U(rng) * 360.0, 1.0);
    cv::warpAffine(img,  img,  R, img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::warpAffine(mask, mask, R, mask.size(), cv::INTER_NEAREST);
    img.convertTo(img, -1, 0.8 + 0.35 * U(rng), U(rng) * 30.0 - 15.0);
    cv::GaussianBlur(img, img, cv::Size(0, 0), 1.2);
    cv::Mat noise(img.size(), CV_16SC3), wide;
    cv::randn(noise, 0, 4);
    img.convertTo(wide, CV_16SC3);
    wide += noise;
    wide.convertTo(img, CV_8UC3);
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", img, jpeg, {cv::IMWRITE_JPEG_QUALITY, 85});
    img = cv::imdecode(jpeg, cv::IMREAD_COLOR);

    cv::Mat bgra;
    cv::cvtColor(img, bgra, cv::COLOR_BGR2BGRA);
    cv::insertChannel(mask, bgra, 3);
    return bgra(cv::boundingRect(mask)).clone();
}

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
        for (const MatchResult& r : handler.locate(simulateCutout(art, c, side, rng), names)) {
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

// Cell (col, row) whose label matches the file name up to the first '_'; (-1, -1) if none
cv::Point cellFromName(const GridConfig& grid, const std::string& stem) {
    const std::string label = stem.substr(0, stem.find('_'));
    cv::Point cell(-1, -1);
    for (const GridCell& c : grid.allCells())
        if (c.label == label)
            cell = cv::Point(c.col, c.row);
    return cell;
}

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

int main(int argc, char* argv[]) {
    int status = 0;
    try {
        const std::string configDir = argc > 1 ? argv[1] : "config";
        const bool real = argc > 3 && std::string(argv[2]) == "--real";
        const int samples = argc > 2 && !real ? std::stoi(argv[2]) : 40;

        const GridConfig grid = GridConfig::fromFile((fs::path(configDir) / "grid.json").string());
        const cv::Mat art = cv::imread(grid.artworkPath, cv::IMREAD_COLOR);
        const std::string model = (fs::path(configDir) / "mobilenetv2.onnx").string();
        const std::string modelPath = fs::exists(model) ? model : "";

        std::vector<int> pieceCounts;
        for (int i = 3; i < argc && !real; ++i)
            pieceCounts.push_back(std::stoi(argv[i]));
        if (pieceCounts.empty())
            pieceCounts.push_back(grid.pieces);

        if (art.empty()) {
            std::fprintf(stderr, "[error] Cannot read artwork: %s\n", grid.artworkPath.c_str());
            status = 2;
        } else if (real) {
            runReal(art, grid, modelPath, argv[3]);
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
