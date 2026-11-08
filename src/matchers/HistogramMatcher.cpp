#include "HistogramMatcher.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr double TILE_STRIDE = 0.5;          // piece count known: tiles of one piece, half-piece steps
const std::vector<int> TILE_DIVISIONS = {2, 3, 4, 6}; // unknown: tiles of cell/2 … cell/6

constexpr int    WORK_PX       = 64;    // tiles and piece are resized to about this side
constexpr double SMOOTH_SIGMA  = 0.7;   // pre-blur at work size, against photo noise
constexpr double RIM_ERODE     = 0.06;  // piece mask eroded by this × side (table/shadow rim)
constexpr int    GREY_BINS     = 16;    // brightness distribution, in standard deviations…
constexpr double Z_RANGE       = 2.5;   // …from −Z_RANGE to +Z_RANGE
constexpr int    LBP_BINS      = 10;    // rotation-invariant uniform LBP: 0–8 bright neighbours, 9 = non-uniform
constexpr int    LBP_RADII[]   = {1, 2};
constexpr float  LBP_TOLERANCE = 2.0f;  // grey levels a neighbour must exceed the centre by
constexpr int    GRAD_BINS     = 8;     // gradient magnitude / contrast…
constexpr double GRAD_RANGE    = 1.0;   // …over [0, GRAD_RANGE], last bin catches the rest
constexpr double CONF_GAIN     = 4.0;   // relative margin over the best other cell → confidence

// Blocks of the descriptor; each is a separately normalized histogram
constexpr std::array<int, 4> BLOCKS = {GREY_BINS, LBP_BINS, LBP_BINS, GRAD_BINS};
constexpr int DESC_LEN = GREY_BINS + 2 * LBP_BINS + GRAD_BINS;

// Offsets of the 8 neighbours on the square ring at Chebyshev distance r, in
// circular order (so LBP transitions can be counted)
// In:     r — ring radius, px (≥ 1)
// Out:    the 8 offsets
// Errors: none
std::array<cv::Point, 8> ring(int r) {
    return {cv::Point(r, 0), cv::Point(r, r), cv::Point(0, r), cv::Point(-r, r),
            cv::Point(-r, 0), cv::Point(-r, -r), cv::Point(0, -r), cv::Point(r, -r)};
}

// Rotation-invariant uniform LBP code of one pixel: a neighbour counts as "on"
// when brighter than the centre by LBP_TOLERANCE; patterns with at most two
// on/off transitions around the ring are coded by their number of "on" bits.
// In:     g       — smoothed grey image, CV_32F
//         p       — pixel, at least the ring radius away from every border
//         offsets — the ring (see ring())
// Out:    0–8 for uniform patterns, 9 for the rest
// Errors: none; p too close to a border reads out of bounds (callers check)
int lbpCode(const cv::Mat& g, cv::Point p, const std::array<cv::Point, 8>& offsets) {
    const float centre = g.at<float>(p) + LBP_TOLERANCE;
    int ones = 0, transitions = 0;
    bool first = false, prev = false;
    for (size_t k = 0; k < offsets.size(); ++k) {
        const bool bit = g.at<float>(p + offsets[k]) >= centre;
        ones += bit;
        if (k == 0)
            first = bit;
        else
            transitions += bit != prev;
        prev = bit;
    }
    transitions += prev != first;
    return transitions <= 2 ? ones : LBP_BINS - 1;
}

// Descriptor of an image region: the brightness distribution in standard
// deviations from the mean, LBP texture codes at radii 1 and 2, and gradient
// magnitude relative to contrast, concatenated as BLOCKS.
// In:     grey8 — grey image, CV_8U, at work size
//         mask  — CV_8U, same size, non-zero on the pixels to describe; an
//                 empty Mat means the whole image
// Out:    DESC_LEN floats, each block L1-normalized (all zero if no pixels)
// Errors: none for non-empty input
std::vector<float> describe(const cv::Mat& grey8, const cv::Mat& mask) {
    cv::Mat g, valid;
    grey8.convertTo(g, CV_32F);
    cv::GaussianBlur(g, g, cv::Size(0, 0), SMOOTH_SIGMA);
    valid = mask.empty() ? cv::Mat(g.size(), CV_8U, cv::Scalar(255)) : mask;

    cv::Scalar mean, stddev;
    cv::meanStdDev(g, mean, stddev, valid);
    const double sd = std::max(stddev[0], 1.0);

    cv::Mat gx, gy, mag;
    cv::Sobel(g, gx, CV_32F, 1, 0, 3, 1.0 / 8);
    cv::Sobel(g, gy, CV_32F, 0, 1, 3, 1.0 / 8);
    cv::magnitude(gx, gy, mag);

    std::vector<float> desc(DESC_LEN, 0.0f);
    float* greyH = desc.data();
    float* lbpH[2] = {greyH + GREY_BINS, greyH + GREY_BINS + LBP_BINS};
    float* gradH = greyH + GREY_BINS + 2 * LBP_BINS;
    const std::array<cv::Point, 8> rings[2] = {ring(LBP_RADII[0]), ring(LBP_RADII[1])};
    const int border = LBP_RADII[1];

    for (int y = 0; y < g.rows; ++y) {
        for (int x = 0; x < g.cols; ++x) {
            if (valid.at<uchar>(y, x)) {
                const double z = (g.at<float>(y, x) - mean[0]) / sd;
                const int zb = static_cast<int>((z + Z_RANGE) / (2 * Z_RANGE) * GREY_BINS);
                greyH[std::clamp(zb, 0, GREY_BINS - 1)] += 1.0f;
                const int gb = static_cast<int>(mag.at<float>(y, x) / sd / GRAD_RANGE * GRAD_BINS);
                gradH[std::clamp(gb, 0, GRAD_BINS - 1)] += 1.0f;
                if (x >= border && y >= border && x < g.cols - border && y < g.rows - border)
                    for (int r = 0; r < 2; ++r)
                        lbpH[r][lbpCode(g, cv::Point(x, y), rings[r])] += 1.0f;
            }
        }
    }

    size_t offset = 0;
    for (const int len : BLOCKS) {
        float sum = 0.0f;
        for (int i = 0; i < len; ++i)
            sum += desc[offset + i];
        for (int i = 0; i < len && sum > 0.0f; ++i)
            desc[offset + i] /= sum;
        offset += len;
    }
    return desc;
}

// Mean Hellinger distance over the descriptor's blocks
// In:     a, b — descriptors from describe()
// Out:    0 = identical … 1 = disjoint (an all-zero block counts as disjoint)
// Errors: none
double distance(const std::vector<float>& a, const std::vector<float>& b) {
    double total = 0.0;
    size_t offset = 0;
    for (const int len : BLOCKS) {
        double bc = 0.0;
        for (int i = 0; i < len; ++i)
            bc += std::sqrt(static_cast<double>(a[offset + i]) * b[offset + i]);
        total += std::sqrt(std::max(0.0, 1.0 - bc));
        offset += len;
    }
    return total / BLOCKS.size();
}

} // namespace

void HistogramMatcher::prepare(const cv::Mat& artwork, const GridConfig& grid) {
    grid_  = grid;
    const double side = grid.pieceSide(artwork.size());
    tiles_ = side > 0.0
           ? matcher_utils::gridTiles(grid, artwork.size(), cvRound(side),
                                      std::max(1, cvRound(side * TILE_STRIDE)))
           : matcher_utils::cellTiles(grid, artwork.size(), TILE_DIVISIONS);

    cv::Mat grey;
    cv::cvtColor(artwork, grey, cv::COLOR_BGR2GRAY);
    tileDescs_.assign(tiles_.size(), {});
    cv::parallel_for_(cv::Range(0, static_cast<int>(tiles_.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            cv::Mat small;
            cv::resize(grey(tiles_[i].rect), small, cv::Size(WORK_PX, WORK_PX), 0, 0, cv::INTER_AREA);
            tileDescs_[i] = describe(small, cv::Mat());
        }
    });
}

MatchResult HistogramMatcher::match(const Piece& piece) const {
    const auto start = std::chrono::steady_clock::now();

    // Piece at work size: its side (√area) becomes WORK_PX, like a tile's
    const double scale = std::min(1.0, WORK_PX / std::sqrt(std::max(piece.area, 1.0)));
    cv::Mat grey, small, mask;
    cv::cvtColor(piece.bgr, grey, cv::COLOR_BGR2GRAY);
    cv::resize(grey, small, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::resize(piece.mask, mask, small.size(), 0, 0, cv::INTER_NEAREST);
    const int erode = cvRound(RIM_ERODE * WORK_PX);
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * erode + 1, 2 * erode + 1)));
    const std::vector<float> desc = describe(small, mask);

    std::vector<double> dists(tiles_.size());
    size_t bestIdx = 0;
    for (size_t i = 0; i < tiles_.size(); ++i) {
        dists[i] = distance(desc, tileDescs_[i]);
        if (dists[i] < dists[bestIdx])
            bestIdx = i;
    }

    // Confidence: how far ahead the best tile is of the best tile in any other cell
    const matcher_utils::Tile& tile = tiles_[bestIdx];
    double otherBest = std::numeric_limits<double>::max();
    for (size_t i = 0; i < tiles_.size(); ++i)
        if (tiles_[i].row != tile.row || tiles_[i].col != tile.col)
            otherBest = std::min(otherBest, dists[i]);
    const double margin = otherBest < std::numeric_limits<double>::max()
                        ? (otherBest - dists[bestIdx]) / std::max(otherBest, 1e-6)
                        : 1.0;
    const float confidence = static_cast<float>(std::clamp(margin * CONF_GAIN, 0.0, 1.0));
    MatchResult result = matcher_utils::makeResult(name(), grid_, tile.row, tile.col, confidence, start);
    result.outline = matcher_utils::rectOutline(tile.rect);
    return result;
}
