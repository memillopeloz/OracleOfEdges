#include "TemplateMatcher.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "MatcherUtils.hpp"

namespace {

// Piece count known: search around the expected scale, with the artwork reduced
// so a piece spans about TARGET_PIECE_PX.
constexpr double TARGET_PIECE_PX = 32.0;
// Factors above 1 dominate: √area includes knobs and the photo's rim, so the
// expected scale comes out too small.
const std::vector<double> SCALE_FACTORS = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6};

// Extra erosion of the outline, as a fraction of √area, before the mean fill:
// drops the dark rim (cut side + shadow) that cut-outs often keep.
constexpr double EXTRA_ERODE_FRAC = 0.025;

// Piece count unknown: fixed working width, and the template's width swept
// geometrically between these fractions of the artwork (≈100–3000 pieces).
constexpr int    WORK_WIDTH     = 1200;
constexpr double MIN_WIDTH_FRAC = 1.0 / 80;
constexpr double MAX_WIDTH_FRAC = 1.0 / 8;
constexpr int    SCALE_STEPS    = 14;

constexpr double COARSE_FACTOR     = 0.5; // coarse pass resolution relative to working
constexpr size_t REFINE_CANDIDATES = 3;   // orientation/scale pairs re-run at working res
constexpr int    MIN_TEMPLATE_PX   = 6;

struct Peak {
    bool        found = false;
    double      score = -1.0;
    cv::Point2f center;          // in the searched image's px
};

struct Candidate {
    double score;
    int    orientation;
    double scale;                // artwork px per piece px
};

// Best normalized-correlation peak of the piece scaled by `scale` into `image`.
// Pixels outside the outline hold the piece's mean, so they add nothing to the
// zero-mean correlation — nearly a masked match at FFT speed.
// In:     image — grey artwork at the resolution being searched
//         piece — grey piece, background already filled with its mean
//         scale — image px per piece px
// Out:    the peak (score in [-1, 1], centre in image px); found = false when
//         the scaled piece is under 6 px or not smaller than the image
// Errors: none (an unsearchable size is reported through found)
Peak bestPeak(const cv::Mat& image, const cv::Mat& piece, double scale) {
    Peak peak;
    cv::Mat tmpl;
    cv::resize(piece, tmpl, cv::Size(), scale, scale, cv::INTER_AREA);

    const bool fits = tmpl.cols >= MIN_TEMPLATE_PX && tmpl.rows >= MIN_TEMPLATE_PX
                   && tmpl.cols <  image.cols      && tmpl.rows <  image.rows;
    if (fits) {
        cv::Mat   response;
        cv::Point loc;
        cv::matchTemplate(image, tmpl, response, cv::TM_CCOEFF_NORMED);
        cv::minMaxLoc(response, nullptr, &peak.score, nullptr, &loc);
        peak.found  = true;
        peak.center = cv::Point2f(loc.x + tmpl.cols / 2.0f, loc.y + tmpl.rows / 2.0f);
    }
    return peak;
}

} // namespace

void TemplateMatcher::prepare(const cv::Mat& artwork, const GridConfig& grid) {
    grid_        = grid;
    artworkSize_ = artwork.size();
    pieceSide_   = grid.pieceSide(artworkSize_);
    workScale_   = pieceSide_ > 0.0 ? std::min(1.0, TARGET_PIECE_PX / pieceSide_)
                                    : static_cast<double>(WORK_WIDTH) / artwork.cols;

    cv::Mat gray;
    cv::cvtColor(artwork, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, work_, cv::Size(), workScale_, workScale_, cv::INTER_AREA);
    cv::resize(work_, coarse_, cv::Size(), COARSE_FACTOR, COARSE_FACTOR, cv::INTER_AREA);
}

MatchResult TemplateMatcher::match(const Piece& piece) const {
    const auto start = std::chrono::steady_clock::now();

    cv::Mat gray;
    cv::cvtColor(piece.bgr, gray, cv::COLOR_BGR2GRAY);
    const int radius = static_cast<int>(std::lround(EXTRA_ERODE_FRAC * std::sqrt(piece.area)));
    // A fresh buffer: piece.mask is shared with matchers running concurrently
    cv::Mat mask;
    if (radius > 0)
        cv::erode(piece.mask, mask,
                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * radius + 1, 2 * radius + 1)));
    else
        mask = piece.mask;
    gray.setTo(cv::mean(gray, mask), ~mask);
    std::vector<cv::Mat> orientations = {gray};
    for (int flag : {cv::ROTATE_90_CLOCKWISE, cv::ROTATE_180, cv::ROTATE_90_COUNTERCLOCKWISE}) {
        cv::Mat rotated;
        cv::rotate(gray, rotated, flag);
        orientations.push_back(rotated);
    }

    // Candidate scales, as artwork px per piece px
    std::vector<double> scales;
    if (pieceSide_ > 0.0) {
        const double expected = pieceSide_ / std::sqrt(piece.area);
        for (double f : SCALE_FACTORS)
            scales.push_back(expected * f);
    } else {
        const double ratio = std::pow(MAX_WIDTH_FRAC / MIN_WIDTH_FRAC, 1.0 / (SCALE_STEPS - 1));
        for (int s = 0; s < SCALE_STEPS; ++s)
            scales.push_back(MIN_WIDTH_FRAC * std::pow(ratio, s) * artworkSize_.width / piece.bgr.cols);
    }

    // Coarse pass: every orientation × scale on the reduced artwork
    const double coarseScale = workScale_ * COARSE_FACTOR;
    std::vector<Candidate> candidates;
    for (int o = 0; o < static_cast<int>(orientations.size()); ++o)
        for (double scale : scales) {
            const Peak p = bestPeak(coarse_, orientations[o], scale * coarseScale);
            if (p.found)
                candidates.push_back({p.score, o, scale});
        }

    // Fine pass: only the strongest coarse candidates, at working resolution
    const size_t refine = std::min(candidates.size(), REFINE_CANDIDATES);
    std::partial_sort(candidates.begin(), candidates.begin() + refine, candidates.end(),
                      [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    Peak best;
    for (size_t i = 0; i < refine; ++i) {
        const Peak p = bestPeak(work_, orientations[candidates[i].orientation],
                                candidates[i].scale * workScale_);
        if (p.found && p.score > best.score)
            best = p;
    }

    const cv::Point cell = best.found
        ? matcher_utils::cellAt(grid_, artworkSize_, best.center * static_cast<float>(1.0 / workScale_))
        : cv::Point(-1, -1);
    MatchResult result = matcher_utils::makeResult(name(), grid_, cell.y, cell.x,
                                                   static_cast<float>(best.score), start);
    if (!best.found)
        result.error = "Piece photo too small to search";
    return result;
}
