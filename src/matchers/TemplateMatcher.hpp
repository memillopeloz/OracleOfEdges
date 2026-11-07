#pragma once
#include <string>
#include <opencv2/core.hpp>
#include "GridConfig.hpp"
#include "Matcher.hpp"

// Masked normalized cross-correlation (cv::matchTemplate) of the piece against the
// whole artwork, in four 90° orientations. With a piece count the expected scale
// is derived from the piece's area; otherwise a wide range of scales is swept.
// Coarse pass first, then the strongest candidates at working resolution.
class TemplateMatcher : public Matcher {
public:
    // Keeps a grey copy of the artwork at working resolution (a piece ≈ 32 px
    // across when the piece count is known, else 1200 px wide) and a half-size
    // copy for the coarse pass.
    // In:     artwork — full reference image, CV_8UC3 BGR
    //         grid    — cell layout; grid.pieces sets the working scale
    // Out:    none
    // Errors: may throw cv::Exception / std::bad_alloc on an unusable artwork
    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;

    // Correlates the grey piece (outside the outline filled with its mean) in 4
    // orientations × candidate scales on the coarse artwork, then re-runs the
    // top 3 at working resolution and takes the strongest peak.
    // In:     piece — normalized piece; its area sets the expected scale
    // Out:    the cell under the peak's centre; confidence = the peak's
    //         normalized correlation (barely lower when wrong — poorly calibrated)
    // Errors: error "Piece photo too small to search" (row/col = -1) when no
    //         scaled template fits between 6 px and the artwork size
    MatchResult match(const Piece& piece) const override;

    // In: none.  Out: "Template".  Errors: none.
    std::string name() const override { return "Template"; }

private:
    GridConfig grid_;
    cv::Size   artworkSize_;
    double     pieceSide_ = 0.0; // expected piece side in artwork px; 0 = unknown
    cv::Mat    work_;            // grayscale artwork at working resolution
    cv::Mat    coarse_;          // work_ at COARSE_FACTOR, for the first pass
    double     workScale_ = 1.0; // working px per artwork px
};
