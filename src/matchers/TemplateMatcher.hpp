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
    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;
    MatchResult match(const Piece& piece) const override;
    std::string name() const override { return "Template"; }

private:
    GridConfig grid_;
    cv::Size   artworkSize_;
    double     pieceSide_ = 0.0; // expected piece side in artwork px; 0 = unknown
    cv::Mat    work_;            // grayscale artwork at working resolution
    cv::Mat    coarse_;          // work_ at COARSE_FACTOR, for the first pass
    double     workScale_ = 1.0; // working px per artwork px
};
