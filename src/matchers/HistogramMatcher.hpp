#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "GridConfig.hpp"
#include "Matcher.hpp"
#include "MatcherUtils.hpp"

// HSV hue/saturation histogram of the piece (outline-masked) compared
// (cv::HISTCMP_CORREL) against overlapping piece-sized tiles of the artwork.
// Rotation-invariant and very fast, but blind to structure — best on artwork
// with distinct colour regions.
class HistogramMatcher : public Matcher {
public:
    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;
    MatchResult match(const Piece& piece) const override;
    std::string name() const override { return "Histogram"; }

private:
    GridConfig                       grid_;
    std::vector<matcher_utils::Tile> tiles_;
    std::vector<cv::Mat>             tileHists_; // parallel to tiles_
};
