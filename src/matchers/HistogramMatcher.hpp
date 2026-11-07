#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "GridConfig.hpp"
#include "Matcher.hpp"
#include "MatcherUtils.hpp"

// Grey-level statistics of the piece (outline-masked) compared against
// overlapping piece-sized tiles of the artwork. Colour is ignored: the physical
// print is near-monochrome sepia. Each tile is described by three histograms that
// survive rotation and changes of brightness and contrast — the shape of its
// brightness distribution, rotation-invariant local binary patterns (texture) at
// two radii, and gradient strength relative to contrast (busy vs. smooth).
// Very fast, but blind to where things are inside the piece.
class HistogramMatcher : public Matcher {
public:
    // Describes every tile: piece-sized at half-piece steps when the piece
    // count is known, else cell/2 … cell/6 tiles. Each tile is resized to
    // 64 px before describing, so texture is compared at a common scale.
    // In:     artwork — full reference image, CV_8UC3 BGR
    //         grid    — cell layout; grid.pieces sets the tile size
    // Out:    none
    // Errors: may throw cv::Exception / std::bad_alloc on an unusable artwork
    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;

    // Describes the piece the same way (resized so √area ≈ 64 px, outline
    // eroded past the table/shadow rim) and returns the closest tile's cell.
    // In:     piece — normalized piece
    // Out:    the cell; confidence = how far ahead the best tile is of the best
    //         tile in any other cell (relative distance margin × 4, clamped)
    // Errors: none, provided prepare() produced tiles (an artwork smaller than
    //         one piece has none, and match() must not be called)
    MatchResult match(const Piece& piece) const override;

    // In: none.  Out: "Histogram".  Errors: none.
    std::string name() const override { return "Histogram"; }

private:
    GridConfig                       grid_;
    std::vector<matcher_utils::Tile> tiles_;
    std::vector<std::vector<float>>  tileDescs_; // parallel to tiles_
};
