#include "HistogramMatcher.hpp"
#include <algorithm>
#include <chrono>
#include <opencv2/imgproc.hpp>

namespace {

constexpr double TILE_STRIDE = 0.5;          // piece count known: tiles of one piece, half-piece steps
const std::vector<int> TILE_DIVISIONS = {2, 3, 4, 6}; // unknown: tiles of cell/2 … cell/6

// Normalized 2-D hue/saturation histogram of a BGR image, optionally masked
cv::Mat hsHistogram(const cv::Mat& bgr, const cv::Mat& mask = cv::Mat()) {
    static const int   channels[] = {0, 1};
    static const int   bins[]     = {18, 16};
    static const float hueRange[] = {0, 180};
    static const float satRange[] = {0, 256};
    static const float* ranges[]  = {hueRange, satRange};

    cv::Mat hsv, hist;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::calcHist(&hsv, 1, channels, mask, hist, 2, bins, ranges);
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    return hist;
}

} // namespace

void HistogramMatcher::prepare(const cv::Mat& artwork, const GridConfig& grid) {
    grid_  = grid;
    const double side = grid.pieceSide(artwork.size());
    tiles_ = side > 0.0
           ? matcher_utils::gridTiles(grid, artwork.size(), cvRound(side),
                                      std::max(1, cvRound(side * TILE_STRIDE)))
           : matcher_utils::cellTiles(grid, artwork.size(), TILE_DIVISIONS);
    tileHists_.clear();
    tileHists_.reserve(tiles_.size());
    for (const matcher_utils::Tile& tile : tiles_)
        tileHists_.push_back(hsHistogram(artwork(tile.rect)));
}

MatchResult HistogramMatcher::match(const Piece& piece) const {
    const auto start = std::chrono::steady_clock::now();

    const cv::Mat hist = hsHistogram(piece.bgr, piece.mask);

    double best    = -1.0;
    size_t bestIdx = 0;
    for (size_t i = 0; i < tileHists_.size(); ++i) {
        const double score = cv::compareHist(hist, tileHists_[i], cv::HISTCMP_CORREL);
        if (score > best) {
            best    = score;
            bestIdx = i;
        }
    }

    const matcher_utils::Tile& tile = tiles_[bestIdx];
    return matcher_utils::makeResult(name(), grid_, tile.row, tile.col, static_cast<float>(best), start);
}
