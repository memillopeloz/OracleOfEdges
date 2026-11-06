#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>   // interpolation flags
#include "GridConfig.hpp"
#include "Matcher.hpp"

// Helpers shared by the matcher implementations
namespace matcher_utils {

// A region of the artwork, labelled with the cell containing its centre. Used by
// matchers that compare the piece against piece-sized regions.
struct Tile {
    cv::Rect rect;
    int      row;
    int      col;
};

// Downscale so the longest side is at most maxDim (never upscales)
cv::Mat limitSize(const cv::Mat& img, int maxDim, int interpolation = cv::INTER_AREA);

// Normalize an uploaded image into a Piece. A 4-channel image is taken as the
// client's cut-out: alpha marks the piece. Otherwise the piece is assumed to be
// centred and the middle of the photo is used. Returns a Piece with empty bgr
// if no piece pixels are found.
Piece makePiece(const cv::Mat& image);

// Square tiles of `side` px stepped by `stride` px over the whole artwork
std::vector<Tile> gridTiles(const GridConfig& grid, const cv::Size& artworkSize,
                            int side, int stride);

// Overlapping tiles inside every cell. For each d in `divisions`, tiles are
// cell/d in size and stepped by half a tile, giving (2d-1)² tiles per cell.
// Used when the piece count is unknown.
std::vector<Tile> cellTiles(const GridConfig& grid, const cv::Size& artworkSize,
                            const std::vector<int>& divisions);

// Cell containing an artwork pixel as (col, row); (-1, -1) if outside the artwork
cv::Point cellAt(const GridConfig& grid, const cv::Size& artworkSize, const cv::Point2f& pt);

// Assemble a result, stamping coordinate label and elapsed time
MatchResult makeResult(const std::string& matcherName, const GridConfig& grid,
                       int row, int col, float confidence,
                       const std::chrono::steady_clock::time_point& start);

} // namespace matcher_utils
