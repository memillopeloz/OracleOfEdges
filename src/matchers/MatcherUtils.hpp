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

// Downscales so the longest side is at most maxDim; never upscales
// In:     img           — any image
//         maxDim        — longest allowed side, px (> 0)
//         interpolation — cv::resize flag (INTER_NEAREST for masks)
// Out:    the resized image, or img itself (shared data) if already small enough
// Errors: cv::Exception if maxDim ≤ 0 (the image would shrink to nothing)
cv::Mat limitSize(const cv::Mat& img, int maxDim, int interpolation = cv::INTER_AREA);

// Normalizes an uploaded image into a Piece: caps it at 800 px, keeps the
// largest blob of the outline, turns it so the body's edges are axis-aligned,
// crops to it, erodes the fringe and finds the inscribed square (core).
// In:     image — a 4-channel cut-out (alpha > 127 = piece), or a 1- or
//                 3-channel photo whose centred 60% is taken as the piece
// Out:    the Piece; bgr, mask and core are empty if there are no piece pixels
// Errors: none for valid 8-bit images (an empty result signals "no piece");
//         cv::Exception for an empty or unsupported image
Piece makePiece(const cv::Mat& image);

// Square tiles stepped over the whole artwork, each labelled with the cell
// holding its centre. Tiles that would cross the right/bottom edge are skipped.
// In:     grid        — cell layout
//         artworkSize — size of the full artwork
//         side        — tile side, px (> 0)
//         stride      — step between tiles, px (> 0)
// Out:    tiles, row-major; empty if side exceeds the artwork
// Errors: none; stride ≤ 0 would loop forever, so callers clamp it to ≥ 1
std::vector<Tile> gridTiles(const GridConfig& grid, const cv::Size& artworkSize,
                            int side, int stride);

// Overlapping tiles inside every cell, for when the piece count is unknown. For
// each d in divisions, tiles are cell/d in size and stepped by half a tile,
// giving (2d-1)² tiles per cell, all labelled with that cell.
// In:     grid        — cell layout
//         artworkSize — size of the full artwork
//         divisions   — tile sizes as fractions of a cell (each ≥ 1)
// Out:    tiles, cell by cell
// Errors: none; cells smaller than d px give zero-sized tiles, which callers
//         must not crop
std::vector<Tile> cellTiles(const GridConfig& grid, const cv::Size& artworkSize,
                            const std::vector<int>& divisions);

// Cell containing an artwork point (exact inverse of GridConfig::cellRect)
// In:     grid        — cell layout
//         artworkSize — size of the full artwork
//         pt          — point in artwork px
// Out:    (col, row) — note x = column; (-1, -1) if pt lies outside the artwork
// Errors: none
cv::Point cellAt(const GridConfig& grid, const cv::Size& artworkSize, const cv::Point2f& pt);

// Axis-aligned square around a point, as a closed polygon (for matchers that
// know where the piece is but not its outline)
// In:     center — in artwork px
//         side   — square side, px
// Out:    the 4 corners, clockwise from top-left
// Errors: none
std::vector<cv::Point2f> squareOutline(const cv::Point2f& center, double side);

// A rectangle as a closed polygon
// In:     rect — in artwork px
// Out:    the 4 corners, clockwise from top-left
// Errors: none
std::vector<cv::Point2f> rectOutline(const cv::Rect& rect);

// Assembles a result, stamping the coordinate label and the time since start
// In:     matcherName — Matcher::name()
//         grid        — for the label
//         row, col    — 0-based cell; -1 when there is no answer
//         confidence  — clamped to [0, 1]
//         start       — when the matcher began work on this piece
// Out:    the result with an empty error (callers set it on failure); coord is
//         "?" for an out-of-range cell
// Errors: none
MatchResult makeResult(const std::string& matcherName, const GridConfig& grid,
                       int row, int col, float confidence,
                       const std::chrono::steady_clock::time_point& start);

} // namespace matcher_utils
