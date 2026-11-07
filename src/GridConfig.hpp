#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// Represents a single cell in the grid
struct GridCell {
    int row;        // 0-based
    int col;        // 0-based
    std::string label; // e.g. "B3"
};

class GridConfig {
public:
    int rows = 0;
    int cols = 0;
    std::vector<std::string> rowLabels;
    std::vector<std::string> colLabels;
    std::string artworkPath;   // resolved against the config directory; empty if none found
    int pieces = 0;            // approximate piece count of the puzzle; 0 = unknown

    // Loads and validates a grid.json. Missing labels are generated (rows A, B, …,
    // AA; cols 1, 2, …). A relative "artwork_path" resolves against the file's
    // directory; without one, the directory is searched for
    // artwork.{jpg,jpeg,png,bmp,tiff}.
    // In:     path — path to grid.json
    // Out:    the parsed config; artworkPath is empty if no artwork was found
    // Errors: throws std::runtime_error if the file can't be opened, rows/cols are
    //         not positive, a label list has the wrong length or pieces is
    //         negative; throws nlohmann::json exceptions on malformed JSON or a
    //         missing/mistyped field; std::filesystem::filesystem_error if the
    //         artwork search can't access the directory
    static GridConfig fromFile(const std::string& path);

    // Human-readable coordinate of a cell, row label then column label
    // In:     row, col — 0-based indices
    // Out:    e.g. "B3"; "?" if either index is out of range
    // Errors: none
    std::string coordLabel(int row, int col) const;

    // Pixel rectangle of a cell. Leftover pixels from integer division are
    // spread across cells, so the cells tile the whole artwork exactly.
    // In:     row, col    — 0-based indices (not range-checked)
    //         artworkSize — size of the full artwork
    // Out:    the cell's rectangle in artwork px
    // Errors: none; out-of-range indices give a rectangle outside the artwork
    cv::Rect cellRect(int row, int col, const cv::Size& artworkSize) const;

    // Expected side of one piece: sqrt(artwork area / pieces)
    // In:     artworkSize — size of the full artwork
    // Out:    side in artwork px; 0 if the piece count is unknown
    // Errors: none
    double pieceSide(const cv::Size& artworkSize) const;

    // Every cell with its label, as a flat row-major list
    // In:     none
    // Out:    rows × cols cells
    // Errors: none
    std::vector<GridCell> allCells() const;
};
