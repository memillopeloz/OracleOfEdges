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

    // Load from a JSON file. Missing labels are generated (rows A, B, …, AA; cols 1, 2, …).
    // A relative "artwork_path" resolves against the file's directory; without one,
    // the directory is searched for artwork.{jpg,jpeg,png,bmp,tiff}.
    // Throws std::runtime_error (or a JSON parse error) on invalid input.
    static GridConfig fromFile(const std::string& path);

    // Build a coordinate label from 0-based indices ("?" if out of range)
    std::string coordLabel(int row, int col) const;

    // Pixel rect of a cell given the full artwork size. Leftover pixels from
    // integer division are spread across cells, so the grid covers the whole artwork.
    cv::Rect cellRect(int row, int col, const cv::Size& artworkSize) const;

    // Expected side of one piece in artwork px (sqrt of artwork area / pieces); 0 if unknown
    double pieceSide(const cv::Size& artworkSize) const;

    // All cells as a flat list, row-major
    std::vector<GridCell> allCells() const;
};
