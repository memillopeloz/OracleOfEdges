#pragma once
#include <string>
#include <opencv2/core.hpp>
#include "GridConfig.hpp"

// Result from any matcher
struct MatchResult {
    int    row        = -1;
    int    col        = -1;
    std::string coord;          // e.g. "B3"
    float  confidence = 0.0f;  // 0.0 – 1.0
    long   timeMs     = 0;      // processing time
    std::string matcherName;
    std::string error;          // empty on success; otherwise why no result was produced
};

// The piece photo, normalized once per request by matcher_utils::makePiece:
// cropped to the piece, turned so its edges are axis-aligned (the remaining
// ambiguity is a multiple of 90°), and capped in size.
struct Piece {
    cv::Mat bgr;          // CV_8UC3 image of the piece
    cv::Mat mask;         // CV_8U, same size; 255 on piece pixels (slightly eroded)
    cv::Mat core;         // CV_8UC3, largest axis-aligned square fully inside the piece
    double  area = 0.0;   // piece pixel count before erosion, in bgr's px
};

// Abstract base — every matcher implements this interface.
// Lifecycle: prepare() once at startup, then match() per request.
class Matcher {
public:
    virtual ~Matcher() = default;

    // Called once at startup with the full reference artwork. Precompute whatever
    // per-cell data the matcher needs (crops, histograms, keypoints, embeddings)
    // and keep a copy of the grid for labelling results.
    virtual void prepare(const cv::Mat& artwork, const GridConfig& grid) = 0;

    // May be called from several threads at once — must not mutate matcher state.
    virtual MatchResult match(const Piece& piece) const = 0;

    virtual std::string name() const = 0;

    // False when the matcher cannot run (e.g. the ML model file is missing)
    virtual bool available() const { return true; }
};
