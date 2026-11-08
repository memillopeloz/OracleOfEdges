#pragma once
#include <string>
#include <vector>
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
    // Where the piece lies, in artwork px, as a closed polygon: the piece's own
    // outline when the matcher knows its pose, else a piece-sized box around the
    // best spot. Empty when there is no answer or no position finer than the cell.
    std::vector<cv::Point2f> outline;
};

// The piece photo, normalized once per request by matcher_utils::makePiece:
// cropped to the piece, turned so its edges are axis-aligned (the remaining
// ambiguity is a multiple of 90°), and capped in size.
struct Piece {
    cv::Mat bgr;          // CV_8UC3 image of the piece
    cv::Mat mask;         // CV_8U, same size; 255 on piece pixels (slightly eroded)
    cv::Mat core;         // CV_8UC3, largest axis-aligned square fully inside the piece
    double  area = 0.0;   // piece pixel count before erosion, in bgr's px
    std::vector<cv::Point2f> contour;   // simplified outline before erosion, in bgr's px
};

// Abstract base — every matcher implements this interface.
// Lifecycle: prepare() once at startup, then match() per request.
class Matcher {
public:
    virtual ~Matcher() = default;

    // Precomputes everything match() needs from the artwork (tiles, histograms,
    // keypoints, embeddings) and keeps a copy of the grid for labelling results.
    // Called once at startup, before any match().
    // In:     artwork — full reference image, CV_8UC3 BGR
    //         grid    — cell layout; grid.pieces (if > 0) gives the expected piece scale
    // Out:    none; the matcher's state is filled
    // Errors: may throw cv::Exception or std::bad_alloc on an unusable artwork.
    //         A matcher that cannot run at all reports it through available().
    virtual void prepare(const cv::Mat& artwork, const GridConfig& grid) = 0;

    // Locates one piece in the artwork. Safe to call from several threads at
    // once: implementations must not mutate matcher state.
    // In:     piece — normalized piece from matcher_utils::makePiece (non-empty bgr)
    // Out:    the best cell, a confidence in [0, 1], elapsed time and matcher name
    // Errors: not expected to throw; sets MatchResult::error (row/col = -1) when no
    //         answer could be produced (no features, piece too small, model missing)
    virtual MatchResult match(const Piece& piece) const = 0;

    // Display name, also the key the client uses to select the matcher
    // In:     none
    // Out:    e.g. "Template"
    // Errors: none
    virtual std::string name() const = 0;

    // Whether the matcher can run (e.g. false when the ML model file is missing)
    // In:     none
    // Out:    true if match() can produce results
    // Errors: none
    virtual bool available() const { return true; }
};
