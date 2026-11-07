#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "GridConfig.hpp"
#include "Matcher.hpp"

// ORB features of the piece matched (brute-force Hamming + Lowe's ratio test)
// against ORB features of the whole artwork. A RANSAC homography maps the piece's
// centre into the artwork; with too few inliers, cells vote by matched keypoints.
// Robust to rotation and moderate scale change.
class KeypointMatcher : public Matcher {
public:
    // Detects ORB features cell by cell, so every cell gets an equal share:
    // 150 per piece (120k–800k in total), each tagged with its cell.
    // In:     artwork — full reference image, CV_8UC3 BGR
    //         grid    — cell layout; grid.pieces sets the feature budget
    // Out:    none
    // Errors: may throw cv::Exception / std::bad_alloc; a featureless artwork
    //         just leaves no features, and match() then reports an error
    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;

    // Matches the piece's ORB features (12-level pyramid, inside the outline)
    // against the artwork's, keeps those passing the ratio test, fits a RANSAC
    // homography and maps the piece's centre into the artwork. Without enough
    // inliers, the cell holding the most matched features wins instead.
    // In:     piece — normalized piece
    // Out:    the cell; confidence = inlier ratio × min(1, inliers / 20), halved
    //         for the voting fallback (≈ 0 when wrong)
    // Errors: error "No distinctive features matched" (row/col = -1) when no
    //         feature survives the ratio test
    MatchResult match(const Piece& piece) const override;

    // In: none.  Out: "Keypoint".  Errors: none.
    std::string name() const override { return "Keypoint"; }

private:
    GridConfig                grid_;
    cv::Size                  artworkSize_;
    std::vector<cv::KeyPoint> keypoints_;   // artwork keypoints, artwork px
    std::vector<cv::Point>    keypointCell_; // (col, row) per keypoint
    cv::Mat                   descriptors_;  // one row per keypoint
};
