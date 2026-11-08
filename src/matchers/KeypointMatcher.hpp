#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "GridConfig.hpp"
#include "Matcher.hpp"

// ORB features of the piece matched (brute-force Hamming + a ratio test against
// the nearest artwork feature at least a piece away) against ORB features of the
// whole artwork. A RANSAC homography maps the piece's
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

    // Shrinks the piece to about twice its footprint in the artwork (when the
    // piece count is known), detects ORB features inside the outline with a low
    // FAST threshold (the print is soft) and matches them against the artwork's.
    // A match is kept when it beats, by the ratio test, the nearest artwork
    // feature at least a piece away (near neighbours are the same spot, not a
    // rival); a match whose 8 nearest all lie together is kept too. Fits a RANSAC
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
    // The piece outline where the geometric match puts it: a similarity
    // (rotation, scale, shift) fitted to the homography's inliers, or a
    // piece-sized square at the matched centre when that fit is implausible
    // In:     contour    — piece outline, Piece px
    //         shrink     — factor the piece was resized by before ORB (≤ 1)
    //         piecePts   — matched points in the resized piece
    //         artPts     — their artwork points
    //         inlierMask — homography RANSAC inliers, one flag per pair
    //         centre     — piece centre mapped by the homography, artwork px
    // Out:    the outline in artwork px; empty if neither placement is possible
    // Errors: none
    std::vector<cv::Point2f> placeOutline(const std::vector<cv::Point2f>& contour, double shrink,
                                          const std::vector<cv::Point2f>& piecePts,
                                          const std::vector<cv::Point2f>& artPts,
                                          const std::vector<unsigned char>& inlierMask,
                                          const cv::Point2f& centre) const;

    GridConfig                grid_;
    cv::Size                  artworkSize_;
    double                    pieceSide_ = 0.0; // expected piece side, artwork px; 0 = unknown
    std::vector<cv::KeyPoint> keypoints_;   // artwork keypoints, artwork px
    std::vector<cv::Point>    keypointCell_; // (col, row) per keypoint
    cv::Mat                   descriptors_;  // one row per keypoint
};
