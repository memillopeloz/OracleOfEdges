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
    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;
    MatchResult match(const Piece& piece) const override;
    std::string name() const override { return "Keypoint"; }

private:
    GridConfig                grid_;
    cv::Size                  artworkSize_;
    std::vector<cv::KeyPoint> keypoints_;   // artwork keypoints, artwork px
    std::vector<cv::Point>    keypointCell_; // (col, row) per keypoint
    cv::Mat                   descriptors_;  // one row per keypoint
};
