#include "KeypointMatcher.hpp"
#include <algorithm>
#include <chrono>
#include <map>
#include <utility>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "MatcherUtils.hpp"

namespace {

constexpr int    ARTWORK_FEATURES    = 120000; // minimum, spread evenly across cells; more for
constexpr int    FEATURES_PER_PIECE  = 150;    //   large puzzles, capped to bound matching time
constexpr int    MAX_ARTWORK_FEATURES = 800000;
constexpr int    PIECE_FEATURES      = 1500;
constexpr int    PIECE_PYRAMID       = 12;    // 1.3^11 ≈ 18× — photo may be far larger than
constexpr float  PIECE_PYRAMID_SCALE = 1.3f;  //   the piece's footprint in the artwork
constexpr int    ORB_BORDER          = 31;    // ORB's default edge threshold
constexpr float  RATIO_TEST          = 0.75f;
constexpr int    MIN_HOMOGRAPHY_PTS  = 8;
constexpr int    MIN_INLIERS         = 6;
constexpr double RANSAC_REPROJ_PX    = 8.0;   // in artwork px
constexpr float  FULL_CONFIDENCE_INLIERS = 20.0f;

// BFMatcher limit on descriptors per train set (OpenCV's IMGIDX_ONE)
constexpr int MATCH_CHUNK = 200000;

// Two nearest artwork descriptors per piece descriptor (Hamming distance),
// searching the artwork in chunks to stay under BFMatcher's train-set limit
// In:     query — piece ORB descriptors, one CV_8U row each
//         train — artwork ORB descriptors, same format
// Out:    per query row, up to 2 matches sorted by distance; trainIdx indexes
//         the whole of train
// Errors: may throw cv::Exception if the descriptor widths differ
std::vector<std::vector<cv::DMatch>> twoNearest(const cv::Mat& query, const cv::Mat& train) {
    std::vector<std::vector<cv::DMatch>> best(query.rows);
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    for (int first = 0; first < train.rows; first += MATCH_CHUNK) {
        const int last = std::min(first + MATCH_CHUNK, train.rows);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(query, train.rowRange(first, last), knn, 2);
        for (const auto& candidates : knn)
            for (cv::DMatch m : candidates) {
                m.trainIdx += first;
                auto& b = best[m.queryIdx];
                b.push_back(m);
                std::sort(b.begin(), b.end(),
                          [](const cv::DMatch& x, const cv::DMatch& y) { return x.distance < y.distance; });
                if (b.size() > 2)
                    b.pop_back();
            }
    }
    return best;
}

} // namespace

void KeypointMatcher::prepare(const cv::Mat& artwork, const GridConfig& grid) {
    grid_        = grid;
    artworkSize_ = artwork.size();
    keypoints_.clear();
    keypointCell_.clear();
    descriptors_.release();

    cv::Mat gray;
    cv::cvtColor(artwork, gray, cv::COLOR_BGR2GRAY);
    const cv::Rect bounds(cv::Point(0, 0), gray.size());
    const int total   = std::clamp(grid.pieces * FEATURES_PER_PIECE, ARTWORK_FEATURES, MAX_ARTWORK_FEATURES);
    const int perCell = total / (grid.rows * grid.cols);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(perCell);

    // Detect per cell so every cell gets its share of features. Each cell is
    // padded by ORB's border so keypoints near cell edges are not lost.
    for (const GridCell& cell : grid.allCells()) {
        const cv::Rect cr     = grid.cellRect(cell.row, cell.col, artworkSize_);
        const cv::Rect padded = (cr + cv::Size(2 * ORB_BORDER, 2 * ORB_BORDER)
                                    - cv::Point(ORB_BORDER, ORB_BORDER)) & bounds;
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        orb->detectAndCompute(gray(padded), cv::noArray(), kps, desc);

        for (size_t i = 0; i < kps.size(); ++i) {
            kps[i].pt += cv::Point2f(padded.tl());
            if (cr.contains(kps[i].pt)) {
                keypoints_.push_back(kps[i]);
                keypointCell_.emplace_back(cell.col, cell.row);
                descriptors_.push_back(desc.row(static_cast<int>(i)));
            }
        }
    }
}

MatchResult KeypointMatcher::match(const Piece& piece) const {
    const auto start = std::chrono::steady_clock::now();

    cv::Mat gray;
    cv::cvtColor(piece.bgr, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    cv::ORB::create(PIECE_FEATURES, PIECE_PYRAMID_SCALE, PIECE_PYRAMID)
        ->detectAndCompute(gray, piece.mask, kps, desc);

    std::vector<cv::Point2f> piecePts, artPts;
    std::vector<int>         artIdx;
    if (!desc.empty() && !descriptors_.empty()) {
        const std::vector<std::vector<cv::DMatch>> knn = twoNearest(desc, descriptors_);
        for (const auto& m : knn)
            if (m.size() == 2 && m[0].distance < RATIO_TEST * m[1].distance) {
                piecePts.push_back(kps[m[0].queryIdx].pt);
                artPts.push_back(keypoints_[m[0].trainIdx].pt);
                artIdx.push_back(m[0].trainIdx);
            }
    }

    cv::Point cell(-1, -1);
    float     confidence = 0.0f;
    int       inliers    = 0;

    if (static_cast<int>(piecePts.size()) >= MIN_HOMOGRAPHY_PTS) {
        std::vector<unsigned char> mask;
        const cv::Mat H = cv::findHomography(piecePts, artPts, cv::RANSAC, RANSAC_REPROJ_PX, mask);
        inliers = cv::countNonZero(mask);
        if (!H.empty() && inliers >= MIN_INLIERS) {
            std::vector<cv::Point2f> centre = {cv::Point2f(gray.cols / 2.0f, gray.rows / 2.0f)}, mapped;
            cv::perspectiveTransform(centre, mapped, H);
            cell = matcher_utils::cellAt(grid_, artworkSize_, mapped[0]);
            confidence = (static_cast<float>(inliers) / piecePts.size())
                       * std::min(1.0f, inliers / FULL_CONFIDENCE_INLIERS);
        }
    }

    // No consistent geometry: fall back to the cell with the most matches
    if (cell.x < 0 && !artIdx.empty()) {
        std::map<std::pair<int, int>, int> votes;
        for (int idx : artIdx)
            ++votes[{keypointCell_[idx].y, keypointCell_[idx].x}];
        const auto top = std::max_element(votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        cell       = cv::Point(top->first.second, top->first.first);
        confidence = 0.5f * top->second / artIdx.size()
                   * std::min(1.0f, top->second / FULL_CONFIDENCE_INLIERS);
    }

    MatchResult result = matcher_utils::makeResult(name(), grid_, cell.y, cell.x, confidence, start);
    if (cell.x < 0)
        result.error = "No distinctive features matched";
    return result;
}
