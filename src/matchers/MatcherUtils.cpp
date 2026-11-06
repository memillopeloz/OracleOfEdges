#include "MatcherUtils.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace matcher_utils {

namespace {

constexpr int    PIECE_MAX_DIM = 800;   // uploaded piece is capped to this
constexpr double FALLBACK_CORE = 0.6;   // centre fraction used when no outline is supplied
constexpr double ERODE_FRAC    = 0.02;  // mask erosion, as a fraction of the piece's size

// Rotate by `angle` degrees, enlarging the canvas so nothing is cut off
cv::Mat rotateBound(const cv::Mat& img, double angle, int interpolation, int border) {
    const cv::Point2f centre(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat R = cv::getRotationMatrix2D(centre, angle, 1.0);
    const cv::Rect2f bounds = cv::RotatedRect(cv::Point2f(), cv::Size2f(img.size()), static_cast<float>(angle))
                                  .boundingRect2f();
    R.at<double>(0, 2) += bounds.width  / 2.0 - centre.x;
    R.at<double>(1, 2) += bounds.height / 2.0 - centre.y;

    cv::Mat out;
    cv::warpAffine(img, out, R, cv::Size(cvCeil(bounds.width), cvCeil(bounds.height)),
                   interpolation, border);
    return out;
}

// Dominant edge direction of a piece outline, in degrees within (-45, 45].
// The body's straight edges agree modulo 90°, while the round knobs point every
// way and cancel out, so a length-weighted circular mean of 4θ finds the edges.
double dominantEdgeAngle(const std::vector<cv::Point>& contour) {
    const size_t n    = contour.size();
    const size_t step = std::max<size_t>(3, n / 60);  // long enough to smooth pixel staircases
    double sumCos = 0.0, sumSin = 0.0;
    for (size_t i = 0; i < n; i += step) {
        const cv::Point2d d = contour[(i + step) % n] - contour[i];
        const double length = std::hypot(d.x, d.y);
        const double theta  = std::atan2(d.y, d.x);
        sumCos += length * std::cos(4.0 * theta);
        sumSin += length * std::sin(4.0 * theta);
    }
    return std::atan2(sumSin, sumCos) / 4.0 * 180.0 / CV_PI;
}

// Largest axis-aligned square fully inside the mask: start from the circle
// inscribed at the distance-transform peak, then grow while it stays inside.
cv::Rect inscribedSquare(const cv::Mat& mask) {
    cv::Mat dist, sum;
    cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    double    radius = 0.0;
    cv::Point centre;
    cv::minMaxLoc(dist, nullptr, &radius, nullptr, &centre);
    cv::integral(mask / 255, sum, CV_32S);

    const cv::Rect bounds(cv::Point(0, 0), mask.size());
    const auto squareAt = [&](int half) {
        return cv::Rect(centre.x - half, centre.y - half, 2 * half + 1, 2 * half + 1);
    };
    const auto inside = [&](const cv::Rect& r) {
        const bool inBounds = (r & bounds) == r;
        return inBounds && sum.at<int>(r.br()) - sum.at<int>(r.y, r.x + r.width)
                         - sum.at<int>(r.y + r.height, r.x) + sum.at<int>(r.tl()) == r.area();
    };

    int half = std::max(0, static_cast<int>(radius / std::sqrt(2.0)) - 1);
    while (inside(squareAt(half + 1)))
        ++half;
    return squareAt(half) & bounds;
}

} // namespace

cv::Mat limitSize(const cv::Mat& img, int maxDim, int interpolation) {
    cv::Mat out = img;
    const int longest = std::max(img.cols, img.rows);
    if (longest > maxDim) {
        const double scale = static_cast<double>(maxDim) / longest;
        cv::resize(img, out, cv::Size(), scale, scale, interpolation);
    }
    return out;
}

Piece makePiece(const cv::Mat& image) {
    cv::Mat bgr, mask;
    if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
        cv::extractChannel(image, mask, 3);
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
    } else {
        if (image.channels() == 1)
            cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
        else
            bgr = image;
        const int w = static_cast<int>(image.cols * FALLBACK_CORE);
        const int h = static_cast<int>(image.rows * FALLBACK_CORE);
        mask = cv::Mat::zeros(image.size(), CV_8U);
        mask(cv::Rect((image.cols - w) / 2, (image.rows - h) / 2, w, h)).setTo(255);
    }
    bgr  = limitSize(bgr, PIECE_MAX_DIM);
    mask = limitSize(mask, PIECE_MAX_DIM, cv::INTER_NEAREST);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    Piece piece;
    if (!contours.empty()) {
        // Keep only the largest blob
        const auto largest = std::max_element(contours.begin(), contours.end(),
            [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
        mask = cv::Mat::zeros(mask.size(), CV_8U);
        cv::drawContours(mask, contours, static_cast<int>(largest - contours.begin()),
                         cv::Scalar(255), cv::FILLED);

        // Turn so the piece's edges are axis-aligned; ±45° covers every case
        const double angle = dominantEdgeAngle(*largest);
        bgr  = rotateBound(bgr,  angle, cv::INTER_LINEAR,  cv::BORDER_REPLICATE);
        mask = rotateBound(mask, angle, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

        const cv::Rect box = cv::boundingRect(mask);
        piece.bgr  = bgr(box).clone();
        piece.mask = mask(box).clone();
        piece.area = cv::countNonZero(piece.mask);

        // Drop the outline's fringe, where background colour bleeds into the piece
        const int r = std::max(1, static_cast<int>(ERODE_FRAC * std::max(box.width, box.height)));
        cv::erode(piece.mask, piece.mask,
                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r + 1, 2 * r + 1)));
        piece.core = piece.bgr(inscribedSquare(piece.mask)).clone();
    }
    return piece;
}

std::vector<Tile> gridTiles(const GridConfig& grid, const cv::Size& artworkSize,
                            int side, int stride) {
    std::vector<Tile> tiles;
    for (int y = 0; y + side <= artworkSize.height; y += stride)
        for (int x = 0; x + side <= artworkSize.width; x += stride) {
            const cv::Rect  rect(x, y, side, side);
            const cv::Point cell = cellAt(grid, artworkSize, (rect.tl() + rect.br()) / 2);
            tiles.push_back({rect, cell.y, cell.x});
        }
    return tiles;
}

std::vector<Tile> cellTiles(const GridConfig& grid, const cv::Size& artworkSize,
                            const std::vector<int>& divisions) {
    std::vector<Tile> tiles;
    for (const GridCell& cell : grid.allCells()) {
        const cv::Rect cr = grid.cellRect(cell.row, cell.col, artworkSize);
        for (int d : divisions) {
            const int tw = cr.width / d;
            const int th = cr.height / d;
            const int steps = 2 * d - 1;
            for (int i = 0; i < steps; ++i)
                for (int k = 0; k < steps; ++k) {
                    const cv::Rect rect(cr.x + k * tw / 2, cr.y + i * th / 2, tw, th);
                    tiles.push_back({rect & cr, cell.row, cell.col});
                }
        }
    }
    return tiles;
}

cv::Point cellAt(const GridConfig& grid, const cv::Size& artworkSize, const cv::Point2f& pt) {
    cv::Point found(-1, -1);
    if (pt.x >= 0 && pt.y >= 0 && pt.x < artworkSize.width && pt.y < artworkSize.height) {
        // Inverse of GridConfig::cellRect's boundaries (floor(i * size / n))
        int col = static_cast<int>(pt.x) * grid.cols / artworkSize.width;
        int row = static_cast<int>(pt.y) * grid.rows / artworkSize.height;
        const cv::Point p(static_cast<int>(pt.x), static_cast<int>(pt.y));
        while (col + 1 < grid.cols && grid.cellRect(row, col + 1, artworkSize).x <= p.x) ++col;
        while (row + 1 < grid.rows && grid.cellRect(row + 1, col, artworkSize).y <= p.y) ++row;
        found = cv::Point(col, row);
    }
    return found;
}

MatchResult makeResult(const std::string& matcherName, const GridConfig& grid,
                       int row, int col, float confidence,
                       const std::chrono::steady_clock::time_point& start) {
    MatchResult r;
    r.row         = row;
    r.col         = col;
    r.coord       = grid.coordLabel(row, col);
    r.confidence  = std::clamp(confidence, 0.0f, 1.0f);
    r.timeMs      = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
    r.matcherName = matcherName;
    return r;
}

} // namespace matcher_utils
