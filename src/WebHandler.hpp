#pragma once
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <httplib.h>
#include "GridConfig.hpp"
#include "Matcher.hpp"

class WebHandler {
public:
    // Builds every matcher and prepares them against the artwork in parallel,
    // then encodes the downscaled overview JPEG served by /api/artwork. Blocks
    // until all matchers are ready (tens of seconds on a large artwork).
    // In:     artwork   — full reference image, CV_8UC3 BGR (non-empty)
    //         grid      — cell layout
    //         modelPath — MobileNetV2 ONNX file; empty makes the ML matcher unavailable
    // Out:    a ready handler
    // Errors: rethrows any exception from a matcher's prepare(). A bad or missing
    //         model doesn't throw: the ML matcher reports itself unavailable.
    WebHandler(cv::Mat artwork, GridConfig grid, std::string modelPath);

    // Adds the JSON API used by the web page:
    //   GET  /api/info     grid layout and matchers: {rows, cols, rowLabels, colLabels,
    //                      matchers: [{name, available}]}
    //   GET  /api/artwork  downscaled artwork JPEG for the overview
    //   POST /api/locate   multipart: "piece" = PNG cut-out (alpha = outline), one
    //                      "matcher" field per selected matcher →
    //                      {results: [{matcher, coord, row, col, confidence, timeMs, error}]}
    // In:     svr — server to register on; this handler must outlive it
    // Out:    none
    // Errors: /api/locate answers 400 {error} when the piece is missing or
    //         undecodable, or no matcher is selected. Per-matcher failures come
    //         back as a result with a non-empty "error", not as an HTTP error.
    void registerRoutes(httplib::Server& svr);

    // Names of all matchers, in display order
    // In:     none
    // Out:    e.g. {"Template", "ML", "Histogram", "Keypoint"}
    // Errors: none
    std::vector<std::string> matcherNames() const;

    // Normalizes the piece once (matcher_utils::makePiece) and runs the selected
    // matchers on it in parallel
    // In:     image    — the client's cut-out (BGRA, alpha = outline), or a plain
    //                    BGR/grey photo whose centre 60% is taken as the piece
    //         selected — matcher names, case-insensitive; unknown names are ignored
    // Out:    one result per selected, known matcher, in display order
    // Errors: when the image has no piece pixels, every result has error "No piece
    //         found in the image"; other per-matcher errors as in Matcher::match.
    //         Rethrows any exception escaping a matcher's match().
    std::vector<MatchResult> locate(const cv::Mat& image,
                                    const std::vector<std::string>& selected) const;

private:
    cv::Mat                               artwork_;
    GridConfig                            grid_;
    std::vector<std::unique_ptr<Matcher>> matchers_;
    std::vector<uchar>                    overviewJpeg_;   // downscaled artwork for the page
};
