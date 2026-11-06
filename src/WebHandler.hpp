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
    // Builds every matcher and prepares them against the artwork (in parallel)
    WebHandler(cv::Mat artwork, GridConfig grid, std::string modelPath);

    // JSON API used by the web page (routes listed in WebHandler.cpp)
    void registerRoutes(httplib::Server& svr);

    // Names of all matchers, in display order
    std::vector<std::string> matcherNames() const;

    // Runs the selected matchers (by name, case-insensitive) on the piece image in
    // parallel. A 4-channel image is the client's cut-out (alpha = piece outline).
    // Results follow display order; unknown names are ignored.
    std::vector<MatchResult> locate(const cv::Mat& image,
                                    const std::vector<std::string>& selected) const;

private:
    cv::Mat                               artwork_;
    GridConfig                            grid_;
    std::vector<std::unique_ptr<Matcher>> matchers_;
    std::vector<uchar>                    overviewJpeg_;   // downscaled artwork for the page
};
