#include "WebHandler.hpp"
#include <algorithm>
#include <cctype>
#include <future>
#include <utility>
#include "matchers/HistogramMatcher.hpp"
#include "matchers/KeypointMatcher.hpp"
#include "matchers/MLMatcher.hpp"
#include "matchers/MatcherUtils.hpp"
#include "matchers/TemplateMatcher.hpp"

namespace {

std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

} // namespace

WebHandler::WebHandler(cv::Mat artwork, GridConfig grid, std::string modelPath)
    : artwork_(std::move(artwork)), grid_(std::move(grid)) {
    matchers_.push_back(std::make_unique<TemplateMatcher>());
    matchers_.push_back(std::make_unique<MLMatcher>(modelPath));
    matchers_.push_back(std::make_unique<HistogramMatcher>());
    matchers_.push_back(std::make_unique<KeypointMatcher>());

    std::vector<std::future<void>> pending;
    for (auto& m : matchers_)
        pending.push_back(std::async(std::launch::async,
                                     [&m, this] { m->prepare(artwork_, grid_); }));
    for (auto& p : pending)
        p.get();
}

void WebHandler::registerRoutes(httplib::Server& svr) {
    // Routes are still to be designed
    (void)svr;
}

std::vector<std::string> WebHandler::matcherNames() const {
    std::vector<std::string> names;
    for (const auto& m : matchers_)
        names.push_back(m->name());
    return names;
}

std::vector<MatchResult> WebHandler::locate(const cv::Mat& image,
                                            const std::vector<std::string>& selected) const {
    const Piece piece = matcher_utils::makePiece(image);
    std::vector<std::string> wanted;
    for (const std::string& s : selected)
        wanted.push_back(lowercase(s));

    std::vector<std::future<MatchResult>> pending;
    std::vector<MatchResult> results;
    for (const auto& m : matchers_)
        if (std::find(wanted.begin(), wanted.end(), lowercase(m->name())) != wanted.end()) {
            if (piece.bgr.empty()) {
                MatchResult r;
                r.matcherName = m->name();
                r.error       = "No piece found in the image";
                results.push_back(r);
            } else {
                pending.push_back(std::async(std::launch::async,
                                             [&m, &piece] { return m->match(piece); }));
            }
        }

    for (auto& p : pending)
        results.push_back(p.get());
    return results;
}
