#include "WebHandler.hpp"
#include <algorithm>
#include <cctype>
#include <future>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include "matchers/CombinedMatcher.hpp"
#include "matchers/HistogramMatcher.hpp"
#include "matchers/KeypointMatcher.hpp"
#include "matchers/MLMatcher.hpp"
#include "matchers/MatcherUtils.hpp"
#include "matchers/TemplateMatcher.hpp"

namespace {

// Longest side of the artwork overview shown on the page
constexpr int OVERVIEW_MAX_DIM = 1600;

// Fills an error response with the JSON body {"error": message}
// In:     res        — response to fill
//         httpStatus — e.g. 400
//         message    — human-readable reason, shown by the page
// Out:    none (res is modified)
// Errors: none
void sendError(httplib::Response& res, int httpStatus, const std::string& message) {
    res.status = httpStatus;
    res.set_content(nlohmann::json{{"error", message}}.dump(), "application/json");
}

// ASCII lower-case copy, for case-insensitive matcher names
// In:     s — any string
// Out:    s with A–Z turned into a–z
// Errors: none
std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

} // namespace

WebHandler::WebHandler(cv::Mat artwork, GridConfig grid, std::string modelPath)
    : artwork_(std::move(artwork)), grid_(std::move(grid)) {
    auto tmpl     = std::make_unique<TemplateMatcher>();
    auto keypoint = std::make_unique<KeypointMatcher>();
    auto combined = std::make_unique<CombinedMatcher>(*keypoint, *tmpl);
    matchers_.push_back(std::move(tmpl));
    matchers_.push_back(std::make_unique<MLMatcher>(modelPath));
    matchers_.push_back(std::make_unique<HistogramMatcher>());
    matchers_.push_back(std::move(keypoint));
    matchers_.push_back(std::move(combined)); // prepare() only keeps the grid

    std::vector<std::future<void>> pending;
    for (auto& m : matchers_)
        pending.push_back(std::async(std::launch::async,
                                     [&m, this] { m->prepare(artwork_, grid_); }));
    for (auto& p : pending)
        p.get();

    cv::imencode(".jpg", matcher_utils::limitSize(artwork_, OVERVIEW_MAX_DIM), overviewJpeg_,
                 {cv::IMWRITE_JPEG_QUALITY, 85});
}

// Routes are documented in WebHandler.hpp
void WebHandler::registerRoutes(httplib::Server& svr) {
    svr.Get("/api/info", [this](const httplib::Request&, httplib::Response& res) {
        nlohmann::json matchers = nlohmann::json::array();
        for (const auto& m : matchers_)
            matchers.push_back({{"name", m->name()}, {"available", m->available()}});
        const nlohmann::json info = {
            {"rows", grid_.rows},           {"cols", grid_.cols},
            {"rowLabels", grid_.rowLabels}, {"colLabels", grid_.colLabels},
            {"matchers", matchers},
        };
        res.set_content(info.dump(), "application/json");
    });

    svr.Get("/api/artwork", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(reinterpret_cast<const char*>(overviewJpeg_.data()),
                        overviewJpeg_.size(), "image/jpeg");
    });

    svr.Post("/api/locate", [this](const httplib::Request& req, httplib::Response& res) {
        std::vector<std::string> selected;
        for (const auto& field : req.get_file_values("matcher"))
            selected.push_back(field.content);

        const std::string& data = req.get_file_value("piece").content;
        cv::Mat image;
        if (!data.empty())
            image = cv::imdecode(std::vector<uchar>(data.begin(), data.end()),
                                 cv::IMREAD_UNCHANGED);

        if (image.empty()) {
            sendError(res, 400, "Missing or unreadable piece image");
        } else if (selected.empty()) {
            sendError(res, 400, "No matcher selected");
        } else {
            nlohmann::json results = nlohmann::json::array();
            for (const MatchResult& r : locate(image, selected))
                results.push_back({
                    {"matcher", r.matcherName}, {"coord", r.coord},
                    {"row", r.row},             {"col", r.col},
                    {"confidence", r.confidence}, {"timeMs", r.timeMs},
                    {"error", r.error},
                });
            res.set_content(nlohmann::json{{"results", results}}.dump(), "application/json");
        }
    });
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
