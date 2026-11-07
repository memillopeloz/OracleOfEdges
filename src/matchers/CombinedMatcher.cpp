#include "CombinedMatcher.hpp"
#include <chrono>
#include <future>
#include "MatcherUtils.hpp"

namespace {

// Keypoint confidence from which its answer is trusted. On the real samples
// anything in 0.02–0.10 gives the same accuracy.
constexpr float KEYPOINT_GATE = 0.05f;

} // namespace

CombinedMatcher::CombinedMatcher(const Matcher& keypoint, const Matcher& tmpl)
    : keypoint_(keypoint), template_(tmpl) {}

void CombinedMatcher::prepare(const cv::Mat&, const GridConfig& grid) {
    grid_ = grid;
}

bool CombinedMatcher::available() const {
    return keypoint_.available() && template_.available();
}

MatchResult CombinedMatcher::match(const Piece& piece) const {
    const auto start = std::chrono::steady_clock::now();

    std::future<MatchResult> tmplFuture = std::async(std::launch::async,
                                                     [this, &piece] { return template_.match(piece); });
    const MatchResult kp = keypoint_.match(piece);
    const MatchResult tm = tmplFuture.get();

    const bool useKeypoint = kp.error.empty() && (kp.confidence >= KEYPOINT_GATE || !tm.error.empty());
    const MatchResult& chosen = useKeypoint ? kp : tm;
    const float confidence = useKeypoint ? kp.confidence : tm.confidence * KEYPOINT_GATE;

    MatchResult result = matcher_utils::makeResult(name(), grid_, chosen.row, chosen.col, confidence, start);
    result.error = chosen.error;
    return result;
}
