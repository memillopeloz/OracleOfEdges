#pragma once
#include <string>
#include "GridConfig.hpp"
#include "Matcher.hpp"

// Keypoint when it is sure, Template otherwise. Keypoint's confidence is well
// calibrated (≈ 0 when wrong) but it often finds too little; Template always
// answers but can't tell a hit from a miss. Their hits overlap only partly, so
// gating on Keypoint's confidence beats either alone. Uses the two matchers it
// is given, already prepared by their owner, and runs them in parallel.
class CombinedMatcher : public Matcher {
public:
    // In:     keypoint, tmpl — the Keypoint and Template matchers; must outlive
    //                          this object and be prepared by their owner
    // Out:    none
    // Errors: none
    CombinedMatcher(const Matcher& keypoint, const Matcher& tmpl);

    // Keeps the grid for labelling; the wrapped matchers are prepared elsewhere.
    // In:     artwork — unused
    //         grid    — cell layout
    // Out:    none
    // Errors: none
    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;

    // Runs Keypoint and Template on the piece in parallel and returns Keypoint's
    // cell when its confidence is at least 0.05, else Template's.
    // In:     piece — normalized piece
    // Out:    the chosen cell; confidence = Keypoint's when it answers, else
    //         Template's score × 0.05, so any Keypoint answer ranks above any
    //         fallback
    // Errors: when Template fails, Keypoint's answer is used whatever its
    //         confidence; when both fail, Template's error (row/col = -1)
    MatchResult match(const Piece& piece) const override;

    // In: none.  Out: true when both wrapped matchers are available.  Errors: none.
    bool available() const override;

    // In: none.  Out: "Combined".  Errors: none.
    std::string name() const override { return "Combined"; }

private:
    const Matcher& keypoint_;
    const Matcher& template_;
    GridConfig     grid_;
};
