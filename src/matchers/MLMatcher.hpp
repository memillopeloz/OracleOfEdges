#pragma once
#include <memory>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include "GridConfig.hpp"
#include "Matcher.hpp"
#include "MatcherUtils.hpp"

// MobileNetV2 (ONNX) embeddings — the pooled features before the classifier,
// exposed at load time by onnx_graph::exposeClassifierInput. The embedding of the square inside the piece is
// compared by cosine similarity against embeddings of overlapping tiles of the artwork.
// Captures semantic content and tolerates lighting changes; the piece is tried
// in four 90° orientations. Unavailable when the model file is missing.
class MLMatcher : public Matcher {
public:
    explicit MLMatcher(const std::string& modelPath);

    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;
    MatchResult match(const Piece& piece) const override;
    std::string name() const override { return "ML"; }
    bool available() const override { return session_ != nullptr; }

private:
    // One L2-normalized embedding row (CV_32F) per input BGR image
    cv::Mat embed(const std::vector<cv::Mat>& images) const;

    std::string                      unavailableReason_;
    std::unique_ptr<Ort::Env>        env_;
    std::unique_ptr<Ort::Session>    session_;   // Run() is thread-safe
    std::string                      inputName_;
    std::string                      outputName_;
    GridConfig                       grid_;
    std::vector<matcher_utils::Tile> tiles_;
    cv::Mat                          tileEmbeddings_; // one row per tile
};
