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
// exposed at load time by onnx_graph::exposeClassifierInput. The embedding of
// the square inside the piece is compared by cosine similarity against
// embeddings of overlapping tiles of the artwork.
// Captures semantic content and tolerates lighting changes; the piece is tried
// in four 90° orientations. Unavailable when the model file is missing.
class MLMatcher : public Matcher {
public:
    // Loads the model, exposes its pooled feature vector as an output and opens
    // an ONNX Runtime session
    // In:     modelPath — MobileNetV2 .onnx file; empty = no model
    // Out:    a matcher, unavailable if the model couldn't be loaded
    // Errors: never throws; an empty path or a load failure (unreadable or
    //         malformed file, ONNX Runtime error) makes available() false and
    //         keeps the reason for match() to report
    explicit MLMatcher(const std::string& modelPath);

    // Embeds every tile in batches of 32: one per piece position, 0.8 piece
    // wide, when the piece count is known, else cell/2 and cell/3 tiles. This
    // dominates startup (~25 s at 5000 pieces). Does nothing when unavailable.
    // In:     artwork — full reference image, CV_8UC3 BGR
    //         grid    — cell layout; grid.pieces sets the tile size
    // Out:    none
    // Errors: may throw Ort::Exception / cv::Exception
    void prepare(const cv::Mat& artwork, const GridConfig& grid) override;

    // Embeds the piece's core square in 4 orientations and returns the tile
    // with the highest cosine similarity to any of them
    // In:     piece — normalized piece (uses core)
    // Out:    the tile's cell; confidence = that cosine similarity (high even
    //         when wrong — poorly calibrated)
    // Errors: error = the load failure reason (row/col = -1) when unavailable;
    //         may throw Ort::Exception if inference fails. Requires tiles from
    //         prepare() (none if the artwork is smaller than one piece).
    MatchResult match(const Piece& piece) const override;

    // In: none.  Out: "ML".  Errors: none.
    std::string name() const override { return "ML"; }

    // In: none.  Out: true when the model loaded.  Errors: none.
    bool available() const override { return session_ != nullptr; }

private:
    // Runs the network on a batch of images
    // In:     images — non-empty BGR images of any size (resized to 224×224,
    //                  ImageNet-normalized)
    // Out:    one L2-normalized embedding row (CV_32F; 1280 wide for MobileNetV2's
    //         pooled features) per image
    // Errors: throws Ort::Exception if inference fails; must only be called
    //         when available()
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
