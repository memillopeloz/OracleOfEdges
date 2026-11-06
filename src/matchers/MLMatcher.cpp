#include "MLMatcher.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iterator>
#include <opencv2/imgproc.hpp>
#include "OnnxGraph.hpp"

namespace {

constexpr int    INPUT_SIZE    = 224;  // MobileNetV2 input resolution
constexpr int    BATCH_SIZE    = 32;
constexpr double TILE_SIDE     = 0.8;  // piece count known: tile ≈ the inside of a piece,
constexpr double TILE_STRIDE   = 1.0;  //   one per piece position (embedding is the slow part)
const std::vector<int> TILE_DIVISIONS = {2, 3}; // unknown: tiles of cell/2 and cell/3

// ImageNet normalization, RGB order
const cv::Scalar MEAN(0.485, 0.456, 0.406);
const cv::Scalar STD (0.229, 0.224, 0.225);

} // namespace

MLMatcher::MLMatcher(const std::string& modelPath) {
    if (modelPath.empty()) {
        unavailableReason_ = "ONNX model not found in config directory";
    } else {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "argus");
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Use the pooled features feeding the classifier rather than the class
            // scores: they describe texture and content far better.
            std::ifstream file(modelPath, std::ios::binary);
            const std::string original((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            const std::string model = onnx_graph::exposeClassifierInput(original, outputName_);
            session_ = std::make_unique<Ort::Session>(*env_, model.data(), model.size(), options);

            Ort::AllocatorWithDefaultOptions allocator;
            inputName_ = session_->GetInputNameAllocated(0, allocator).get();
            if (outputName_.empty())
                outputName_ = session_->GetOutputNameAllocated(0, allocator).get();
        } catch (const std::exception& e) {
            session_.reset();
            unavailableReason_ = std::string("Failed to load ONNX model: ") + e.what();
        }
    }
}

cv::Mat MLMatcher::embed(const std::vector<cv::Mat>& images) const {
    // Fill an NCHW float tensor: each image's channels are split straight into it
    const size_t plane = INPUT_SIZE * INPUT_SIZE;
    std::vector<float> blob(images.size() * 3 * plane);
    for (size_t n = 0; n < images.size(); ++n) {
        cv::Mat rgb, f;
        cv::resize(images[n], rgb, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_AREA);
        cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(f, CV_32F, 1.0 / 255);
        cv::subtract(f, MEAN, f);
        cv::divide(f, STD, f);

        std::vector<cv::Mat> channels;
        for (size_t c = 0; c < 3; ++c)
            channels.emplace_back(INPUT_SIZE, INPUT_SIZE, CV_32F, blob.data() + (n * 3 + c) * plane);
        cv::split(f, channels);
    }

    const std::array<int64_t, 4> shape = {static_cast<int64_t>(images.size()), 3, INPUT_SIZE, INPUT_SIZE};
    const Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memInfo, blob.data(), blob.size(), shape.data(), shape.size());

    const char* inNames[]  = {inputName_.c_str()};
    const char* outNames[] = {outputName_.c_str()};
    std::vector<Ort::Value> outputs =
        session_->Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 1);

    const std::vector<int64_t> outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const cv::Mat raw(static_cast<int>(outShape[0]), static_cast<int>(outShape[1]), CV_32F,
                      outputs[0].GetTensorMutableData<float>());

    cv::Mat embeddings = raw.clone();
    for (int r = 0; r < embeddings.rows; ++r)
        cv::normalize(embeddings.row(r), embeddings.row(r));
    return embeddings;
}

void MLMatcher::prepare(const cv::Mat& artwork, const GridConfig& grid) {
    grid_ = grid;
    tiles_.clear();
    tileEmbeddings_.release();

    if (available()) {
        const double side = grid.pieceSide(artwork.size());
        tiles_ = side > 0.0
               ? matcher_utils::gridTiles(grid, artwork.size(), std::max(1, cvRound(side * TILE_SIDE)),
                                          std::max(1, cvRound(side * TILE_STRIDE)))
               : matcher_utils::cellTiles(grid, artwork.size(), TILE_DIVISIONS);
        for (size_t first = 0; first < tiles_.size(); first += BATCH_SIZE) {
            const size_t last = std::min(first + BATCH_SIZE, tiles_.size());
            std::vector<cv::Mat> batch;
            for (size_t i = first; i < last; ++i)
                batch.push_back(artwork(tiles_[i].rect));
            tileEmbeddings_.push_back(embed(batch));
        }
    }
}

MatchResult MLMatcher::match(const Piece& piece) const {
    const auto start = std::chrono::steady_clock::now();
    MatchResult result;

    if (!available()) {
        result = matcher_utils::makeResult(name(), grid_, -1, -1, 0.0f, start);
        result.error = unavailableReason_;
    } else {
        const cv::Mat& core = piece.core;
        std::vector<cv::Mat> orientations = {core};
        for (int flag : {cv::ROTATE_90_CLOCKWISE, cv::ROTATE_180, cv::ROTATE_90_COUNTERCLOCKWISE}) {
            cv::Mat rotated;
            cv::rotate(core, rotated, flag);
            orientations.push_back(rotated);
        }

        // Rows are unit vectors, so the product holds cosine similarities
        const cv::Mat similarity = embed(orientations) * tileEmbeddings_.t();
        double    best = 0.0;
        cv::Point loc;
        cv::minMaxLoc(similarity, nullptr, &best, nullptr, &loc);

        const matcher_utils::Tile& tile = tiles_[loc.x];
        result = matcher_utils::makeResult(name(), grid_, tile.row, tile.col, static_cast<float>(best), start);
    }
    return result;
}
