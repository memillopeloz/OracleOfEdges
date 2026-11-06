#pragma once
#include <string>

// Minimal in-memory edits of a serialized ONNX model (protobuf wire format),
// so no ONNX tooling is needed at build or run time.
namespace onnx_graph {

// For a classification network, returns the model with the data input of its
// final Gemm (fully connected) layer — the pooled feature vector — added as an
// extra float graph output, and sets featureName to that tensor's name.
// If there is no Gemm, returns the model unchanged and clears featureName.
// Throws std::runtime_error if the bytes are not a well-formed protobuf.
std::string exposeClassifierInput(const std::string& model, std::string& featureName);

} // namespace onnx_graph
