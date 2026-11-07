#pragma once
#include <string>

// Minimal in-memory edits of a serialized ONNX model (protobuf wire format),
// so no ONNX tooling is needed at build or run time.
namespace onnx_graph {

// Adds the data input of a classification network's final Gemm (fully
// connected) layer — the pooled feature vector — as an extra float graph output
// In:     model       — serialized ONNX ModelProto bytes
// Out:    the edited model; featureName = that tensor's name. Without a Gemm
//         node, the model unchanged and featureName empty.
// Errors: throws std::runtime_error if the bytes are not well-formed protobuf
//         (bad varint, unknown wire type, length past the end). The graph's
//         semantics aren't checked; ONNX Runtime validates them on load.
std::string exposeClassifierInput(const std::string& model, std::string& featureName);

} // namespace onnx_graph
