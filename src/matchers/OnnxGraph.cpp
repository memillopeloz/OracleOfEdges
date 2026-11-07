#include "OnnxGraph.hpp"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace onnx_graph {

namespace {

// Field numbers from onnx.proto
constexpr uint32_t MODEL_GRAPH        = 7;   // ModelProto.graph
constexpr uint32_t GRAPH_NODE         = 1;   // GraphProto.node
constexpr uint32_t GRAPH_OUTPUT       = 12;  // GraphProto.output
constexpr uint32_t NODE_INPUT         = 1;   // NodeProto.input
constexpr uint32_t NODE_OP_TYPE       = 4;   // NodeProto.op_type
constexpr uint32_t VALUE_INFO_NAME    = 1;   // ValueInfoProto.name
constexpr uint32_t VALUE_INFO_TYPE    = 2;   // ValueInfoProto.type
constexpr uint32_t TYPE_TENSOR        = 1;   // TypeProto.tensor_type
constexpr uint32_t TENSOR_ELEM_TYPE   = 1;   // TypeProto.Tensor.elem_type
constexpr uint64_t ELEM_FLOAT         = 1;   // TensorProto.DataType.FLOAT

// Protobuf wire types
constexpr uint32_t WIRE_VARINT = 0;
constexpr uint32_t WIRE_FIXED64 = 1;
constexpr uint32_t WIRE_LENGTH = 2;
constexpr uint32_t WIRE_FIXED32 = 5;

struct Field {
    uint32_t         number;
    uint32_t         wireType;
    std::string_view payload; // contents of a length-delimited field
    std::string_view raw;     // the whole encoded field, key included
};

// Reads one base-128 varint
// In:     bytes — encoded data
//         pos   — offset of the varint; advanced past it
// Out:    the decoded value
// Errors: throws std::runtime_error if the data ends mid-varint or the varint
//         is longer than 64 bits
uint64_t readVarint(std::string_view bytes, size_t& pos) {
    uint64_t value = 0;
    int      shift = 0;
    bool     more  = true;
    while (more) {
        if (pos >= bytes.size() || shift > 63)
            throw std::runtime_error("Malformed ONNX model (varint)");
        const auto byte = static_cast<uint8_t>(bytes[pos++]);
        value |= static_cast<uint64_t>(byte & 0x7f) << shift;
        shift += 7;
        more   = (byte & 0x80) != 0;
    }
    return value;
}

// Splits a protobuf message into its top-level fields, without decoding them
// In:     bytes — an encoded message (views into it are returned, so it must
//                 outlive the result)
// Out:    the fields in order
// Errors: throws std::runtime_error on a bad varint, an unsupported wire type
//         (groups) or a length running past the end
std::vector<Field> parseFields(std::string_view bytes) {
    std::vector<Field> fields;
    size_t pos = 0;
    while (pos < bytes.size()) {
        const size_t   start = pos;
        const uint64_t key   = readVarint(bytes, pos);
        Field f{static_cast<uint32_t>(key >> 3), static_cast<uint32_t>(key & 7), {}, {}};
        size_t length = 0;
        if (f.wireType == WIRE_VARINT)       readVarint(bytes, pos);
        else if (f.wireType == WIRE_FIXED64) length = 8;
        else if (f.wireType == WIRE_FIXED32) length = 4;
        else if (f.wireType == WIRE_LENGTH)  length = static_cast<size_t>(readVarint(bytes, pos));
        else throw std::runtime_error("Malformed ONNX model (wire type)");

        if (length > bytes.size() - pos)
            throw std::runtime_error("Malformed ONNX model (length)");
        f.payload = bytes.substr(pos, length);
        pos += length;
        f.raw = bytes.substr(start, pos - start);
        fields.push_back(f);
    }
    return fields;
}

// Encodes a base-128 varint
// In:     value — any value
// Out:    1–10 bytes
// Errors: none
std::string encodeVarint(uint64_t value) {
    std::string out;
    do {
        uint8_t byte = value & 0x7f;
        value >>= 7;
        if (value != 0)
            byte |= 0x80;
        out.push_back(static_cast<char>(byte));
    } while (value != 0);
    return out;
}

// Encodes a varint field (key + value)
// In:     number — field number
//         value  — field value
// Out:    the encoded field
// Errors: none
std::string varintField(uint32_t number, uint64_t value) {
    return encodeVarint(static_cast<uint64_t>(number) << 3 | WIRE_VARINT) + encodeVarint(value);
}

// Encodes a length-delimited field (key + length + payload): strings,
// bytes and nested messages
// In:     number  — field number
//         payload — field contents
// Out:    the encoded field
// Errors: none
std::string lengthField(uint32_t number, std::string_view payload) {
    return encodeVarint(static_cast<uint64_t>(number) << 3 | WIRE_LENGTH)
         + encodeVarint(payload.size()) + std::string(payload);
}

} // namespace

std::string exposeClassifierInput(const std::string& model, std::string& featureName) {
    featureName.clear();
    const std::vector<Field> top = parseFields(model);

    std::string_view graph;
    for (const Field& f : top)
        if (f.number == MODEL_GRAPH && f.wireType == WIRE_LENGTH)
            graph = f.payload;

    // Data input of the last Gemm node
    for (const Field& node : parseFields(graph))
        if (node.number == GRAPH_NODE && node.wireType == WIRE_LENGTH) {
            std::string_view opType, firstInput;
            for (const Field& f : parseFields(node.payload)) {
                if (f.number == NODE_OP_TYPE)
                    opType = f.payload;
                if (f.number == NODE_INPUT && firstInput.empty())
                    firstInput = f.payload;
            }
            if (opType == "Gemm")
                featureName = std::string(firstInput);
        }

    std::string result = model;
    if (!featureName.empty()) {
        const std::string tensorType = lengthField(TYPE_TENSOR, varintField(TENSOR_ELEM_TYPE, ELEM_FLOAT));
        const std::string valueInfo  = lengthField(VALUE_INFO_NAME, featureName)
                                     + lengthField(VALUE_INFO_TYPE, tensorType);
        const std::string newGraph   = std::string(graph) + lengthField(GRAPH_OUTPUT, valueInfo);

        result.clear();
        for (const Field& f : top)
            result += f.number == MODEL_GRAPH ? lengthField(MODEL_GRAPH, newGraph) : std::string(f.raw);
    }
    return result;
}

} // namespace onnx_graph
