cmake_minimum_required(VERSION 3.16)
project(OracleOfEdges VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # build/compile_commands.json for editor IntelliSense

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(OpenCV 4 REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)

# cpp-httplib (header-only HTTP server)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(httplib)

# nlohmann/json (header-only JSON parser)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ONNX Runtime (required — for ML matcher)
# Prebuilt Linux x64 release is downloaded automatically. To use a local install
# instead, pass: cmake -DONNXRUNTIME_DIR=/path/to/onnxruntime ..
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime root directory (empty = download)")

if(ONNXRUNTIME_DIR STREQUAL "")
    FetchContent_Declare(
        onnxruntime
        URL      https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
        URL_HASH SHA256=67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime)
    set(ONNXRUNTIME_DIR "${onnxruntime_SOURCE_DIR}")
endif()

set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIBS         "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")

# ── Targets ───────────────────────────────────────────────────────────────────
# Everything but main() lives in argus_core, shared by the server and the bench.
add_library(argus_core STATIC
    src/AdminRoutes.cpp
    src/GridConfig.cpp
    src/WebHandler.cpp
    src/matchers/MatcherUtils.cpp
    src/matchers/TemplateMatcher.cpp
    src/matchers/HistogramMatcher.cpp
    src/matchers/KeypointMatcher.cpp
    src/matchers/CombinedMatcher.cpp
    src/matchers/MLMatcher.cpp
    src/matchers/OnnxGraph.cpp
)

target_include_directories(argus_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${httplib_SOURCE_DIR}
    ${nlohmann_json_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(argus_core PUBLIC
    ${OpenCV_LIBS}
    Threads::Threads
    ${ONNXRUNTIME_LIBS}
)

# The server
add_executable(argus src/main.cpp)
target_link_libraries(argus PRIVATE argus_core)
# Default config/ and web/ are the repo's own, so argus runs from any directory
target_compile_definitions(argus PRIVATE ARGUS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Matcher accuracy/speed benchmark on simulated pieces (see tools/bench.cpp)
add_executable(argus_bench tools/bench.cpp)
target_link_libraries(argus_bench PRIVATE argus_core)
target_compile_definitions(argus_bench PRIVATE ARGUS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ── Defaults ──────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV ver  : ${OpenCV_VERSION}")
