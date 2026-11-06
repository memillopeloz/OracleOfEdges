// OracleOfEdges — Argus: locates a photographed puzzle piece within its artwork.
#include "GridConfig.hpp"
#include "WebHandler.hpp"
#include <httplib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <exception>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>

namespace fs = std::filesystem;

namespace {

// ── Defaults (override via CLI options) ──────────────────────────────────────
const std::string DEFAULT_CONFIG_DIR = "config";
const std::string DEFAULT_WEB_DIR    = "web";
const std::string GRID_FILE          = "grid.json";
const std::string MODEL_FILE         = "mobilenetv2.onnx";
constexpr int     DEFAULT_PORT       = 8080;

// ── Exit status (0 = success) ────────────────────────────────────────────────
enum Status : int {
    STATUS_OK           = 0,
    STATUS_BAD_ARGS     = 1,
    STATUS_CONFIG_ERROR = 2,
    STATUS_SERVER_ERROR = 3,
};

struct Options {
    std::string configDir = DEFAULT_CONFIG_DIR;
    std::string webDir    = DEFAULT_WEB_DIR;
    int         port      = DEFAULT_PORT;
    bool        showHelp  = false;
};

// ─────────────────────────────────────────────────────────────────────────────
void printUsage(const char* prog, std::ostream& out) {
    out << "Usage: " << prog << " [options]\n"
        << "\nOptions:\n"
        << "  --config-dir <dir>   Directory containing grid.json, artwork, model (default: config/)\n"
        << "  --web-dir <dir>      Directory with the web page (default: web/)\n"
        << "  --port <port>        HTTP listen port (default: " << DEFAULT_PORT << ")\n"
        << "  --help, -h           Show this message\n"
        << "\nConfig directory should contain:\n"
        << "  grid.json            Grid layout definition\n"
        << "  artwork.jpg/.png     Reference artwork image (or as set by artwork_path)\n"
        << "  mobilenetv2.onnx     MobileNetV2 model (ML matcher is unavailable without it)\n";
}

// Parses a TCP port (1–65535) into `port`
int parsePort(const std::string& text, int& port) {
    int status = STATUS_BAD_ARGS;
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed == text.size() && value > 0 && value <= 65535) {
            port   = value;
            status = STATUS_OK;
        }
    } catch (const std::exception&) {
        // Not a number — status stays STATUS_BAD_ARGS
    }
    if (status != STATUS_OK)
        std::cerr << "[error] Invalid port: " << text << "\n";
    return status;
}

int parseArgs(int argc, char* argv[], Options& opts) {
    int status = STATUS_OK;
    for (int i = 1; i < argc && status == STATUS_OK; ++i) {
        const std::string arg = argv[i];
        const bool hasValue   = i + 1 < argc;

        if (arg == "--help" || arg == "-h")           opts.showHelp  = true;
        else if (arg == "--config-dir" && hasValue)   opts.configDir = argv[++i];
        else if (arg == "--web-dir"    && hasValue)   opts.webDir    = argv[++i];
        else if (arg == "--port"       && hasValue)   status = parsePort(argv[++i], opts.port);
        else {
            std::cerr << "[error] Unknown or incomplete argument: " << arg << "\n";
            status = STATUS_BAD_ARGS;
        }
    }
    return status;
}

int loadGrid(const std::string& configDir, GridConfig& grid) {
    int status = STATUS_OK;
    const std::string gridPath = (fs::path(configDir) / GRID_FILE).string();
    std::cout << "[startup] Loading grid config: " << gridPath << "\n";
    try {
        grid = GridConfig::fromFile(gridPath);
        std::cout << "[startup] Grid: " << grid.rows << " rows × " << grid.cols << " cols\n";
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        status = STATUS_CONFIG_ERROR;
    }
    return status;
}

int loadArtwork(const std::string& configDir, const GridConfig& grid, cv::Mat& artwork) {
    int status = STATUS_CONFIG_ERROR;
    if (grid.artworkPath.empty()) {
        std::cerr << "[error] No artwork image found in " << configDir
                  << " (expected artwork.jpg / .png / .bmp, or set artwork_path in "
                  << GRID_FILE << ")\n";
    } else {
        std::cout << "[startup] Loading artwork: " << grid.artworkPath << "\n";
        artwork = cv::imread(grid.artworkPath, cv::IMREAD_COLOR);
        if (artwork.empty()) {
            std::cerr << "[error] Failed to read artwork image: " << grid.artworkPath << "\n";
        } else {
            std::cout << "[startup] Artwork size: " << artwork.cols << "×" << artwork.rows << "\n";
            status = STATUS_OK;
        }
    }
    return status;
}

// Path to the ONNX model, or empty if absent (the ML matcher then reports itself unavailable)
std::string findModel(const std::string& configDir) {
    std::string modelPath = (fs::path(configDir) / MODEL_FILE).string();
    if (fs::exists(modelPath)) {
        std::cout << "[startup] ONNX model: " << modelPath << "\n";
    } else {
        std::cout << "[startup] ONNX model not found at " << modelPath
                  << " — ML matcher will be unavailable.\n";
        modelPath.clear();
    }
    return modelPath;
}

void printBanner(int port) {
    std::cout << "\n========================================\n"
              << "  Argus is watching  (OracleOfEdges)\n"
              << "  Local:   http://localhost:" << port << "\n"
              << "  Network: http://<your-laptop-ip>:" << port << "\n"
              << "  (Find IP with: ip addr | grep 'inet ')\n"
              << "========================================\n\n";
}

// Blocks until the server stops
int runServer(const cv::Mat& artwork, const GridConfig& grid,
              const std::string& modelPath, const std::string& webDir, int port) {
    int status = STATUS_OK;
    httplib::Server svr;

    // Static page at /; the JSON API lives under /api (WebHandler). Checked
    // before the slow matcher setup so a bad path fails fast.
    if (!svr.set_mount_point("/", webDir)) {
        std::cerr << "[error] Web directory not found: " << webDir << "\n";
        status = STATUS_CONFIG_ERROR;
    } else {
        std::cout << "[startup] Initializing matchers (this may take a moment)...\n";
        WebHandler handler(artwork, grid, modelPath);
        handler.registerRoutes(svr);

        // Log every request before routing
        svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response&) {
            std::cout << "[http] " << req.method << " " << req.path << "\n";
            return httplib::Server::HandlerResponse::Unhandled;
        });

        printBanner(port);

        // 0.0.0.0 so phones on the local network can reach the server
        if (!svr.listen("0.0.0.0", port)) {
            std::cerr << "[error] Failed to bind to port " << port << "\n";
            status = STATUS_SERVER_ERROR;
        }
    }
    return status;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Options    opts;
    GridConfig grid;
    cv::Mat    artwork;

    int status = parseArgs(argc, argv, opts);

    if (status != STATUS_OK) {
        printUsage(argv[0], std::cerr);
    } else if (opts.showHelp) {
        printUsage(argv[0], std::cout);
    } else {
        status = loadGrid(opts.configDir, grid);
        if (status == STATUS_OK)
            status = loadArtwork(opts.configDir, grid, artwork);
        if (status == STATUS_OK)
            status = runServer(artwork, grid, findModel(opts.configDir), opts.webDir, opts.port);
    }

    return status;
}
