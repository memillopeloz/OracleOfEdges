#include "GridConfig.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

// Spreadsheet-style row labels: A … Z, AA, AB, …
std::vector<std::string> defaultRowLabels(int count) {
    std::vector<std::string> labels;
    for (int i = 0; i < count; ++i) {
        std::string label;
        for (int n = i + 1; n > 0; n = (n - 1) / 26)
            label.insert(label.begin(), static_cast<char>('A' + (n - 1) % 26));
        labels.push_back(label);
    }
    return labels;
}

// 1-based numeric column labels: 1, 2, …
std::vector<std::string> defaultColLabels(int count) {
    std::vector<std::string> labels;
    for (int i = 1; i <= count; ++i)
        labels.push_back(std::to_string(i));
    return labels;
}

// First artwork.<ext> found in dir, or empty
std::string findArtwork(const fs::path& dir) {
    std::string found;
    for (const char* ext : {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}) {
        const fs::path candidate = dir / (std::string("artwork") + ext);
        if (found.empty() && fs::exists(candidate))
            found = candidate.string();
    }
    return found;
}

} // namespace

GridConfig GridConfig::fromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open grid config: " + path);

    nlohmann::json j;
    f >> j;

    GridConfig cfg;
    cfg.rows = j.at("rows").get<int>();
    cfg.cols = j.at("cols").get<int>();
    if (cfg.rows <= 0 || cfg.cols <= 0)
        throw std::runtime_error("rows and cols must be positive");

    cfg.rowLabels = j.contains("row_labels")
                  ? j.at("row_labels").get<std::vector<std::string>>()
                  : defaultRowLabels(cfg.rows);
    cfg.colLabels = j.contains("col_labels")
                  ? j.at("col_labels").get<std::vector<std::string>>()
                  : defaultColLabels(cfg.cols);

    if ((int)cfg.rowLabels.size() != cfg.rows)
        throw std::runtime_error("row_labels count does not match rows");
    if ((int)cfg.colLabels.size() != cfg.cols)
        throw std::runtime_error("col_labels count does not match cols");

    cfg.pieces = j.value("pieces", 0);
    if (cfg.pieces < 0)
        throw std::runtime_error("pieces must not be negative");

    const fs::path configDir = fs::path(path).parent_path();
    cfg.artworkPath = j.contains("artwork_path")
                    ? (configDir / j.at("artwork_path").get<std::string>()).string()
                    : findArtwork(configDir);

    return cfg;
}

std::string GridConfig::coordLabel(int row, int col) const {
    std::string label = "?";
    if (row >= 0 && row < rows && col >= 0 && col < cols)
        label = rowLabels[row] + colLabels[col];
    return label;
}

cv::Rect GridConfig::cellRect(int row, int col, const cv::Size& artworkSize) const {
    const int x0 =  col      * artworkSize.width  / cols;
    const int x1 = (col + 1) * artworkSize.width  / cols;
    const int y0 =  row      * artworkSize.height / rows;
    const int y1 = (row + 1) * artworkSize.height / rows;
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

double GridConfig::pieceSide(const cv::Size& artworkSize) const {
    double side = 0.0;
    if (pieces > 0)
        side = std::sqrt(static_cast<double>(artworkSize.area()) / pieces);
    return side;
}

std::vector<GridCell> GridConfig::allCells() const {
    std::vector<GridCell> cells;
    cells.reserve(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            cells.push_back({r, c, coordLabel(r, c)});
    return std::move(cells);
}
