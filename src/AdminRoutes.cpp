#include "AdminRoutes.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

const std::vector<std::string> IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"};

void sendError(httplib::Response& res, int httpStatus, const std::string& message) {
    res.status = httpStatus;
    res.set_content(nlohmann::json{{"error", message}}.dump(), "application/json");
}

std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isCellLabel(const GridConfig& grid, const std::string& label) {
    bool found = false;
    for (const GridCell& c : grid.allCells())
        found = found || c.label == label;
    return found;
}

// Sample files per cell label, by name: "C5.jpg" and "C5_2.jpg" both belong to C5
std::map<std::string, std::vector<std::string>> listSamples(const GridConfig& grid,
                                                            const std::string& samplesDir) {
    std::map<std::string, std::vector<std::string>> cells;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(samplesDir, ec)) {
        const std::string ext  = lowercase(entry.path().extension().string());
        const std::string stem = entry.path().stem().string();
        const std::string label = stem.substr(0, stem.find('_'));
        const bool isImage = std::find(IMAGE_EXTENSIONS.begin(), IMAGE_EXTENSIONS.end(), ext) !=
                             IMAGE_EXTENSIONS.end();
        if (entry.is_regular_file() && isImage && isCellLabel(grid, label))
            cells[label].push_back(entry.path().filename().string());
    }
    for (auto& [label, files] : cells)
        std::sort(files.begin(), files.end());
    return cells;
}

// First free name for a new sample of `label`: C5.jpg, else C5_2.jpg, C5_3.jpg, …
std::string nextName(const std::vector<std::string>& existing, const std::string& label,
                     const std::string& ext) {
    std::vector<std::string> stems;
    for (const std::string& f : existing)
        stems.push_back(fs::path(f).stem().string());
    std::string stem = label;
    for (int n = 2; std::find(stems.begin(), stems.end(), stem) != stems.end(); ++n)
        stem = label + "_" + std::to_string(n);
    return stem + ext;
}

} // namespace

void registerAdminRoutes(httplib::Server& svr, const GridConfig& grid, const std::string& samplesDir) {
    svr.Get("/api/admin/samples", [grid, samplesDir](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"cells", listSamples(grid, samplesDir)}}.dump(),
                        "application/json");
    });

    svr.Post("/api/admin/sample", [grid, samplesDir](const httplib::Request& req, httplib::Response& res) {
        const httplib::MultipartFormData photo = req.get_file_value("photo");
        const std::string label   = req.get_file_value("cell").content;
        const std::string replace = req.get_file_value("replace").content;
        std::string ext = lowercase(fs::path(photo.filename).extension().string());
        if (std::find(IMAGE_EXTENSIONS.begin(), IMAGE_EXTENSIONS.end(), ext) == IMAGE_EXTENSIONS.end())
            ext = ".jpg";

        const auto cells = listSamples(grid, samplesDir);
        const auto it = cells.find(label);
        const std::vector<std::string> existing = it != cells.end() ? it->second : std::vector<std::string>{};

        if (photo.content.empty()) {
            sendError(res, 400, "Missing photo");
        } else if (!isCellLabel(grid, label)) {
            sendError(res, 400, "Unknown cell: " + label);
        } else if (!replace.empty() &&
                   std::find(existing.begin(), existing.end(), replace) == existing.end()) {
            sendError(res, 400, "Not an existing sample of " + label + ": " + replace);
        } else {
            // A replacement keeps the old name unless the format changed (C5.png → C5.jpg)
            std::string name = nextName(existing, label, ext);
            if (!replace.empty()) {
                name = fs::path(replace).stem().string() + ext;
                fs::remove(fs::path(samplesDir) / replace);
            }
            fs::create_directories(samplesDir);
            std::ofstream out(fs::path(samplesDir) / name, std::ios::binary);
            out.write(photo.content.data(), static_cast<std::streamsize>(photo.content.size()));
            out.close();
            if (out) {
                std::cout << "[admin] Saved sample " << name << " (" << photo.content.size() << " bytes)\n";
                res.set_content(nlohmann::json{{"saved", name}}.dump(), "application/json");
            } else {
                sendError(res, 500, "Could not write " + name);
            }
        }
    });
}
