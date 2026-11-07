#pragma once
#include <string>
#include <httplib.h>
#include "GridConfig.hpp"

// Temporary admin API for collecting real sample photos (see web/admin.html).
// Only registered when argus runs with --samples-dir. Routes:
//   GET  /api/admin/samples  existing sample files per cell: {"cells": {"C5": ["C5.jpg", "C5_2.jpg"]}}
//   POST /api/admin/sample   multipart: "photo" = original image file, "cell" = label (e.g. "C5"),
//                            optional "replace" = an existing file of that cell to overwrite;
//                            without it the photo is added as the next free name (C5, C5_2, …)
void registerAdminRoutes(httplib::Server& svr, const GridConfig& grid, const std::string& samplesDir);
