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
//                            → {"saved": "C5_2.jpg"}
// In:     svr        — server to register on
//         grid       — validates cell labels (copied into the handlers)
//         samplesDir — where photos are listed and saved; created on first save
// Out:    none
// Errors: POST answers 400 {error} for a missing photo, an unknown cell or a
//         "replace" that isn't an existing sample of that cell, and 500 {error}
//         if the file can't be written (filesystem exceptions while replacing
//         or creating the directory also end as a 500, via httplib). A missing
//         samplesDir lists as empty.
void registerAdminRoutes(httplib::Server& svr, const GridConfig& grid, const std::string& samplesDir);
