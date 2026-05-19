#pragma once
#include <cstdint>

// Bump when any baked-asset format (VXL3/MSH1/MSH2) changes. The viewer
// detects mismatch on load and re-runs voxelize.exe to regenerate.
constexpr uint32_t kAssetVersion = 3;
