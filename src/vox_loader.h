#pragma once
#include "mesh.h"
#include <string>

bool LoadVoxScene(const char* path, Scene& out, std::string& err);

// Re-reads the raw .vox, runs LZ4 + sub-cluster + Huffman analysis, writes
// results into out.comp* fields + colorHistogram. Heavy — invoke on demand.
bool RunCompressionAnalysis(const char* voxPath, Scene& out, std::string& err);
