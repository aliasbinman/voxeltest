#pragma once
#include "mesh.h"
#include <string>

bool LoadVoxScene(const char* path, Scene& out, std::string& err);
