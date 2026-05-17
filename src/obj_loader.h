#pragma once
#include "obj_scene.h"
#include <string>

bool LoadObjScene(const char* objPath, ObjScene& out, std::string& err);
