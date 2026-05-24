// Compilation unit for the zeux microprofile fork. Pulls in the full impl
// + the web HTML payload and the D3D11 GPU timer backend.
#define _CRT_SECURE_NO_WARNINGS
#define MICROPROFILE_IMPL
#define MICROPROFILE_ENABLED 1
#define MICROPROFILE_WEBSERVER 1
#define MICROPROFILE_GPU_TIMERS_D3D11 1
#include "microprofile.h"
