#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>

#include "renderer.h"
#include "camera.h"
#include "vox_loader.h"
#include "asset_version.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"

#include "microprofile.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace {

constexpr const char* kSettingsPath = "voxeltest.settings";

struct Settings {
    std::string lastVox;
    int         adapterIdx = -1;     // -1 = system default
};

Settings LoadSettings()
{
    Settings s;
    std::ifstream f(kSettingsPath);
    if (!f) return s;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("lastVox=", 0) == 0) s.lastVox = line.substr(8);
        else if (line.rfind("adapter=", 0) == 0) s.adapterIdx = std::atoi(line.c_str() + 8);
    }
    return s;
}
void SaveSettings(const Settings& s)
{
    std::ofstream f(kSettingsPath, std::ios::trunc);
    if (!f) return;
    f << "lastVox=" << s.lastVox << "\n";
    f << "adapter=" << s.adapterIdx << "\n";
}

// Compat wrappers — call sites pass a path-only or want the path only.
std::string LoadLastVoxFromSettings() { return LoadSettings().lastVox; }
void SaveLastVoxToSettings(const std::string& path)
{
    Settings s = LoadSettings();
    s.lastVox = path;
    SaveSettings(s);
}

// Enumerate assets/*.vox + assets/*.lw, sorted alphabetically.
std::vector<std::string> DiscoverDatasets()
{
    std::vector<std::string> out;
    std::error_code ec;
    if (!std::filesystem::is_directory("assets", ec)) return out;
    for (const auto& e : std::filesystem::directory_iterator("assets", ec)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension();
        if (ext == ".vox" || ext == ".lw") {
            out.push_back(e.path().generic_string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}
static inline bool IsLwPath(const std::string& p)
{
    return p.size() >= 3 && p.compare(p.size() - 3, 3, ".lw") == 0;
}

void SetCwdToProjectRoot()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    wchar_t* slash = wcsrchr(exePath, L'\\');
    if (slash) *slash = 0;
    SetCurrentDirectoryW(exePath);
    for (int i = 0; i < 8; ++i) {
        if (GetFileAttributesW(L"shaders\\voxel.hlsl") != INVALID_FILE_ATTRIBUTES) return;
        SetCurrentDirectoryW(L"..");
    }
}

struct AppState {
    Renderer renderer;
    Camera   camera;
    ShadingMode mode = ShadingMode::Lit;
    RenderTech  tech    = RenderTech::PolyAxis;  // close tech
    RenderTech  techFar = RenderTech::Splat;     // far tech
    bool        closeEnabled = true;
    bool        farEnabled   = true;
    float    sunPitchDeg = 60.0f;
    float    sunYawDeg   = 63.0f;
    float    sunIntensityEV = 0.0f;     // log2 stops; linear = 2^EV
    bool     sunShadows  = true;
    int      shadowCascades  = 1;       // 1..4 (only cascade 0 active)
    int      shadowMapSizeIdx = 2;      // index into {512,1024,2048,4096}
    float    shadowBias  = 0.00015f;
    bool     shadowCullFront = false;
    bool     shadowForceRebuild = false;
    int      shadowLodIdx = 0;  // 0=Auto, 1..4 = L0..L3
    bool     shadowBlur = false;
    float    exposureEV     = 0.0f;     // log2 stops; linear = 2^EV
    float    roughness      = 0.6f;
    bool     vsync = false;
    int      gridSize = 1;
    bool     taa = true;
    PointLighting pointLight = PointLighting::Complex;
    PointLod pointLod = PointLod::Auto;
    float    pointLodScale = 1.0f;
    bool     splatFilter = true;
    bool     splatDilate2Pass = false;
    int      splatRadius = 3;       // CS dilation half-window in pixels
    int      fogMode = 1;        // 0 = off, 1 = depth (drives effFogDensity gating)
    float    fogDensity = 0.0004f;
    float    fogColor[3] = { 0.55f, 0.60f, 0.70f };
    float    heightFogDensity = 4.5f;        // 0 = off
    float    heightFogFalloff = 0.05f;       // exp falloff per unit height
    float    heightFogStart   = -6.5f;        // world Y of fog ground plane
    bool     rmbDown = false;
    POINT    lastMouse = { 0, 0 };
    bool     keys[256] = {};
    bool     wantQuit = false;
    bool     sceneReady = false;
    std::string loadStatus = "Loading...";
    std::vector<std::string> graphicsAdapters;  // populated at startup
    int      adapterIdx = -1;                   // selected adapter idx (-1=default)
    int      activeAdapterIdx = -1;             // adapter actually in use this run
    bool     lwShowBounds = false;              // debug: draw per-chunk AABBs
    std::string currentVoxPath;
    std::vector<std::string> datasetPaths;   // discovered assets/*.vox at startup
    int         datasetIdx = 0;              // index into datasetPaths
    Scene       pendingScene;
    lw::World   pendingLwWorld;
    bool        pendingIsLw = false;
    // Snapshot kept after upload so on-demand compression analysis can run
    // without holding the full pointVertices/ao6 buffers.
    Scene       compScene;
    std::string compStatus;
    bool        everLoaded = false;
    std::atomic<bool> reloadRequested{ false };
    std::atomic<bool> loadDone{ false };
    std::atomic<bool> loadOk{ false };
    std::string loadErr;
    float    bgColor[4] = { 0.10f, 0.12f, 0.16f, 1.0f };

    // perf
    double   cpuFrameMs = 0.0;
    float    fpsAvg = 0.0f;
    float    fpsHist[120] = {};
    int      fpsHistIdx = 0;

    // window visibility
    bool     showControls = true;
    bool     showFps      = true;
    bool     showStats    = true;

};

AppState g_app;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) {
            g_app.renderer.Resize(LOWORD(lp), HIWORD(lp));
        }
        return 0;
    case WM_CLOSE:
        g_app.wantQuit = true;
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (wp < 256) g_app.keys[wp] = true;
        if (wp == VK_ESCAPE) g_app.wantQuit = true;
        return 0;
    case WM_KEYUP:
        if (wp < 256) g_app.keys[wp] = false;
        return 0;
    case WM_RBUTTONDOWN:
        g_app.rmbDown = true;
        GetCursorPos(&g_app.lastMouse);
        SetCapture(hwnd);
        ShowCursor(FALSE);
        return 0;
    case WM_RBUTTONUP:
        g_app.rmbDown = false;
        ReleaseCapture();
        ShowCursor(TRUE);
        return 0;
    case WM_MOUSEWHEEL: {
        if (ImGui::GetIO().WantCaptureMouse) return 0;
        short delta = (short)HIWORD(wp);
        float notches = (float)delta / (float)WHEEL_DELTA;
        g_app.camera.moveSpeed *= powf(1.2f, notches);
        g_app.camera.moveSpeed = std::clamp(g_app.camera.moveSpeed, 0.05f, 10000.0f);
        return 0;
    }
    case WM_MOUSEMOVE:
        if (g_app.rmbDown) {
            POINT cur; GetCursorPos(&cur);
            int dx = cur.x - g_app.lastMouse.x;
            int dy = cur.y - g_app.lastMouse.y;
            g_app.camera.yaw   += dx * g_app.camera.lookSens;
            g_app.camera.pitch -= dy * g_app.camera.lookSens;
            const float lim = 1.55334f;
            g_app.camera.pitch = std::clamp(g_app.camera.pitch, -lim, lim);
            SetCursorPos(g_app.lastMouse.x, g_app.lastMouse.y);
        }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

void UpdateCamera(float dt)
{
    if (ImGui::GetIO().WantTextInput) return;
    Camera& c = g_app.camera;
    float speed = c.moveSpeed * dt;

    hlslpp::float3 f = c.forward();
    hlslpp::float3 r = c.right();
    hlslpp::float3 mv(0.0f, 0.0f, 0.0f);
    if (g_app.keys['W']) mv += f;
    if (g_app.keys['S']) mv -= f;
    if (g_app.keys['D']) mv += r;
    if (g_app.keys['A']) mv -= r;
    float vy = 0.0f;
    if (g_app.keys['E'] || g_app.keys[VK_SPACE])   vy += 1.0f;
    if (g_app.keys['Q'] || g_app.keys[VK_CONTROL]) vy -= 1.0f;
    mv += hlslpp::float3(0.0f, vy, 0.0f);

    float len = (float)hlslpp::length(mv);
    if (len > 1e-5f) {
        c.position += hlslpp::normalize(mv) * speed;
    }
}

void FrameMenuBar()
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Windows")) {
            ImGui::MenuItem("Controls", nullptr, &g_app.showControls);
            ImGui::MenuItem("FPS",      nullptr, &g_app.showFps);
            ImGui::MenuItem("Stats",    nullptr, &g_app.showStats);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void FrameFpsWindow()
{
    g_app.fpsAvg = 1000.0f / (g_app.cpuFrameMs > 0.001 ? (float)g_app.cpuFrameMs : 16.0f);
    g_app.fpsHist[g_app.fpsHistIdx] = g_app.fpsAvg;
    g_app.fpsHistIdx = (g_app.fpsHistIdx + 1) % IM_ARRAYSIZE(g_app.fpsHist);

    if (!g_app.showFps) return;
    if (!ImGui::Begin("FPS", &g_app.showFps)) { ImGui::End(); return; }
    ImGui::Text("CPU frame: %.2f ms (%.0f FPS)", g_app.cpuFrameMs, g_app.fpsAvg);
    ImGui::PlotLines("FPS", g_app.fpsHist, IM_ARRAYSIZE(g_app.fpsHist),
                     g_app.fpsHistIdx, nullptr, 0.0f, 240.0f, ImVec2(0, 60));
    ImGui::End();
}

void FrameStatsWindow()
{
    if (!g_app.showStats) return;
    if (!ImGui::Begin("Stats", &g_app.showStats)) { ImGui::End(); return; }

    auto mb = [](uint64_t b) { return (double)b / (1024.0 * 1024.0); };

    ImGui::Text("Scene");
    {
        size_t totalChunks = g_app.renderer.DrawCount() * (size_t)g_app.gridSize * (size_t)g_app.gridSize;
        ImGui::Text("  Chunks: %zu  drawn: %u  (%.1f%%)",
                    totalChunks,
                    g_app.renderer.LastDrawnCount(),
                    totalChunks ? 100.0f * g_app.renderer.LastDrawnCount() / (float)totalChunks : 0.0f);
    }
    ImGui::Text("  Points drawn: %llu", (unsigned long long)g_app.renderer.LastPointCount());

    ImGui::Separator();
    ImGui::Text("Points (per LOD)");
    {
        const uint64_t bpp = sizeof(Vertex) + sizeof(uint32_t); // 12 + 4 = 16 B/pt
        const uint64_t l0 = g_app.renderer.PointCountL0();
        const uint64_t l1 = g_app.renderer.PointCountL1();
        const uint64_t l2 = g_app.renderer.PointCountL2();
        const uint64_t l3 = g_app.renderer.PointCountL3();
        const uint64_t tot = l0 + l1 + l2 + l3;
        ImGui::Text("  L0 (1 per voxel): %10llu  (%.2f MB)", (unsigned long long)l0, mb(l0 * bpp));
        ImGui::Text("  L1 (2x2x2):       %10llu  (%.2f MB)", (unsigned long long)l1, mb(l1 * bpp));
        ImGui::Text("  L2 (4x4x4):       %10llu  (%.2f MB)", (unsigned long long)l2, mb(l2 * bpp));
        ImGui::Text("  L3 (8x8x8):       %10llu  (%.2f MB)", (unsigned long long)l3, mb(l3 * bpp));
        ImGui::Text("  TOTAL:            %10llu  (%.2f MB)", (unsigned long long)tot, mb(g_app.renderer.PointBytes()));
    }

    ImGui::Separator();
    ImGui::Text("GPU Memory");
    {
        const uint64_t pb = g_app.renderer.PointBytes();
        const uint64_t rt = g_app.renderer.SplatRtBytes();
        ImGui::Text("  Point VB + AO6:   %7.2f MB", mb(pb));
        ImGui::Text("  Splat RTs/UAVs:   %7.2f MB", mb(rt));
        ImGui::Text("  TOTAL:            %7.2f MB", mb(pb + rt));
    }

    ImGui::Separator();
    ImGui::Text("CPU Memory (world)");
    {
        const uint64_t subsB = g_app.renderer.SubsBytes();
        const uint64_t compSubsB = g_app.compScene.subs.size() * sizeof(SubMesh);
        const uint64_t compHistB = g_app.compScene.colorHistogram.size()
                                 * sizeof(std::pair<uint32_t, uint64_t>);
        const uint64_t total = subsB + compSubsB + compHistB;
        ImGui::Text("  Renderer subs_:        %7.2f MB  (%zu chunks)",
                    mb(subsB), (size_t)g_app.renderer.DrawCount());
        ImGui::Text("  Compress-scene mirror: %7.2f MB  (subs %zu + hist %zu)",
                    mb(compSubsB + compHistB),
                    g_app.compScene.subs.size(), g_app.compScene.colorHistogram.size());
        ImGui::Text("  TOTAL:                 %7.2f MB", mb(total));
    }

    ImGui::Separator();
    ImGui::Text("Disk Compression Sim (L0 stream)");
    if (ImGui::Button("Run analysis") && !g_app.currentVoxPath.empty()) {
        std::string err;
        if (RunCompressionAnalysis(g_app.currentVoxPath.c_str(), g_app.compScene, err)) {
            g_app.compStatus = "done";
        } else {
            g_app.compStatus = "error: " + err;
        }
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(g_app.compStatus.c_str());
    {
        const Scene& cs = g_app.compScene;
        const uint64_t raw   = cs.compRawBytes;
        const uint64_t pos   = cs.compPosBytes;
        const uint64_t mask  = cs.compMaskBytes;
        const uint64_t ao    = cs.compAoBytes;
        const uint64_t pal   = cs.compPaletteBytes;
        const uint64_t cpal  = cs.compColorPalIdxBytes;
        const uint64_t chuff = cs.compColorHuffBytes;
        const uint64_t fixed = pos + mask + ao;
        const uint64_t totPal  = fixed + cpal  + pal;
        const uint64_t totHuff = fixed + chuff + pal;
        auto ratio = [&](uint64_t t) { return t ? (double)raw / (double)t : 0.0; };
        if (raw == 0) {
            ImGui::TextUnformatted("  (no results yet — click Run analysis)");
        } else {
            ImGui::Text("  chunkDim %u, posBits/axis %u",
                        cs.compChunkDim, cs.compPosBitsPerAxis);
            ImGui::Text("  Raw (in-mem):   %7.2f MB", mb(raw));
            ImGui::Text("  Pos stream:     %7.2f MB", mb(pos));
            ImGui::Text("  visMask stream: %7.2f MB", mb(mask));
            ImGui::Text("  AO stream:      %7.2f MB", mb(ao));
            ImGui::Text("  Color (pal8):   %7.2f MB  + palette %llu B", mb(cpal), (unsigned long long)pal);
            ImGui::Text("  Color (huff):   %7.2f MB  + palette %llu B", mb(chuff), (unsigned long long)pal);
            ImGui::Separator();
            ImGui::Text("  TOTAL pal8:     %7.2f MB  (%.2fx vs raw)", mb(totPal),  ratio(totPal));
            ImGui::Text("  TOTAL huff:     %7.2f MB  (%.2fx vs raw)", mb(totHuff), ratio(totHuff));

            const uint64_t subPos = cs.compSubclusterPosBytes;
            const uint32_t subDim = cs.compSubclusterDim;
            const uint64_t totSub = (fixed - pos) + subPos + cpal + pal;
            ImGui::Separator();
            ImGui::Text("Sub-cluster pos (S=%u): %.2f MB  (raw pos %.2f MB)",
                        subDim, mb(subPos), mb(pos));
            ImGui::Text("  TOTAL pal8+sub: %7.2f MB  (%.2fx vs raw)", mb(totSub), ratio(totSub));

            const uint64_t lzP = cs.compLz4PosBytes;
            const uint64_t lzM = cs.compLz4MaskBytes;
            const uint64_t lzA = cs.compLz4AoBytes;
            const uint64_t lzC = cs.compLz4ColorPalBytes;
            const uint64_t lzT = cs.compLz4TotalBytes;
            ImGui::Separator();
            ImGui::Text("LZ4 per-cluster:");
            ImGui::Text("  pos     %7.2f MB  (-> %.1f%% of raw)", mb(lzP), pos  ? 100.0 * lzP / (double)pos  : 0.0);
            ImGui::Text("  mask    %7.2f MB  (-> %.1f%% of raw)", mb(lzM), mask ? 100.0 * lzM / (double)mask : 0.0);
            ImGui::Text("  ao      %7.2f MB  (-> %.1f%% of raw)", mb(lzA), ao   ? 100.0 * lzA / (double)ao   : 0.0);
            ImGui::Text("  colorPI %7.2f MB  (-> %.1f%% of raw)", mb(lzC), cpal ? 100.0 * lzC / (double)cpal : 0.0);
            ImGui::Text("  TOTAL   %7.2f MB  (%.2fx vs raw, %.2fx vs pal8 uncompressed)",
                        mb(lzT), ratio(lzT), totPal ? (double)totPal / (double)lzT : 0.0);
        }
    }

    ImGui::Separator();
    {
        const auto& hist = g_app.renderer.ColorHistogram();
        uint64_t totalVoxels = 0;
        for (const auto& p : hist) totalVoxels += p.second;
        ImGui::Text("Color Histogram (%zu unique, %llu voxels)",
                    hist.size(), (unsigned long long)totalVoxels);
        {
            const double hBits   = g_app.renderer.ColorEntropyBits();
            const double huff    = g_app.renderer.ColorHuffmanBits();
            const uint32_t palBits = g_app.renderer.ColorPaletteBits();
            const double raw24Mb = mb((uint64_t)totalVoxels * 3);   // 24 bpp
            const double palMb   = mb(((uint64_t)totalVoxels * palBits + 7) / 8);
            const double huffMb  = (totalVoxels * huff) / 8.0 / (1024.0 * 1024.0);
            const double entMb   = (totalVoxels * hBits) / 8.0 / (1024.0 * 1024.0);
            ImGui::Text("  Raw RGB24:    24.00 bits/color  (%7.2f MB)", raw24Mb);
            ImGui::Text("  Palette idx:  %2u   bits/color  (%7.2f MB)  [palette %zu*3B = %.1f KB]",
                        palBits, palMb, hist.size(), hist.size() * 3 / 1024.0);
            ImGui::Text("  Huffman avg:  %5.3f bits/color  (%7.2f MB)", huff, huffMb);
            ImGui::Text("  Shannon H:    %5.3f bits/color  (%7.2f MB, theoretical min)",
                        hBits, entMb);
            if (raw24Mb > 0.0) {
                ImGui::Text("  Huffman ratio vs RGB24: %.2fx smaller", raw24Mb / (huffMb > 0.0 ? huffMb : 1e-9));
            }
        }
        if (ImGui::BeginChild("##colorhist", ImVec2(0, 220), true,
                              ImGuiWindowFlags_HorizontalScrollbar)) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float sw = ImGui::GetFontSize();
            for (const auto& p : hist) {
                const uint32_t c = p.first;
                const uint8_t r = (uint8_t)(c & 0xFF);
                const uint8_t g = (uint8_t)((c >> 8) & 0xFF);
                const uint8_t b = (uint8_t)((c >> 16) & 0xFF);
                ImVec2 pos = ImGui::GetCursorScreenPos();
                dl->AddRectFilled(pos, ImVec2(pos.x + sw, pos.y + sw),
                                  IM_COL32(r, g, b, 255));
                ImGui::Dummy(ImVec2(sw, sw));
                ImGui::SameLine();
                const double pct = totalVoxels ? 100.0 * (double)p.second / (double)totalVoxels : 0.0;
                ImGui::Text("#%02X%02X%02X  %10llu  (%.2f%%)",
                            r, g, b, (unsigned long long)p.second, pct);
            }
        }
        ImGui::EndChild();
    }

    ImGui::Separator();
    ImGui::Text("Camera");
    ImGui::Text("  Pos: %.1f %.1f %.1f",
                (float)g_app.camera.position.x,
                (float)g_app.camera.position.y,
                (float)g_app.camera.position.z);
    ImGui::Text("  Yaw: %.2f  Pitch: %.2f", g_app.camera.yaw, g_app.camera.pitch);

    if (!g_app.sceneReady) {
        ImGui::Separator();
        ImGui::TextUnformatted(g_app.loadStatus.c_str());
    }

    ImGui::End();
}

void FrameControlsWindow()
{
    if (!g_app.showControls) return;
    if (!ImGui::Begin("Controls", &g_app.showControls)) { ImGui::End(); return; }

    {
        // Dataset combo built from discovered assets/*.vox at startup.
        std::vector<const char*> names;
        names.reserve(g_app.datasetPaths.size());
        for (const auto& p : g_app.datasetPaths) names.push_back(p.c_str());
        if (names.empty()) {
            ImGui::TextUnformatted("Dataset: (no .vox files in assets/)");
        } else {
            int idx = g_app.datasetIdx;
            if (idx < 0 || idx >= (int)names.size()) idx = 0;
            if (ImGui::Combo("Dataset", &idx, names.data(), (int)names.size())) {
                g_app.datasetIdx = idx;
                SaveLastVoxToSettings(g_app.datasetPaths[idx]);
                g_app.reloadRequested.store(true);
            }
        }
    }
    // Single tech combo: point-derived techniques only.
    struct TechEntry { const char* name; RenderTech val; };
    static const TechEntry kTechList[] = {
        { "Points",       RenderTech::Points       },
        // HexSprite disabled — dropped to allow single structured-SRV point buffer.
        // { "HexSprite",    RenderTech::HexSprite    },
        { "PointCS",      RenderTech::PointCS      },
        { "PolyVID",      RenderTech::PolyVID      },
        { "Billboard",    RenderTech::Billboard    },
        { "BillboardTri", RenderTech::BillboardTri },
        { "Splat",        RenderTech::Splat        },
        { "PolyAxis",     RenderTech::PolyAxis     },
        { "PolyAxisInst", RenderTech::PolyAxisInstanced },
    };
    const int kTechCount = (int)(sizeof(kTechList) / sizeof(kTechList[0]));
    auto techIdxFrom = [&](RenderTech v) -> int {
        for (int k = 0; k < kTechCount; ++k) if (kTechList[k].val == v) return k;
        return 0;
    };
    {
        const char* techNames[16];
        for (int k = 0; k < kTechCount; ++k) techNames[k] = kTechList[k].name;
        // Close (near) tech
        int tt = techIdxFrom(g_app.tech);
        ImGui::PushItemWidth(180.0f);
        if (ImGui::Combo("##TechClose", &tt, techNames, kTechCount, kTechCount)) {
            g_app.tech = kTechList[tt].val;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Checkbox("Close", &g_app.closeEnabled);
        // Far tech
        int tf = techIdxFrom(g_app.techFar);
        ImGui::PushItemWidth(180.0f);
        if (ImGui::Combo("##TechFar", &tf, techNames, kTechCount, kTechCount)) {
            g_app.techFar = kTechList[tf].val;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Checkbox("Far", &g_app.farEnabled);
    }

    ImGui::Separator();
    const char* lods[] = { "L0 (1 per voxel)", "L1 (2x2x2)", "L2 (4x4x4)", "L3 (8x8x8)", "Auto" };
    int lo = (int)g_app.pointLod;
    if (ImGui::Combo("Point LOD", &lo, lods, IM_ARRAYSIZE(lods))) {
        g_app.pointLod = (PointLod)lo;
    }
    ImGui::SliderFloat("LOD distance", &g_app.pointLodScale, 0.25f, 8.0f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Splat CS filter", &g_app.splatFilter);
    ImGui::Checkbox("Splat 2-pass dilate", &g_app.splatDilate2Pass);
    ImGui::SliderInt("Splat radius", &g_app.splatRadius, 1, 16);
    if (ImGui::CollapsingHeader("Fog")) {
        const char* fogModes[] = { "Off", "Depth" };
        int fm = g_app.fogMode;
        if (ImGui::Combo("Mode", &fm, fogModes, IM_ARRAYSIZE(fogModes))) {
            g_app.fogMode = fm;
        }
        ImGui::SliderFloat("Density", &g_app.fogDensity, 0.0f, 0.005f, "%.5f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Height density", &g_app.heightFogDensity, 0.0f, 5.0f, "%.3f");
        ImGui::SliderFloat("Height falloff", &g_app.heightFogFalloff, 0.0f, 0.2f, "%.4f");
        ImGui::SliderFloat("Height start Y", &g_app.heightFogStart,   -100.0f, 200.0f, "%.1f");
        ImGui::ColorEdit3("Color", g_app.fogColor);
    }
    const char* pls[] = { "Simple", "Complex" };
    int pli = (int)g_app.pointLight;
    if (ImGui::Combo("Point lighting", &pli, pls, IM_ARRAYSIZE(pls))) {
        g_app.pointLight = (PointLighting)pli;
    }
    const char* modes[] = { "Lit", "Flat Color", "Normals", "AO", "AO+LODViz" };
    int m = (int)g_app.mode;
    if (ImGui::Combo("Shading", &m, modes, IM_ARRAYSIZE(modes))) {
        g_app.mode = (ShadingMode)m;
    }
    ImGui::Checkbox("VSync", &g_app.vsync);
    ImGui::SliderInt("Grid size", &g_app.gridSize, 1, 10);
    ImGui::Checkbox("TAA", &g_app.taa);

    if (ImGui::BeginTabBar("##controlTabs")) {
        if (ImGui::BeginTabItem("Render")) {
            if (g_app.graphicsAdapters.size() > 1) {
                std::vector<const char*> names;
                names.reserve(g_app.graphicsAdapters.size() + 1);
                names.push_back("(system default)");
                for (auto& n : g_app.graphicsAdapters) names.push_back(n.c_str());
                int sel = (g_app.adapterIdx < 0) ? 0 : (g_app.adapterIdx + 1);
                if (sel >= (int)names.size()) sel = 0;
                if (ImGui::Combo("GPU adapter", &sel, names.data(), (int)names.size())) {
                    g_app.adapterIdx = (sel == 0) ? -1 : (sel - 1);
                    Settings st = LoadSettings();
                    st.adapterIdx = g_app.adapterIdx;
                    SaveSettings(st);
                }
                if (g_app.adapterIdx != g_app.activeAdapterIdx) {
                    ImGui::TextColored(ImVec4(1, 0.7f, 0.2f, 1), "Restart to apply");
                }
            }
            ImGui::SliderFloat("Sun pitch",     &g_app.sunPitchDeg, 5.0f, 89.0f, "%.1f deg");
            ImGui::SliderFloat("Sun yaw",       &g_app.sunYawDeg, -180.0f, 180.0f, "%.1f deg");
            ImGui::SliderFloat("Sun intensity", &g_app.sunIntensityEV, -4.0f, 4.0f, "%.2f EV");
            ImGui::SliderFloat("Exposure",      &g_app.exposureEV,     -3.0f, 3.0f, "%.2f EV");
            ImGui::SliderFloat("Roughness",     &g_app.roughness,       0.05f, 1.0f, "%.2f");
            ImGui::ColorEdit3("Clear color", g_app.bgColor);
            ImGui::Checkbox("LW: draw chunk bounds (LOD coloured)", &g_app.lwShowBounds);
            ImGui::Separator();
            ImGui::Text("Camera");
            ImGui::SliderFloat("Move speed", &g_app.camera.moveSpeed, 0.1f, 5000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("FOV", &g_app.camera.fovDeg, 30.0f, 110.0f, "%.0f");
            ImGui::Separator();
            ImGui::TextUnformatted("RMB drag: look | WASD: move | Wheel: speed | Q/E or Ctrl/Space: down/up");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Shadow")) {
            ImGui::Checkbox("Sun shadows",  &g_app.sunShadows);
            ImGui::SliderInt("Cascades",    &g_app.shadowCascades, 1, 4, "%d (only 1 wired)");
            {
                const char* sizes[] = { "512", "1024", "2048", "4096" };
                int idx = std::clamp(g_app.shadowMapSizeIdx, 0, 3);
                if (ImGui::Combo("Shadow map", &idx, sizes, IM_ARRAYSIZE(sizes))) {
                    g_app.shadowMapSizeIdx = idx;
                }
            }
            ImGui::SliderFloat("Shadow bias", &g_app.shadowBias, 0.0f, 0.01f, "%.4f");
            ImGui::Checkbox("Shadow cull front (else back)", &g_app.shadowCullFront);
            ImGui::Checkbox("Force shadow rebuild (profiling)", &g_app.shadowForceRebuild);
            {
                const char* slods[] = { "Auto", "L0", "L1", "L2", "L3" };
                ImGui::Combo("Shadow LOD", &g_app.shadowLodIdx, slods, IM_ARRAYSIZE(slods));
            }
            ImGui::Checkbox("Shadow blur fill (empty texels = neighbour avg)", &g_app.shadowBlur);

            ImGui::Separator();
            ImGui::Text("Shadow map preview");
            ID3D11ShaderResourceView* srv = g_app.shadowBlur
                ? g_app.renderer.ShadowFilledSrv()
                : g_app.renderer.ShadowSrv();
            const uint32_t smSize = g_app.renderer.ShadowMapSize();
            if (srv && smSize > 0) {
                float avail = ImGui::GetContentRegionAvail().x;
                float side  = std::max(64.0f, std::min(avail, 512.0f));
                ImGui::Image((ImTextureID)srv, ImVec2(side, side));
                ImGui::Text("source: %s  size: %u x %u",
                            g_app.shadowBlur ? "filled (blurred)" : "raw caster",
                            smSize, smSize);
            } else {
                ImGui::TextUnformatted("(no shadow map — enable Sun shadows)");
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

} // namespace

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int)
{
    SetCwdToProjectRoot();
    WNDCLASSEXW wc = { sizeof(wc) };
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"VoxelTestWnd";
    RegisterClassExW(&wc);

    DWORD style = WS_OVERLAPPEDWINDOW;
    RECT rc = { 0, 0, 1600, 900 };
    AdjustWindowRect(&rc, style, FALSE);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"VoxelTest",
                                style, CW_USEDEFAULT, CW_USEDEFAULT,
                                rc.right - rc.left, rc.bottom - rc.top,
                                nullptr, nullptr, hInst, nullptr);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    g_app.graphicsAdapters = Renderer::EnumerateAdapters();
    {
        Settings st = LoadSettings();
        g_app.adapterIdx = st.adapterIdx;
    }
    g_app.activeAdapterIdx = g_app.adapterIdx;
    if (!g_app.renderer.Init(hwnd, g_app.adapterIdx)) {
        MessageBoxA(nullptr, "Renderer init failed", "VoxelTest", MB_ICONERROR);
        return 1;
    }

    MicroProfileOnThreadCreate("Main");
    MicroProfileSetForceEnable(true);
    MicroProfileSetEnableAllGroups(true);
    MicroProfileSetForceMetaCounters(true);
    MicroProfileGpuInitD3D11(g_app.renderer.Device());
    MicroProfileWebServerStart();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_app.renderer.Device(), g_app.renderer.Context());

    // Enumerate available datasets + restore last-used selection.
    g_app.datasetPaths = DiscoverDatasets();
    {
        std::string last = LoadLastVoxFromSettings();
        if (!last.empty()) {
            for (int i = 0; i < (int)g_app.datasetPaths.size(); ++i) {
                if (g_app.datasetPaths[i] == last) { g_app.datasetIdx = i; break; }
            }
        }
    }

    auto kickLoader = []() {
        std::thread([] {
            std::string err;
            if (g_app.datasetPaths.empty()) {
                g_app.loadErr = "no .vox files found in assets/";
                g_app.loadOk.store(false);
                g_app.loadDone.store(true);
                return;
            }
            int idx = g_app.datasetIdx;
            if (idx < 0 || idx >= (int)g_app.datasetPaths.size()) idx = 0;
            const std::string& voxPath = g_app.datasetPaths[idx];
            g_app.currentVoxPath = voxPath;
            g_app.loadStatus = std::string("Loading ") + voxPath + "...";
            bool ok;
            if (IsLwPath(voxPath)) {
                g_app.pendingIsLw = true;
                ok = lw::LoadWorld(voxPath.c_str(), g_app.pendingLwWorld, err);
                if (!ok && err.empty()) err = "lw load failed";
            } else {
                g_app.pendingIsLw = false;
                ok = LoadVoxScene(voxPath.c_str(), g_app.pendingScene, err);
                if (!ok && err.empty()) err = "vox load failed";
            }
            g_app.loadErr = err;
            g_app.loadOk.store(ok);
            g_app.loadDone.store(true);
        }).detach();
    };
    kickLoader();

    auto last = std::chrono::high_resolution_clock::now();

    MSG msg = {};
    while (!g_app.wantQuit) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) g_app.wantQuit = true;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (g_app.wantQuit) break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        dt = std::min(dt, 0.1f);

        // Handle data-source reload request.
        if (g_app.reloadRequested.load() && g_app.sceneReady) {
            g_app.reloadRequested.store(false);
            g_app.sceneReady = false;
            g_app.loadDone.store(false);
            g_app.loadOk.store(false);
            kickLoader();
        }

        // upload scene once loader done
        if (g_app.loadDone.load() && !g_app.sceneReady) {
            if (g_app.loadOk.load()) {
                bool firstUpload = !g_app.everLoaded;
                if (g_app.pendingIsLw) {
                    // Free legacy Scene side; upload the LW world.
                    g_app.renderer.UploadScene(Scene{});  // clear legacy SRVs
                    g_app.renderer.UploadLwWorld(g_app.pendingLwWorld);
                    // Always re-center on LW load (legacy path only re-centers
                    // on first-ever upload, but switching format wants a reset).
                    {
                        const auto& w = g_app.pendingLwWorld;
                        float cx = 0.5f * (float)(w.worldAabbMin[0] + w.worldAabbMax[0]);
                        float cz = 0.5f * (float)(w.worldAabbMin[2] + w.worldAabbMax[2]);
                        float topY = (float)w.worldAabbMax[1];
                        float dx = (float)(w.worldAabbMax[0] - w.worldAabbMin[0]);
                        float dz = (float)(w.worldAabbMax[2] - w.worldAabbMin[2]);
                        float ext = (dx > dz ? dx : dz);
                        // Pull back + up + tilt down so the whole footprint is
                        // visible from the off-bat (vs the legacy default which
                        // skylines you over the model).
                        g_app.camera.position = hlslpp::float3(cx, topY + ext * 0.5f, cz - ext * 0.5f);
                        g_app.camera.yaw   = 0.0f;
                        g_app.camera.pitch = -0.5f;          // ~ -28 deg
                        g_app.camera.moveSpeed = ext * 0.05f;
                        g_app.camera.farZ = ext * 4.0f + 1000.0f;
                    }
                    g_app.pendingLwWorld = lw::World{};
                } else {
                    g_app.renderer.ClearLwWorld();
                    g_app.renderer.UploadScene(g_app.pendingScene);
                    if (firstUpload) {
                        float cx = 0.5f * (g_app.pendingScene.aabbMin[0] + g_app.pendingScene.aabbMax[0]);
                        float cz = 0.5f * (g_app.pendingScene.aabbMin[2] + g_app.pendingScene.aabbMax[2]);
                        float topY = g_app.pendingScene.aabbMax[1];
                        float dx = g_app.pendingScene.aabbMax[0] - g_app.pendingScene.aabbMin[0];
                        float dz = g_app.pendingScene.aabbMax[2] - g_app.pendingScene.aabbMin[2];
                        float ext = (dx > dz ? dx : dz);
                        g_app.camera.position = hlslpp::float3(cx, topY, cz);
                        g_app.camera.yaw = 0.0f;
                        g_app.camera.pitch = 0.0f;
                        g_app.camera.moveSpeed = ext * 0.05f;
                        g_app.camera.farZ = ext * 4.0f + 1000.0f;
                    }
                }
                g_app.everLoaded = true;
                // Snapshot lightweight bits for on-demand compression analysis
                // (histogram + per-chunk subs). Keeps full pointVertices out of
                // memory after the upload.
                g_app.compScene = Scene{};
                g_app.compScene.colorHistogram = g_app.pendingScene.colorHistogram;
                g_app.compScene.subs           = g_app.pendingScene.subs;
                g_app.compStatus.clear();
                g_app.pendingScene = Scene{};
                g_app.sceneReady = true;
                g_app.loadStatus = "Loaded.";
            } else {
                g_app.loadStatus = "Load failed: " + g_app.loadErr;
                g_app.sceneReady = true;
            }
        }

        UpdateCamera(dt);

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        FrameMenuBar();
        FrameFpsWindow();
        FrameStatsWindow();
        FrameControlsWindow();
        ImGui::Render();

        float clear[4];
        if (g_app.fogMode != 0) {
            clear[0] = g_app.fogColor[0];
            clear[1] = g_app.fogColor[1];
            clear[2] = g_app.fogColor[2];
            clear[3] = 1.0f;
        } else {
            clear[0] = g_app.bgColor[0];
            clear[1] = g_app.bgColor[1];
            clear[2] = g_app.bgColor[2];
            clear[3] = g_app.bgColor[3];
        }
        g_app.renderer.BeginFrame(clear);
        if (g_app.sceneReady && g_app.loadOk.load()) {
            // LodViz forces fog off so the per-LOD colors aren't dimmed/tinted.
            const bool fogOff = (g_app.fogMode == 0) || (g_app.mode == ShadingMode::LodViz);
            float effFogDensity   = fogOff ? 0.0f : g_app.fogDensity;
            float effHeightFogDen = (g_app.mode == ShadingMode::LodViz) ? 0.0f : g_app.heightFogDensity;
            const float kDeg2Rad = 3.14159265358979f / 180.0f;
            float pr = g_app.sunPitchDeg * kDeg2Rad;
            float yr = g_app.sunYawDeg   * kDeg2Rad;
            float sunDir[3] = {
                cosf(pr) * sinf(yr),
                sinf(pr),
                cosf(pr) * cosf(yr),
            };

            DrawSceneParams ps;
            ps.mode             = g_app.mode;
            ps.gridSize         = g_app.gridSize;
            ps.tech             = g_app.tech;
            ps.techFar          = g_app.techFar;
            ps.closeEnabled     = g_app.closeEnabled;
            ps.farEnabled       = g_app.farEnabled;
            ps.pointLight       = g_app.pointLight;
            ps.pointLod         = g_app.pointLod;
            ps.pointLodScale    = g_app.pointLodScale;
            ps.splatFilter      = g_app.splatFilter;
            ps.splatDilate2Pass = g_app.splatDilate2Pass;
            ps.fogColor[0]      = g_app.fogColor[0];
            ps.fogColor[1]      = g_app.fogColor[1];
            ps.fogColor[2]      = g_app.fogColor[2];
            ps.fogDensity       = effFogDensity;
            ps.heightFogDensity = effHeightFogDen;
            ps.heightFogFalloff = g_app.heightFogFalloff;
            ps.heightFogStart   = g_app.heightFogStart;
            ps.splatRadius      = g_app.splatRadius;
            ps.taa              = g_app.taa;
            ps.sunDir[0]        = sunDir[0];
            ps.sunDir[1]        = sunDir[1];
            ps.sunDir[2]        = sunDir[2];
            ps.sunIntensity     = exp2f(g_app.sunIntensityEV);
            ps.sunShadows       = g_app.sunShadows;
            ps.shadowCascades   = g_app.shadowCascades;
            {
                const int sizes[] = { 512, 1024, 2048, 4096 };
                ps.shadowMapSize  = sizes[std::clamp(g_app.shadowMapSizeIdx, 0, 3)];
            }
            ps.shadowBias       = g_app.shadowBias;
            ps.shadowCullFront  = g_app.shadowCullFront;
            ps.shadowForceRebuild = g_app.shadowForceRebuild;
            ps.shadowLod          = g_app.shadowLodIdx - 1;   // 0 -> Auto (-1)
            ps.shadowBlur         = g_app.shadowBlur;
            ps.lwShowBounds       = g_app.lwShowBounds;
            ps.exposure         = exp2f(g_app.exposureEV);
            ps.roughness        = g_app.roughness;
            if (g_app.renderer.HasLwWorld()) {
                g_app.renderer.DrawLwScene(g_app.camera, ps);
            } else {
                g_app.renderer.DrawScene(g_app.camera, ps);
            }
        }
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_app.renderer.EndFrame(g_app.vsync);

        MicroProfileFlip();

        auto end = std::chrono::high_resolution_clock::now();
        g_app.cpuFrameMs = std::chrono::duration<double, std::milli>(end - now).count();
    }

    // MicroProfileShutdown joins worker threads (web server, context-switch ETW
    // tracer, GPU timers). On Windows the ETW unregister can take seconds.
    // Skip it — the OS reclaims sockets/threads at process exit.

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    g_app.renderer.Shutdown();
    DestroyWindow(hwnd);
    return 0;
}
