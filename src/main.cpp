#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>

#include "renderer.h"
#include "camera.h"
#include "vox_loader.h"
#include "mesh_loader.h"
#include "asset_version.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"

#include "microprofile.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <atomic>
#include <algorithm>
#include <string>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace {

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

// Check magic + version of a baked asset. Returns false if file is missing,
// has wrong magic, or has stale version -> caller regenerates.
bool AssetIsCurrent(const char* path, const char* magic4)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    char m[4];
    if (fread(m, 1, 4, f) != 4 || memcmp(m, magic4, 4) != 0) { fclose(f); return false; }
    uint32_t v = 0;
    bool okv = (fread(&v, sizeof(uint32_t), 1, f) == 1 && v == kAssetVersion);
    fclose(f);
    return okv;
}

bool AllAssetsCurrent()
{
    return AssetIsCurrent("assets/kingslanding.vox",        "VXL3")
        && AssetIsCurrent("assets/kingslanding_culled.vox", "VXL3")
        && AssetIsCurrent("assets/kingslanding_merged.msh", "MSH1")
        && AssetIsCurrent("assets/kingslanding_atlas.msh",  "MSH2");
}

// Synchronously runs voxelize.exe (which sits next to voxeltest.exe). Returns
// true on success. Stdout from voxelize is inherited so users see progress.
bool RunVoxelize()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    wchar_t* slash = wcsrchr(exePath, L'\\');
    if (!slash) return false;
    *(slash + 1) = 0;
    std::wstring tool = std::wstring(exePath) + L"voxelize.exe";

    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    std::wstring cmd = L"\"" + tool + L"\" "
                       L"\"assets\\KingsLanding2017\\KingsLandingFull.obj\" "
                       L"\"assets\\kingslanding.vox\"";
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                        0, nullptr, nullptr, &si, &pi)) {
        return false;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return code == 0;
}


struct AppState {
    Renderer renderer;
    Camera   camera;
    ShadingMode mode = ShadingMode::Lit;
    RenderTech  tech = RenderTech::PolygonBased;     // "Close" tech
    RenderTech  techFar = RenderTech::None;          // None = use Close for all chunks
    float    sunPitchDeg = 60.0f;
    float    sunYawDeg   = 63.0f;
    float    sunIntensityEV = 0.0f;     // log2 stops; linear = 2^EV
    float    exposureEV     = 0.0f;     // log2 stops; linear = 2^EV
    float    roughness      = 0.6f;
    bool     sunShadows     = false;
    bool     colorizeClusters = false;
    bool     vsync = false;
    int      gridSize = 1;
    bool     showChunkBounds = false;
    bool     wireframe = false;
    bool     taa = true;
    bool     zPrepass = false;
    PointLighting pointLight = PointLighting::Complex;
    PointLod pointLod = PointLod::Auto;
    float    pointLodScale = 1.0f;
    bool     splatFilter = true;
    int      splatRadius = 3;       // CS dilation half-window in pixels
    int      fogMode = 1;        // 0 = off, 1 = depth (drives effFogDensity gating)
    float    fogDensity = 0.0004f;
    float    fogColor[3] = { 0.55f, 0.60f, 0.70f };
    float    heightFogDensity = 4.5f;        // 0 = off
    float    heightFogFalloff = 0.05f;       // exp falloff per unit height
    float    heightFogStart   = -6.5f;        // world Y of fog ground plane
    int      msaa = 1;
    bool     rmbDown = false;
    POINT    lastMouse = { 0, 0 };
    bool     keys[256] = {};
    bool     wantQuit = false;
    bool     sceneReady = false;
    std::string loadStatus = "Loading kingslanding.obj...";
    Scene       pendingScene;
    MergedMesh  pendingMerged;
    bool        mergedLoaded = false;
    AtlasMesh   pendingAtlas;
    bool        atlasLoaded = false;
    bool        everLoaded = false;
    DataSet     dataset = DataSet::Reduced;
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
        // log-scale: each notch multiplies speed by ~1.2
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

void FrameTopBar()
{
    ImGui::Begin("VoxelTest");

    // fps stats
    g_app.fpsAvg = 1000.0f / (g_app.cpuFrameMs > 0.001 ? (float)g_app.cpuFrameMs : 16.0f);
    g_app.fpsHist[g_app.fpsHistIdx] = g_app.fpsAvg;
    g_app.fpsHistIdx = (g_app.fpsHistIdx + 1) % IM_ARRAYSIZE(g_app.fpsHist);

    ImGui::Text("CPU frame: %.2f ms (%.0f FPS)", g_app.cpuFrameMs, g_app.fpsAvg);
    ImGui::PlotLines("FPS", g_app.fpsHist, IM_ARRAYSIZE(g_app.fpsHist),
                     g_app.fpsHistIdx, nullptr, 0.0f, 240.0f, ImVec2(0, 60));

    const char* datasets[] = { "Full", "Merged", "Reduced" };
    {
        int ds = (int)g_app.dataset;
        if (ImGui::Combo("Dataset", &ds, datasets, IM_ARRAYSIZE(datasets))) {
            DataSet prev = g_app.dataset;
            g_app.dataset = (DataSet)ds;
            // Full+Merged share the Original .vox; Reduced uses the Culled .vox.
            bool needReload = ((prev == DataSet::Reduced) != (g_app.dataset == DataSet::Reduced));
            if (needReload) g_app.reloadRequested.store(true);
        }
    }
    // Non-hybrid techs only. Close/Far selectors compose hybrids implicitly.
    struct TechEntry { const char* name; RenderTech val; };
    static const TechEntry kTechList[] = {
        { "TriMesh",      RenderTech::PolygonBased },
        { "Points",       RenderTech::Points       },
        { "HexSprite",    RenderTech::HexSprite    },
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
        // Close pulldown: required (no None entry).
        const char* closeNames[16]; for (int k = 0; k < kTechCount; ++k) closeNames[k] = kTechList[k].name;
        int tt = techIdxFrom(g_app.tech);
        if (ImGui::Combo("Technique Close", &tt, closeNames, kTechCount, kTechCount)) {
            g_app.tech = kTechList[tt].val;
        }
        // Far pulldown: same list prefixed with "None".
        const char* farNames[17]; farNames[0] = "None";
        for (int k = 0; k < kTechCount; ++k) farNames[k + 1] = kTechList[k].name;
        int tf = (g_app.techFar == RenderTech::None) ? 0 : (techIdxFrom(g_app.techFar) + 1);
        if (ImGui::Combo("Technique Far", &tf, farNames, kTechCount + 1, kTechCount + 1)) {
            g_app.techFar = (tf == 0) ? RenderTech::None : kTechList[tf - 1].val;
        }
    }

    ImGui::Separator();
    const char* lods[] = { "L0 (1 per voxel)", "L1 (2x2x2)", "L2 (4x4x4)", "Auto" };
    int lo = (int)g_app.pointLod;
    if (ImGui::Combo("Point LOD", &lo, lods, IM_ARRAYSIZE(lods))) {
        g_app.pointLod = (PointLod)lo;
    }
    ImGui::SliderFloat("LOD distance", &g_app.pointLodScale, 0.25f, 8.0f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Splat CS filter", &g_app.splatFilter);
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
    const char* modes[] = { "Lit", "Flat Color", "Normals", "AO" };
    int m = (int)g_app.mode;
    if (ImGui::Combo("Shading", &m, modes, IM_ARRAYSIZE(modes))) {
        g_app.mode = (ShadingMode)m;
    }
    ImGui::Checkbox("VSync", &g_app.vsync);
    ImGui::SliderInt("Grid size", &g_app.gridSize, 1, 10);
    ImGui::Checkbox("Show chunk bounds", &g_app.showChunkBounds);
    ImGui::Checkbox("Wireframe", &g_app.wireframe);
    ImGui::Checkbox("TAA", &g_app.taa);
    ImGui::Checkbox("Z Prepass", &g_app.zPrepass);
    ImGui::Checkbox("Sun shadows", &g_app.sunShadows);
    ImGui::Checkbox("Colorize clusters", &g_app.colorizeClusters);
    ImGui::SliderFloat("Sun pitch",     &g_app.sunPitchDeg, 5.0f, 89.0f, "%.1f deg");
    ImGui::SliderFloat("Sun yaw",       &g_app.sunYawDeg, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Sun intensity", &g_app.sunIntensityEV, -4.0f, 4.0f, "%.2f EV");
    ImGui::SliderFloat("Exposure",      &g_app.exposureEV,     -3.0f, 3.0f, "%.2f EV");
    ImGui::SliderFloat("Roughness",     &g_app.roughness,       0.05f, 1.0f, "%.2f");
    {
        const char* msaaItems[] = { "Off", "2x", "4x", "8x" };
        const int   msaaVals[]  = { 1, 2, 4, 8 };
        int sel = 0;
        for (int k = 0; k < 4; ++k) if (msaaVals[k] == g_app.msaa) sel = k;
        if (ImGui::Combo("MSAA", &sel, msaaItems, IM_ARRAYSIZE(msaaItems))) {
            g_app.msaa = msaaVals[sel];
            g_app.renderer.SetMsaa((uint32_t)g_app.msaa);
        }
    }
    ImGui::ColorEdit3("Clear color", g_app.bgColor);

    ImGui::Separator();
    ImGui::Text("Scene");
    {
        size_t totalChunks = g_app.renderer.DrawCount() * (size_t)g_app.gridSize * (size_t)g_app.gridSize;
        ImGui::Text("  Chunks: %zu  drawn: %u  (%.1f%%)",
                    totalChunks,
                    g_app.renderer.LastDrawnCount(),
                    totalChunks ? 100.0f * g_app.renderer.LastDrawnCount() / (float)totalChunks : 0.0f);
    }
    ImGui::Text("  Tris drawn: %llu / %llu",
                (unsigned long long)g_app.renderer.LastDrawnTris(),
                (unsigned long long)g_app.renderer.TotalTriangles());
    ImGui::Text("  Poly:  %llu tris  %llu verts",
                (unsigned long long)g_app.renderer.LastPolyTris(),
                (unsigned long long)g_app.renderer.LastPolyVerts());
    ImGui::Text("  Points: %llu", (unsigned long long)g_app.renderer.LastPointCount());
    ImGui::Text("  Vertices:  %llu", (unsigned long long)g_app.renderer.TotalVertices());
    ImGui::Text("  Triangles: %llu", (unsigned long long)g_app.renderer.TotalTriangles());
    ImGui::Text("  VB: %.1f MB   IB: %.1f MB   (24 B/vert, 4 B/idx)",
                g_app.renderer.VbBytes() / (1024.0 * 1024.0),
                g_app.renderer.IbBytes() / (1024.0 * 1024.0));

    ImGui::Separator();
    ImGui::Text("Camera");
    ImGui::Text("  Pos: %.1f %.1f %.1f",
                (float)g_app.camera.position.x,
                (float)g_app.camera.position.y,
                (float)g_app.camera.position.z);
    ImGui::Text("  Yaw: %.2f  Pitch: %.2f", g_app.camera.yaw, g_app.camera.pitch);
    ImGui::SliderFloat("Move speed", &g_app.camera.moveSpeed, 0.1f, 5000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("FOV", &g_app.camera.fovDeg, 30.0f, 110.0f, "%.0f");

    if (!g_app.sceneReady) {
        ImGui::Separator();
        ImGui::TextUnformatted(g_app.loadStatus.c_str());
    }

    ImGui::Separator();
    ImGui::TextUnformatted("RMB drag: look | WASD: move | Wheel: speed | Q/E or Ctrl/Space: down/up");

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

    if (!g_app.renderer.Init(hwnd)) {
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

    auto kickLoader = [](bool firstLoad) {
        std::thread([firstLoad] {
            std::string err;
            // Detect missing/stale baked assets and regen via voxelize.exe.
            if (!AllAssetsCurrent()) {
                g_app.loadStatus = "Regenerating baked assets (running voxelize.exe)...";
                if (!RunVoxelize() || !AllAssetsCurrent()) {
                    g_app.loadErr = "voxelize.exe failed (missing OBJ?)";
                    g_app.loadOk.store(false);
                    g_app.loadDone.store(true);
                    return;
                }
            }
            const char* voxPath = (g_app.dataset == DataSet::Reduced)
                ? "assets/kingslanding_culled.vox"
                : "assets/kingslanding.vox";
            g_app.loadStatus = std::string("Loading ") + voxPath + "...";
            bool ok = LoadVoxScene(voxPath, g_app.pendingScene, err);
            if (ok && firstLoad) {
                std::string merr;
                g_app.mergedLoaded = LoadMergedMesh("assets/kingslanding_merged.msh",
                                                    g_app.pendingMerged, merr);
                std::string aerr;
                g_app.atlasLoaded = LoadAtlasMesh("assets/kingslanding_atlas.msh",
                                                  g_app.pendingAtlas, aerr);
            }
            g_app.loadErr = err;
            g_app.loadOk.store(ok);
            g_app.loadDone.store(true);
        }).detach();
    };
    kickLoader(true);

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
            kickLoader(false);
        }

        // upload scene once loader done
        if (g_app.loadDone.load() && !g_app.sceneReady) {
            if (g_app.loadOk.load()) {
                bool firstUpload = !g_app.everLoaded;
                g_app.renderer.UploadScene(g_app.pendingScene);
                // Merged/Atlas are static across reloads; upload only when we
                // actually loaded fresh data (firstUpload). Re-uploading empties
                // would reset the GPU buffers to null.
                if (firstUpload && g_app.mergedLoaded && !g_app.pendingMerged.vertices.empty()) {
                    g_app.renderer.UploadMergedMesh(g_app.pendingMerged);
                    g_app.pendingMerged = MergedMesh{};
                }
                if (firstUpload && g_app.atlasLoaded && !g_app.pendingAtlas.vertices.empty()) {
                    g_app.renderer.UploadAtlasMesh(g_app.pendingAtlas);
                    g_app.pendingAtlas = AtlasMesh{};
                }
                // Center the camera only on the very first load; keep the
                // user's view when switching dataset.
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
                g_app.everLoaded = true;
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
        FrameTopBar();
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
            float effFogDensity = (g_app.fogMode == 0) ? 0.0f : g_app.fogDensity;
            // Spherical sun direction from pitch/yaw sliders.
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
            ps.techClose        = g_app.tech;
            ps.techFar          = g_app.techFar;
            ps.dataset          = g_app.dataset;
            ps.showChunkBounds  = g_app.showChunkBounds;
            ps.zPrepass         = g_app.zPrepass;
            ps.pointLight       = g_app.pointLight;
            ps.pointLod         = g_app.pointLod;
            ps.pointLodScale    = g_app.pointLodScale;
            ps.splatFilter      = g_app.splatFilter;
            ps.fogColor[0]      = g_app.fogColor[0];
            ps.fogColor[1]      = g_app.fogColor[1];
            ps.fogColor[2]      = g_app.fogColor[2];
            ps.fogDensity       = effFogDensity;
            ps.heightFogDensity = g_app.heightFogDensity;
            ps.heightFogFalloff = g_app.heightFogFalloff;
            ps.heightFogStart   = g_app.heightFogStart;
            ps.hybridThreshold  = 1.0f;
            ps.wireframe        = g_app.wireframe;
            ps.splatRadius      = g_app.splatRadius;
            ps.taa              = g_app.taa;
            ps.sunDir[0]        = sunDir[0];
            ps.sunDir[1]        = sunDir[1];
            ps.sunDir[2]        = sunDir[2];
            ps.sunIntensity     = exp2f(g_app.sunIntensityEV);
            ps.exposure         = exp2f(g_app.exposureEV);
            ps.roughness        = g_app.roughness;
            ps.sunShadows       = g_app.sunShadows;
            ps.colorizeClusters = g_app.colorizeClusters;
            g_app.renderer.DrawScene(g_app.camera, ps);
        }
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_app.renderer.EndFrame(g_app.vsync);

        MicroProfileFlip();

        auto end = std::chrono::high_resolution_clock::now();
        g_app.cpuFrameMs = std::chrono::duration<double, std::milli>(end - now).count();
    }

    // Loader threads are detached; nothing to join.

    MicroProfileShutdown();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    g_app.renderer.Shutdown();
    DestroyWindow(hwnd);
    return 0;
}
