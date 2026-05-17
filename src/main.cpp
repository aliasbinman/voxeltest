#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>

#include "renderer.h"
#include "camera.h"
#include "vox_loader.h"
#include "mesh_loader.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <atomic>
#include <string>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace {

void SetCwdToProjectRoot() {
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
    RenderTech  tech = RenderTech::Hybrid;
    bool     vsync = false;
    int      gridSize = 6;
    bool     showChunkBounds = false;
    bool     zPrepass = false;
    PointLighting pointLight = PointLighting::Complex;
    PointLod pointLod = PointLod::Auto;
    float    pointLodScale = 1.0f;
    float    hybridThreshold = 1.0f;   // ppv >= this -> polygons, else points
    bool     splatFilter = true;
    int      fogMode = 0;        // 0 = off, 1 = depth
    float    fogDensity = 0.0005f;
    float    fogColor[3] = { 0.55f, 0.60f, 0.70f };
    int      msaa = 1;
    bool     rmbDown = false;
    POINT    lastMouse = { 0, 0 };
    bool     keys[256] = {};
    bool     wantQuit = false;
    bool     sceneReady = false;
    std::string loadStatus = "Loading rungholt.obj...";
    Scene       pendingScene;
    MergedMesh  pendingMerged;
    bool        mergedLoaded = false;
    AtlasMesh   pendingAtlas;
    bool        atlasLoaded = false;
    int         voxSource = 0;   // 0=original, 1=culled
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

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
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
        if (g_app.camera.moveSpeed < 0.05f)    g_app.camera.moveSpeed = 0.05f;
        if (g_app.camera.moveSpeed > 10000.0f) g_app.camera.moveSpeed = 10000.0f;
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
            if (g_app.camera.pitch >  lim) g_app.camera.pitch =  lim;
            if (g_app.camera.pitch < -lim) g_app.camera.pitch = -lim;
            SetCursorPos(g_app.lastMouse.x, g_app.lastMouse.y);
        }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

void UpdateCamera(float dt) {
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

void FrameTopBar() {
    ImGui::Begin("VoxelTest");

    // fps stats
    g_app.fpsAvg = 1000.0f / (g_app.cpuFrameMs > 0.001 ? (float)g_app.cpuFrameMs : 16.0f);
    g_app.fpsHist[g_app.fpsHistIdx] = g_app.fpsAvg;
    g_app.fpsHistIdx = (g_app.fpsHistIdx + 1) % IM_ARRAYSIZE(g_app.fpsHist);

    ImGui::Text("CPU frame: %.2f ms (%.0f FPS)", g_app.cpuFrameMs, g_app.fpsAvg);
    ImGui::PlotLines("FPS", g_app.fpsHist, IM_ARRAYSIZE(g_app.fpsHist),
                     g_app.fpsHistIdx, nullptr, 0.0f, 240.0f, ImVec2(0, 60));

    const char* techs[] = { "PolygonBased", "Points", "Hybrid", "HexSprite", "PointCS", "PolyVID", "Billboard", "BillboardTri", "Hybrid2", "MergedMesh", "Splat", "SplatHybrid", "AtlasMesh" };
    {
        int tt = (int)g_app.tech;
        if (ImGui::Combo("Technique", &tt, techs, IM_ARRAYSIZE(techs), IM_ARRAYSIZE(techs))) {
            g_app.tech = (RenderTech)tt;
        }
    }

    ImGui::Separator();
    {
        const char* sources[] = { "Original", "Culled" };
        int sel = g_app.voxSource;
        if (ImGui::Combo("Voxel data", &sel, sources, IM_ARRAYSIZE(sources))) {
            g_app.voxSource = sel;
            g_app.reloadRequested.store(true);
        }
    }
    const char* lods[] = { "L0 (1 per voxel)", "L1 (2x2x2)", "L2 (4x4x4)", "Auto" };
    int lo = (int)g_app.pointLod;
    if (ImGui::Combo("Point LOD", &lo, lods, IM_ARRAYSIZE(lods))) {
        g_app.pointLod = (PointLod)lo;
    }
    ImGui::SliderFloat("LOD distance", &g_app.pointLodScale, 0.25f, 8.0f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Hybrid poly threshold (ppv)", &g_app.hybridThreshold, 0.25f, 16.0f, "%.2f px", ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Splat CS filter", &g_app.splatFilter);
    if (ImGui::CollapsingHeader("Fog")) {
        const char* fogModes[] = { "Off", "Depth" };
        int fm = g_app.fogMode;
        if (ImGui::Combo("Mode", &fm, fogModes, IM_ARRAYSIZE(fogModes))) {
            g_app.fogMode = fm;
        }
        ImGui::SliderFloat("Density", &g_app.fogDensity, 0.0f, 0.005f, "%.5f", ImGuiSliderFlags_Logarithmic);
        ImGui::ColorEdit3("Color", g_app.fogColor);
    }
    const char* pls[] = { "Simple", "Complex" };
    int pli = (int)g_app.pointLight;
    if (ImGui::Combo("Point lighting", &pli, pls, IM_ARRAYSIZE(pls))) {
        g_app.pointLight = (PointLighting)pli;
    }
    const char* modes[] = { "Lit", "Flat Color", "Normals" };
    int m = (int)g_app.mode;
    if (ImGui::Combo("Shading", &m, modes, IM_ARRAYSIZE(modes))) {
        g_app.mode = (ShadingMode)m;
    }
    ImGui::Checkbox("VSync", &g_app.vsync);
    ImGui::SliderInt("Grid size", &g_app.gridSize, 1, 10);
    ImGui::Checkbox("Show chunk bounds", &g_app.showChunkBounds);
    ImGui::Checkbox("Z Prepass", &g_app.zPrepass);
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

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
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
            const char* voxPath = (g_app.voxSource == 1)
                ? "assets/rungholt_culled.vox"
                : "assets/rungholt.vox";
            g_app.loadStatus = std::string("Loading ") + voxPath + "...";
            bool ok = LoadVoxScene(voxPath, g_app.pendingScene, err);
            if (ok && firstLoad) {
                std::string merr;
                g_app.mergedLoaded = LoadMergedMesh("assets/rungholt_merged.msh",
                                                    g_app.pendingMerged, merr);
                std::string aerr;
                g_app.atlasLoaded = LoadAtlasMesh("assets/rungholt_atlas.msh",
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
        if (dt > 0.1f) dt = 0.1f;

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
                g_app.renderer.UploadScene(g_app.pendingScene);
                if (g_app.mergedLoaded) {
                    g_app.renderer.UploadMergedMesh(g_app.pendingMerged);
                    g_app.pendingMerged = MergedMesh{};
                }
                if (g_app.atlasLoaded) {
                    g_app.renderer.UploadAtlasMesh(g_app.pendingAtlas);
                    g_app.pendingAtlas = AtlasMesh{};
                }
                // center camera on aabb
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
            g_app.renderer.DrawScene(g_app.camera, g_app.mode, g_app.gridSize, g_app.tech, g_app.showChunkBounds, g_app.zPrepass, g_app.pointLight, g_app.pointLod, g_app.pointLodScale, g_app.splatFilter, g_app.fogColor, effFogDensity, g_app.hybridThreshold);
        }
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_app.renderer.EndFrame(g_app.vsync);

        auto end = std::chrono::high_resolution_clock::now();
        g_app.cpuFrameMs = std::chrono::duration<double, std::milli>(end - now).count();
    }

    // Loader threads are detached; nothing to join.

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    g_app.renderer.Shutdown();
    DestroyWindow(hwnd);
    return 0;
}
