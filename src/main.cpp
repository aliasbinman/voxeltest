#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <windowsx.h>

#include "renderer.h"
#include "camera.h"
#include "asset_version.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"

#include "microprofile.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace
{

constexpr const char* kSettingsPath = "voxeltest.settings";

struct Settings
{
    std::string lastVox;
    int adapterIdx = -1;  // -1 = system default
    int monitorIdx = -1;  // -1 = system default placement; else EnumDisplayMonitors index
};

Settings LoadSettings()
{
    Settings s;
    std::ifstream f(kSettingsPath);
    if (!f)
        return s;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.rfind("lastVox=", 0) == 0)
            s.lastVox = line.substr(8);
        else if (line.rfind("adapter=", 0) == 0)
            s.adapterIdx = std::atoi(line.c_str() + 8);
        else if (line.rfind("monitor=", 0) == 0)
            s.monitorIdx = std::atoi(line.c_str() + 8);
    }
    return s;
}
void SaveSettings(const Settings& s)
{
    std::ofstream f(kSettingsPath, std::ios::trunc);
    if (!f)
        return;
    f << "lastVox=" << s.lastVox << "\n";
    f << "adapter=" << s.adapterIdx << "\n";
    f << "monitor=" << s.monitorIdx << "\n";
}

// Compat wrappers â€” call sites pass a path-only or want the path only.
std::string LoadLastVoxFromSettings()
{
    return LoadSettings().lastVox;
}
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
    if (!std::filesystem::is_directory("assets", ec))
        return out;
    for (const auto& e : std::filesystem::directory_iterator("assets", ec))
    {
        if (!e.is_regular_file())
            continue;
        auto ext = e.path().extension();
        if (ext == ".vox" || ext == ".lw")
        {
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
    if (slash)
        *slash = 0;
    SetCurrentDirectoryW(exePath);
    for (int i = 0; i < 8; ++i)
    {
        if (GetFileAttributesW(L"shaders\\voxel.hlsl") != INVALID_FILE_ATTRIBUTES)
            return;
        SetCurrentDirectoryW(L"..");
    }
}

struct AppState
{
    HWND hwnd = nullptr;
    // Resolution combo request; applied at top of main loop (SetWindowPos
    // sends WM_SIZE synchronously, which must not run mid-frame).
    int pendingClientW = 0, pendingClientH = 0;
    Renderer renderer;
    Camera camera;
    ShadingMode mode = ShadingMode::Lit;
    RenderTech tech    = RenderTech::OctetBillboards; // close-ring tech
    RenderTech techFar = RenderTech::PointCS_Block;   // far-ring tech
    bool closeEnabled = true;
    bool farEnabled   = true;
    float sunPitchDeg = 10.0f;
    float sunYawDeg = 63.0f;
    float sunIntensityEV = 0.0f; // log2 stops; linear = 2^EV
    bool sunShadows = true;
    int shadowCascades = 1;   // 1..4 (only cascade 0 active)
    int shadowMapSizeIdx = 2; // index into {512,1024,2048,4096}
    float shadowBias = 0.00015f;
    bool shadowCullFront = false;
    bool shadowForceRebuild = false;
    int shadowLodIdx = 0; // 0=Auto, 1..4 = L0..L3
    bool shadowBlur = false;
    float exposureEV = 0.0f; // log2 stops; linear = 2^EV
    float roughness = 0.6f;
    bool vsync = true;
    int gridSize = 1;
    bool taa = true;
    PointLighting pointLight = PointLighting::Complex;
    PointLod pointLod = PointLod::Auto;
    float pointLodScale = 1.0f;
    bool splatFilter = true;
    bool splatDilate2Pass = false;
    int splatRadius = 1; // CS dilation half-window in pixels
    int fogMode = 1;     // 0 = off, 1 = depth (drives effFogDensity gating)
    float fogDensity = 0.0004f;
    float fogColor[3] = {0.55f, 0.60f, 0.70f};
    float heightFogDensity = 4.5f;  // 0 = off
    float heightFogFalloff = 0.05f; // exp falloff per unit height
    float heightFogStart = -6.5f;   // world Y of fog ground plane
    float godrayStrength = 0.55f;
    float godrayAngleDeg = 10.0f;
    float godrayEmaAlpha = 0.15f;
    bool  godrayAniso = false;       // false = 24-tap line blur, true = SampleGrad anisotropic
    bool  godraySeparable = false;   // 2-pass sparse + fill
    int   godraySeparableStride = 6; // stride (pixels) between sparse taps
    float lastGodrayCamPos[3] = {0, 0, 0};
    float godrayTint[3] = {1.00f, 0.85f, 0.45f};
    bool rmbDown = false;
    POINT lastMouse = {0, 0};
    bool keys[256] = {};
    // ---- Camera recording / playback ----
    enum class RecMode { Idle, Recording, Playing };
    struct CamSample { float dt; float pos[3]; float yaw; float pitch; };
    RecMode recMode = RecMode::Idle;
    std::vector<CamSample> recSamples;
    std::vector<std::string> recClips; // filenames under recordings/
    int recSelected = -1;
    std::vector<CamSample> playSamples;
    float playT = 0.0f;
    float playTotal = 0.0f;
    bool  playPaused = false;
    float smoothSec = 0.0f;
    bool  showRecording = false;
    bool  showGpuProfile = true;
    bool  showGpuRes  = false;
    static constexpr int kGpuProfHistory = 10;
    float  gpuProfHist[1024][kGpuProfHistory] = {};
    // Parallel ring: did the marker actually fire this frame? Avg ignores
    // non-firing frames so toggles don't poison the rolling average.
    bool   gpuProfFired[1024][kGpuProfHistory] = {};
    int    gpuProfHistIdx = 0;
    float  gpuProfUsPerPx = 10.0f; // Âµs of GPU time per displayed pixel
    // Sticky "first-seen" registry â€” once a GPU timer fires it stays in the
    // table (in registration / first-seen order) so rows don't jump as
    // features get toggled.
    bool  gpuProfSeen[1024] = {};
    bool  recDirScanned = false;
    bool  recCursorHidden = false;
    bool wantQuit = false;
    bool sceneReady = false;
    std::string loadStatus = "Loading...";
    std::vector<std::string> graphicsAdapters; // populated at startup
    int adapterIdx = -1;                       // selected adapter idx (-1=default)
    int activeAdapterIdx = -1;                 // adapter actually in use this run
    bool lwShowBounds = false;                 // debug: draw per-chunk AABBs
    bool cheapAO = false;                      // top-down depth-based AO
    float aoFadeUnits = 16.0f;
    float aoPushTexels = 1.0f;
    float aoStrength = 1.0f;
    float ambient = 0.35f; // ambient term boost
    // Saved camera views. view1 = auto-fit from world AABB (legacy default,
    // captured on first load). view2 = curated viewpoint hardcoded below.
    float view1Pos[3] = {0.0f, 0.0f, 0.0f};
    float view1Yaw = 0.0f;
    float view1Pitch = 0.0f;
    bool  view1Valid = false;
    std::string currentVoxPath;
    std::vector<std::string> datasetPaths; // discovered assets/*.vox at startup
    int datasetIdx = 0;                    // index into datasetPaths
    lw::World pendingLwWorld;
    bool everLoaded = false;
    std::atomic<bool> reloadRequested{false};
    std::atomic<bool> loadDone{false};
    // Per-LOD streaming flags: -1=not loaded, 0=loaded but not uploaded, 1=uploaded.
    std::atomic<int> lodReadyFlag[lw::kLodCount] = {{-1}, {-1}, {-1}, {-1}, {-1}};
    // Phase 2a continuous streaming.
    std::atomic<float> camPosAtomic[3] = {{0.0f}, {0.0f}, {0.0f}};
    std::atomic<float> frustumAtomic[24] = {}; // 6 planes Ã— 4 floats
    std::atomic<bool> frustumValid{false};
    std::atomic<bool> loaderQuit{false};
    std::atomic<bool> loaderTrigger{false};
    std::atomic<float> streamRadiusScale{3.0f}; // wider shells preload finer LODs farther out â†’ less pop-in
    // psmain_post writes every pixel (sky branch + scene branch both emit
    // float4(c, 1.0)) so the swapchain RTV clear is redundant when post is on.
    bool skipBackbufferClear = true;
    std::mutex loaderMu;
    std::condition_variable loaderCv;
    std::thread loaderThread;
    float lastTriggerCam[3] = {0, 0, 0};
    // Per-LOD mutex guards pendingLwWorld.lods[L] across the loader/main
    // boundary â€” worker holds while moving fresh LOD in, main holds while
    // reading during UploadLwLodOnly.
    std::mutex lodMu[lw::kLodCount];
    std::atomic<bool> loadOk{false};
    std::string loadErr;
    float bgColor[4] = {0.10f, 0.12f, 0.16f, 1.0f};

    // perf
    double cpuFrameMs = 0.0;
    float fpsAvg = 0.0f;
    float fpsHist[120] = {};
    int fpsHistIdx = 0;

    // window visibility
    bool showControls = true;
    bool showFps = false;
    bool showStats = false;
};

AppState g_app;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED)
        {
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
        if (wp < 256)
            g_app.keys[wp] = true;
        if (wp == VK_ESCAPE)
            g_app.wantQuit = true;
        return 0;
    case WM_KEYUP:
        if (wp < 256)
            g_app.keys[wp] = false;
        return 0;
    case WM_RBUTTONDOWN:
        // While recording, RMB stops the take (don't enter normal rotate-mode).
        if (g_app.recMode == AppState::RecMode::Recording)
        {
            // signal stop; main loop saves the clip.
            g_app.recMode = AppState::RecMode::Idle;
            return 0;
        }
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
    case WM_MOUSEWHEEL:
    {
        // Recording captures the cursor â€” let wheel through even if a hidden
        // ImGui window thinks it wants the mouse.
        if (ImGui::GetIO().WantCaptureMouse &&
            g_app.recMode != AppState::RecMode::Recording)
            return 0;
        short delta = (short)HIWORD(wp);
        float notches = (float)delta / (float)WHEEL_DELTA;
        g_app.camera.moveSpeed *= powf(1.2f, notches);
        g_app.camera.moveSpeed = std::clamp(g_app.camera.moveSpeed, 0.05f, 10000.0f);
        return 0;
    }
    case WM_MOUSEMOVE:
    {
        bool freeLook = g_app.rmbDown || g_app.recMode == AppState::RecMode::Recording;
        if (freeLook)
        {
            POINT cur;
            GetCursorPos(&cur);
            int dx = cur.x - g_app.lastMouse.x;
            int dy = cur.y - g_app.lastMouse.y;
            g_app.camera.yaw += dx * g_app.camera.lookSens;
            g_app.camera.pitch -= dy * g_app.camera.lookSens;
            const float lim = 1.55334f;
            g_app.camera.pitch = std::clamp(g_app.camera.pitch, -lim, lim);
            SetCursorPos(g_app.lastMouse.x, g_app.lastMouse.y);
        }
        return 0;
    }
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// ---- Camera recording / playback helpers ----
namespace recfx
{
constexpr uint32_t kMagic = 0x434D5243u; // 'CRMC'
const char* kDir = "recordings";

void EnsureDir()
{
    std::error_code ec;
    std::filesystem::create_directories(kDir, ec);
}

std::string MakeFilename()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf), "clip_%04d%02d%02d_%02d%02d%02d.cam",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return std::string(kDir) + "/" + buf;
}

bool Save(const std::string& path, const std::vector<AppState::CamSample>& s)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t mag = kMagic;
    uint32_t n = (uint32_t)s.size();
    f.write((const char*)&mag, 4);
    f.write((const char*)&n, 4);
    if (n) f.write((const char*)s.data(), n * sizeof(AppState::CamSample));
    return (bool)f;
}

bool Load(const std::string& path, std::vector<AppState::CamSample>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t mag = 0, n = 0;
    f.read((char*)&mag, 4); f.read((char*)&n, 4);
    if (mag != kMagic) return false;
    out.resize(n);
    if (n) f.read((char*)out.data(), n * sizeof(AppState::CamSample));
    return (bool)f;
}

void ScanDir(std::vector<std::string>& out)
{
    out.clear();
    EnsureDir();
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(kDir, ec))
    {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() == ".cam")
            out.push_back(e.path().filename().string());
    }
    std::sort(out.begin(), out.end());
}

// Catmull-Rom on 4 control points, t in [0,1].
inline float CR(float p0, float p1, float p2, float p3, float t)
{
    float t2 = t * t;
    float t3 = t2 * t;
    return 0.5f * (
        (2.0f * p1) +
        (-p0 + p2) * t +
        (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
        (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

// Subsample raw samples at fixed time intervals â†’ keyframes for spline.
struct Key { float t; float pos[3]; float yaw; float pitch; };
void BuildKeyframes(const std::vector<AppState::CamSample>& s,
                    float interval, std::vector<Key>& out)
{
    out.clear();
    if (s.empty()) return;
    // Build cumulative times.
    std::vector<float> times(s.size());
    float tAcc = 0.0f;
    for (size_t i = 0; i < s.size(); ++i)
    {
        tAcc += s[i].dt;
        times[i] = tAcc;
    }
    float total = times.back();
    if (interval <= 1e-3f)
    {
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i)
        {
            Key k; k.t = times[i];
            k.pos[0] = s[i].pos[0]; k.pos[1] = s[i].pos[1]; k.pos[2] = s[i].pos[2];
            k.yaw = s[i].yaw; k.pitch = s[i].pitch;
            out.push_back(k);
        }
        return;
    }
    auto lerpAt = [&](float t, Key& k)
    {
        if (t <= 0.0f) { k.pos[0]=s.front().pos[0]; k.pos[1]=s.front().pos[1]; k.pos[2]=s.front().pos[2]; k.yaw=s.front().yaw; k.pitch=s.front().pitch; return; }
        if (t >= total) { k.pos[0]=s.back().pos[0]; k.pos[1]=s.back().pos[1]; k.pos[2]=s.back().pos[2]; k.yaw=s.back().yaw; k.pitch=s.back().pitch; return; }
        size_t i = 1;
        while (i < times.size() && times[i] < t) ++i;
        float t0 = times[i - 1], t1 = times[i];
        float u = (t - t0) / std::max(1e-5f, t1 - t0);
        for (int a = 0; a < 3; ++a)
            k.pos[a] = s[i-1].pos[a] + (s[i].pos[a] - s[i-1].pos[a]) * u;
        k.yaw   = s[i-1].yaw   + (s[i].yaw   - s[i-1].yaw)   * u;
        k.pitch = s[i-1].pitch + (s[i].pitch - s[i-1].pitch) * u;
    };
    for (float t = 0.0f; t < total; t += interval)
    {
        Key k; k.t = t; lerpAt(t, k); out.push_back(k);
    }
    Key kEnd; kEnd.t = total; lerpAt(total, kEnd); out.push_back(kEnd);
}

void Evaluate(const std::vector<Key>& keys, float t,
              float outPos[3], float& outYaw, float& outPitch)
{
    if (keys.empty()) return;
    if (t <= keys.front().t) {
        outPos[0]=keys.front().pos[0]; outPos[1]=keys.front().pos[1]; outPos[2]=keys.front().pos[2];
        outYaw=keys.front().yaw; outPitch=keys.front().pitch; return;
    }
    if (t >= keys.back().t) {
        outPos[0]=keys.back().pos[0]; outPos[1]=keys.back().pos[1]; outPos[2]=keys.back().pos[2];
        outYaw=keys.back().yaw; outPitch=keys.back().pitch; return;
    }
    size_t i = 1;
    while (i < keys.size() && keys[i].t < t) ++i;
    size_t i0 = (i >= 2) ? i - 2 : 0;
    size_t i1 = i - 1;
    size_t i2 = i;
    size_t i3 = (i + 1 < keys.size()) ? i + 1 : i;
    float u = (t - keys[i1].t) / std::max(1e-5f, keys[i2].t - keys[i1].t);
    for (int a = 0; a < 3; ++a)
        outPos[a] = CR(keys[i0].pos[a], keys[i1].pos[a], keys[i2].pos[a], keys[i3].pos[a], u);
    outYaw   = CR(keys[i0].yaw,   keys[i1].yaw,   keys[i2].yaw,   keys[i3].yaw,   u);
    outPitch = CR(keys[i0].pitch, keys[i1].pitch, keys[i2].pitch, keys[i3].pitch, u);
}
} // namespace recfx

void UpdateCamera(float dt)
{
    if (ImGui::GetIO().WantTextInput)
        return;
    // Playback overrides camera entirely.
    if (g_app.recMode == AppState::RecMode::Playing)
    {
        if (g_app.keys[VK_ESCAPE])
        {
            g_app.recMode = AppState::RecMode::Idle;
            g_app.playPaused = false;
            return;
        }
        if (!g_app.playPaused)
            g_app.playT += dt;
        std::vector<recfx::Key> keys;
        recfx::BuildKeyframes(g_app.playSamples, g_app.smoothSec, keys);
        if (!keys.empty())
        {
            float p[3]; float yaw, pitch;
            recfx::Evaluate(keys, g_app.playT, p, yaw, pitch);
            g_app.camera.position = hlslpp::float3(p[0], p[1], p[2]);
            g_app.camera.yaw   = yaw;
            g_app.camera.pitch = pitch;
        }
        if (!g_app.playPaused && g_app.playT >= g_app.playTotal)
        {
            g_app.playT = g_app.playTotal;
            g_app.playPaused = true; // hold at end; user can scrub or stop
        }
        return;
    }
    Camera& c = g_app.camera;
    float speed = c.moveSpeed * dt;

    hlslpp::float3 f = c.forward();
    hlslpp::float3 r = c.right();
    hlslpp::float3 mv(0.0f, 0.0f, 0.0f);
    if (g_app.keys['W'])
        mv += f;
    if (g_app.keys['S'])
        mv -= f;
    if (g_app.keys['D'])
        mv += r;
    if (g_app.keys['A'])
        mv -= r;
    float vy = 0.0f;
    if (g_app.keys['E'] || g_app.keys[VK_SPACE])
        vy += 1.0f;
    if (g_app.keys['Q'] || g_app.keys[VK_CONTROL])
        vy -= 1.0f;
    mv += hlslpp::float3(0.0f, vy, 0.0f);

    float len = (float)hlslpp::length(mv);
    if (len > 1e-5f)
    {
        c.position += hlslpp::normalize(mv) * speed;
    }
}

void FrameMenuBar()
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Windows"))
        {
            ImGui::MenuItem("Controls", nullptr, &g_app.showControls);
            ImGui::MenuItem("FPS", nullptr, &g_app.showFps);
            ImGui::MenuItem("Stats", nullptr, &g_app.showStats);
            ImGui::MenuItem("Recording", nullptr, &g_app.showRecording);
            ImGui::MenuItem("GPU Profile", nullptr, &g_app.showGpuProfile);
            ImGui::MenuItem("GPU Resources", nullptr, &g_app.showGpuRes);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void FrameRecordingWindow()
{
    if (!g_app.showRecording) return;
    if (!ImGui::Begin("Recording", &g_app.showRecording))
    {
        ImGui::End();
        return;
    }
    if (!g_app.recDirScanned)
    {
        recfx::ScanDir(g_app.recClips);
        g_app.recDirScanned = true;
        if (g_app.recSelected >= (int)g_app.recClips.size())
            g_app.recSelected = -1;
    }

    const char* stateStr =
        g_app.recMode == AppState::RecMode::Recording ? "RECORDING (RMB to stop)" :
        g_app.recMode == AppState::RecMode::Playing   ? "PLAYING (Esc to stop)"   :
        "Idle";
    ImGui::Text("State: %s", stateStr);

    ImGui::BeginDisabled(g_app.recMode != AppState::RecMode::Idle);
    if (ImGui::Button("Start recording"))
    {
        g_app.recSamples.clear();
        g_app.recMode = AppState::RecMode::Recording;
    }
    ImGui::EndDisabled();

    if (g_app.recMode == AppState::RecMode::Recording)
        ImGui::Text("Samples: %zu", g_app.recSamples.size());

    ImGui::Separator();
    ImGui::SliderFloat("Smoothing (s)", &g_app.smoothSec, 0.0f, 10.0f, "%.2f");

    ImGui::Text("Clips:");
    ImGui::BeginChild("##clips", ImVec2(0, 200), true);
    for (int i = 0; i < (int)g_app.recClips.size(); ++i)
    {
        bool sel = (i == g_app.recSelected);
        if (ImGui::Selectable(g_app.recClips[i].c_str(), sel))
            g_app.recSelected = i;
    }
    ImGui::EndChild();

    bool canPlay = (g_app.recMode == AppState::RecMode::Idle) &&
                   (g_app.recSelected >= 0) &&
                   (g_app.recSelected < (int)g_app.recClips.size());
    ImGui::BeginDisabled(!canPlay);
    if (ImGui::Button("Play"))
    {
        std::string path = std::string(recfx::kDir) + "/" + g_app.recClips[g_app.recSelected];
        if (recfx::Load(path, g_app.playSamples) && !g_app.playSamples.empty())
        {
            g_app.playT = 0.0f;
            g_app.playTotal = 0.0f;
            for (auto& s : g_app.playSamples) g_app.playTotal += s.dt;
            g_app.playPaused = false;
            g_app.recMode = AppState::RecMode::Playing;
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Refresh"))
        g_app.recDirScanned = false;
    ImGui::SameLine();
    ImGui::BeginDisabled(!canPlay);
    if (ImGui::Button("Delete"))
    {
        std::error_code ec;
        std::filesystem::remove(std::string(recfx::kDir) + "/" +
                                g_app.recClips[g_app.recSelected], ec);
        g_app.recSelected = -1;
        g_app.recDirScanned = false;
    }
    ImGui::EndDisabled();

    // ---- Playback transport ----
    if (g_app.recMode == AppState::RecMode::Playing)
    {
        ImGui::Separator();
        if (ImGui::Button("Stop"))
        {
            g_app.recMode = AppState::RecMode::Idle;
            g_app.playPaused = false;
        }
        ImGui::SameLine();
        if (ImGui::Button(g_app.playPaused ? "Resume" : "Pause"))
            g_app.playPaused = !g_app.playPaused;
        ImGui::SameLine();
        if (ImGui::Button("Restart"))
        {
            g_app.playT = 0.0f;
            g_app.playPaused = false;
        }
        // Scrub bar.
        float t = g_app.playT;
        if (ImGui::SliderFloat("Time", &t, 0.0f, g_app.playTotal, "%.2f s"))
        {
            g_app.playT = std::clamp(t, 0.0f, g_app.playTotal);
            g_app.playPaused = true; // pause while scrubbing
        }
    }

    ImGui::End();
}

void FrameFpsWindow()
{
    g_app.fpsAvg = 1000.0f / (g_app.cpuFrameMs > 0.001 ? (float)g_app.cpuFrameMs : 16.0f);
    g_app.fpsHist[g_app.fpsHistIdx] = g_app.fpsAvg;
    g_app.fpsHistIdx = (g_app.fpsHistIdx + 1) % IM_ARRAYSIZE(g_app.fpsHist);

    if (!g_app.showFps)
        return;
    if (!ImGui::Begin("FPS", &g_app.showFps))
    {
        ImGui::End();
        return;
    }
    ImGui::Text("CPU frame: %.2f ms (%.0f FPS)", g_app.cpuFrameMs, g_app.fpsAvg);
    ImGui::PlotLines("FPS", g_app.fpsHist, IM_ARRAYSIZE(g_app.fpsHist),
                     g_app.fpsHistIdx, nullptr, 0.0f, 240.0f, ImVec2(0, 60));
    ImGui::End();
}

void FrameGpuResourcesWindow()
{
    if (!g_app.showGpuRes) return;
    if (!ImGui::Begin("GPU Resources", &g_app.showGpuRes))
    {
        ImGui::End();
        return;
    }
    auto fmtBytes = [](uint64_t b) -> std::string {
        char buf[32];
        if (b >= (1ull << 30))      std::snprintf(buf, sizeof(buf), "%.2f GB", b / 1073741824.0);
        else if (b >= (1ull << 20)) std::snprintf(buf, sizeof(buf), "%.2f MB", b / 1048576.0);
        else if (b >= (1ull << 10)) std::snprintf(buf, sizeof(buf), "%.1f KB", b / 1024.0);
        else                        std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)b);
        return std::string(buf);
    };
    const ImU32 colFill = IM_COL32(60, 200, 60, 255);
    const ImU32 colBack = IM_COL32(60, 60, 60, 160);

    // Find max per-LOD total bytes so bars use a common scale within the window.
    uint64_t maxLodTotal = 1;
    Renderer::StreamLodInfo cached[lw::kLodCount];
    for (int L = 0; L < lw::kLodCount; ++L)
    {
        cached[L] = g_app.renderer.GetStreamLodInfo(L);
        uint64_t t = 0;
        for (const auto& p : cached[L].pools) t += p.bytes;
        if (t > maxLodTotal) maxLodTotal = t;
    }

    for (int L = 0; L < lw::kLodCount; ++L)
    {
        const auto& li = cached[L];
        uint64_t totalBytes = 0;
        for (const auto& p : li.pools) totalBytes += p.bytes;
        ImGui::PushID(L);
        ImGui::Text("LOD %d  total %s", L, fmtBytes(totalBytes).c_str());
        for (const auto& p : li.pools)
        {
            ImGui::Text("  %-13s elem %u (stride %u)  %s",
                        p.name, p.numElements, p.stride, fmtBytes(p.bytes).c_str());
            ImVec2 size(ImGui::GetContentRegionAvail().x - 8.0f, 12.0f);
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), colBack, 2.0f);
            float w = size.x * (float)((double)p.bytes / (double)maxLodTotal);
            if (w > 0.0f)
                dl->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + size.y), colFill, 2.0f);
            ImGui::Dummy(size);
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::TextDisabled("Bars scaled to largest per-LOD total. Buffers are IMMUTABLE â€” used == capacity.");
    ImGui::End();
}

void FrameGpuProfileWindow()
{
    if (!g_app.showGpuProfile) return;
    if (!ImGui::Begin("GPU Profile", &g_app.showGpuProfile))
    {
        ImGui::End();
        return;
    }
    MicroProfile* mp = MicroProfileGet();
    const float fToMsGpu = MicroProfileTickToMsMultiplier(MicroProfileTicksPerSecondGpu());
    // Advance ring buffer first; record this frame's raw values into slot.
    int slot = g_app.gpuProfHistIdx;
    g_app.gpuProfHistIdx = (g_app.gpuProfHistIdx + 1) %
                           AppState::kGpuProfHistory;
    for (uint32_t i = 0; i < mp->nTotalTimers; ++i)
    {
        const MicroProfileTimerInfo& ti = mp->TimerInfo[i];
        const MicroProfileGroupInfo& gi = mp->GroupInfo[ti.nGroupIndex];
        if (gi.Type != MicroProfileTokenTypeGpu) continue;
        float ms = (float)mp->Frame[i].nTicks * fToMsGpu;
        bool fired = mp->Frame[i].nCount > 0;
        if (i < 1024)
        {
            g_app.gpuProfHist[i][slot]  = ms;
            g_app.gpuProfFired[i][slot] = fired;
        }
    }
    // Sort entries by smoothed ms desc for readability.
    struct Row {
        uint32_t idx;
        float raw;
        float avg;
        float maxV;
    };
    std::vector<Row> rows;
    rows.reserve(64);
    for (uint32_t i = 0; i < mp->nTotalTimers && i < 1024; ++i)
    {
        const MicroProfileTimerInfo& ti = mp->TimerInfo[i];
        const MicroProfileGroupInfo& gi = mp->GroupInfo[ti.nGroupIndex];
        if (gi.Type != MicroProfileTokenTypeGpu) continue;
        float sum = 0.0f, mx = 0.0f;
        int   firedCount = 0;
        for (int s = 0; s < AppState::kGpuProfHistory; ++s)
        {
            if (!g_app.gpuProfFired[i][s]) continue;
            float v = g_app.gpuProfHist[i][s];
            sum += v;
            ++firedCount;
            if (v > mx) mx = v;
        }
        float avg = (firedCount > 0) ? (sum / (float)firedCount) : 0.0f;
        float raw = g_app.gpuProfFired[i][slot] ? g_app.gpuProfHist[i][slot] : 0.0f;
        // Stick the row on first non-zero appearance; from then on it's shown
        // even if its current avg drops to 0. Keeps positions stable.
        if (mx > 0.0f) g_app.gpuProfSeen[i] = true;
        if (!g_app.gpuProfSeen[i]) continue;
        rows.push_back({i, raw, avg, mx});
    }
    // Âµs/pixel â€” bar widths stay constant across frames + toggles.
    ImGui::SetNextItemWidth(180.0f);
    ImGui::SliderFloat("Âµs / pixel", &g_app.gpuProfUsPerPx, 0.5f, 200.0f, "%.1f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SameLine();
    ImGui::Text("smooth %df    active %d", AppState::kGpuProfHistory, (int)rows.size());

    const float usPerPx = std::max(0.1f, g_app.gpuProfUsPerPx);
    auto barPxFor = [&](float ms) { return (ms * 1000.0f) / usPerPx; };

    const float rowH    = ImGui::GetTextLineHeightWithSpacing();
    const ImU32 colAvg  = IM_COL32( 80, 180,  80, 255);
    const ImU32 colRaw  = IM_COL32(240, 220,  60, 200);
    const ImU32 colMax  = IM_COL32(255,  80,  80, 200);
    const ImU32 colGrid = IM_COL32(120, 120, 120,  60);

    if (ImGui::BeginTable("gpu_prof_bars", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                          ImGuiTableFlags_ScrollY))
    {
        ImGui::TableSetupColumn("Marker", ImGuiTableColumnFlags_WidthFixed, 160.0f);
        ImGui::TableSetupColumn("Avg",    ImGuiTableColumnFlags_WidthFixed,  64.0f);
        ImGui::TableSetupColumn("Raw",    ImGuiTableColumnFlags_WidthFixed,  64.0f);
        ImGui::TableSetupColumn("Bar",    ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (const Row& r : rows)
        {
            const MicroProfileTimerInfo& ti = mp->TimerInfo[r.idx];
            const MicroProfileGroupInfo& gi = mp->GroupInfo[ti.nGroupIndex];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s/%s", gi.pName, ti.pName);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.3f", r.avg);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.3f", r.raw);
            ImGui::TableSetColumnIndex(3);
            ImVec2 cursor = ImGui::GetCursorScreenPos();
            float avail = ImGui::GetContentRegionAvail().x;
            float h = rowH - 4.0f;
            // Reserve space so the table row claims the right height.
            ImGui::Dummy(ImVec2(avail, rowH));
            float barX0 = cursor.x;
            float barY  = cursor.y + 2.0f;
            float avgW  = std::min(avail, barPxFor(r.avg));
            float rawW  = std::min(avail, barPxFor(r.raw));
            float maxX  = barX0 + std::min(avail, barPxFor(r.maxV));
            for (int ms = 1; ms <= 64; ++ms)
            {
                float gx = barX0 + barPxFor((float)ms);
                if (gx > barX0 + avail) break;
                dl->AddLine(ImVec2(gx, barY),
                            ImVec2(gx, barY + h),
                            colGrid, 1.0f);
            }
            if (avgW > 0.0f)
                dl->AddRectFilled(ImVec2(barX0, barY),
                                  ImVec2(barX0 + avgW, barY + h),
                                  colAvg, 2.0f);
            if (rawW > 0.0f)
                dl->AddRectFilled(ImVec2(barX0, barY),
                                  ImVec2(barX0 + rawW, barY + h * 0.35f),
                                  colRaw, 2.0f);
            dl->AddLine(ImVec2(maxX, barY),
                        ImVec2(maxX, barY + h),
                        colMax, 2.0f);
        }
        ImGui::EndTable();
    }
    ImGui::Dummy(ImVec2(0, 4));
    ImGui::TextDisabled("green = avg10ms    yellow = raw    red = max10ms    "
                        "grey ticks = 1 ms");
    ImGui::End();
}

void FrameStatsWindow()
{
    if (!g_app.showStats)
        return;
    if (!ImGui::Begin("Stats", &g_app.showStats))
    {
        ImGui::End();
        return;
    }

    auto mb = [](uint64_t b)
    { return (double)b / (1024.0 * 1024.0); };

    ImGui::Text("Frame");
    {
        uint32_t total = g_app.renderer.LastDrawnCount();
        uint32_t splat = g_app.renderer.LastSplatDrawCalls();
        uint32_t poly = g_app.renderer.LastPolyDrawCalls();
        uint32_t fast = g_app.renderer.LastFastDrawCalls();
        ImGui::Text("  Draw calls:      %u  (splat %u  +  poly %u)", total, splat, poly);
        ImGui::Text("  Fast draws:      %u / %u  (same chunk as prev, CB-only update)",
                    fast, total);

        uint64_t voxTot = g_app.renderer.LastPointCount();
        uint64_t voxSpl = g_app.renderer.LastSplatVoxelCount();
        uint64_t voxPoly = g_app.renderer.LastPolyVoxelCount();
        uint64_t tris = g_app.renderer.LastTriCount();
        ImGui::Text("  Voxels drawn:    %llu  (splat %llu  +  poly %llu)",
                    (unsigned long long)voxTot,
                    (unsigned long long)voxSpl,
                    (unsigned long long)voxPoly);
        ImGui::Text("  Triangles drawn: %llu  (12 per poly voxel)",
                    (unsigned long long)tris);
        ImGui::Text("  Splat points:    %llu  (1 point primitive each)",
                    (unsigned long long)voxSpl);

        uint64_t blk = g_app.renderer.LastBlockTotal();
        uint32_t blkDisp = g_app.renderer.LastBlockDispatches();
        if (blk || blkDisp)
        {
            // Each block = 1 thread = up to 8 voxel atomics ("octet").
            // No GPU readback for actual pixels-written, so report block count
            // and upper-bound atomic count (= blocks * 8).
            ImGui::Text("  Block CS:        %llu blocks  (â‰¤ %llu atomics)  %u dispatches",
                        (unsigned long long)blk,
                        (unsigned long long)(blk * 8ull),
                        blkDisp);
            ImGui::Text("    per-LOD blocks: L0=%llu L1=%llu L2=%llu L3=%llu L4=%llu",
                        (unsigned long long)g_app.renderer.LastBlockTotalAt(0),
                        (unsigned long long)g_app.renderer.LastBlockTotalAt(1),
                        (unsigned long long)g_app.renderer.LastBlockTotalAt(2),
                        (unsigned long long)g_app.renderer.LastBlockTotalAt(3),
                        (unsigned long long)g_app.renderer.LastBlockTotalAt(4));
        }
    }

    if (g_app.renderer.HasLwWorld())
    {
        ImGui::Separator();
        ImGui::Text("LW (lodworld)");
        uint64_t totBytes = 0;
        uint32_t totChunks = 0;
        uint64_t totPoints = 0;
        for (int L = 0; L < lw::kLodCount; ++L)
        {
            uint32_t s = g_app.renderer.LwSlotCount(L);
            uint32_t pc = g_app.renderer.LwPointCount(L);
            uint64_t bs = g_app.renderer.LwBytes(L);
            ImGui::Text("  L%d chunks=%-4u points=%9u  %7.2f MB", L, s, pc, mb(bs));
            totChunks += s;
            totPoints += pc;
            totBytes += bs;
        }
        ImGui::Text("  TOTAL chunks=%u points=%llu  %.2f MB",
                    totChunks, (unsigned long long)totPoints, mb(totBytes));
    }

    ImGui::Separator();
    ImGui::Text("GPU (shared)");
    ImGui::Text("  Splat RTs/UAVs:   %7.2f MB", mb(g_app.renderer.SplatRtBytes()));

    ImGui::Separator();
    ImGui::Text("Camera");
    ImGui::Text("  Pos: %.1f %.1f %.1f",
                (float)g_app.camera.position.x,
                (float)g_app.camera.position.y,
                (float)g_app.camera.position.z);
    ImGui::Text("  Yaw: %.2f  Pitch: %.2f", g_app.camera.yaw, g_app.camera.pitch);

    if (!g_app.sceneReady)
    {
        ImGui::Separator();
        ImGui::TextUnformatted(g_app.loadStatus.c_str());
    }

    ImGui::End();
}

void FrameControlsWindow()
{
    if (!g_app.showControls)
        return;
    if (!ImGui::Begin("Controls", &g_app.showControls))
    {
        ImGui::End();
        return;
    }

    if (g_app.graphicsAdapters.size() > 1)
    {
        std::vector<const char*> names;
        names.reserve(g_app.graphicsAdapters.size() + 1);
        names.push_back("(system default)");
        for (auto& n : g_app.graphicsAdapters)
            names.push_back(n.c_str());
        int sel = (g_app.adapterIdx < 0) ? 0 : (g_app.adapterIdx + 1);
        if (sel >= (int)names.size())
            sel = 0;
        if (ImGui::Combo("GPU adapter", &sel, names.data(), (int)names.size()))
        {
            g_app.adapterIdx = (sel == 0) ? -1 : (sel - 1);
            Settings st = LoadSettings();
            st.adapterIdx = g_app.adapterIdx;
            SaveSettings(st);
        }
        if (g_app.adapterIdx != g_app.activeAdapterIdx)
        {
            ImGui::TextColored(ImVec4(1, 0.7f, 0.2f, 1), "Restart to apply");
        }
    }
    {
        // Preset window resolutions; shows "Custom" when the client area
        // was resized manually. WM_SIZE resizes the swapchain.
        static const POINT kRes[] = {{1280, 720}, {1920, 1080}, {2560, 1440}};
        static const char* kResNames[] = {"Custom", "1280x720", "1920x1080", "2560x1440"};
        RECT cr{};
        GetClientRect(g_app.hwnd, &cr);
        int sel = 0;
        for (int i = 0; i < IM_ARRAYSIZE(kRes); ++i)
        {
            if (cr.right == kRes[i].x && cr.bottom == kRes[i].y)
            {
                sel = i + 1;
                break;
            }
        }
        if (ImGui::Combo("Resolution", &sel, kResNames, IM_ARRAYSIZE(kResNames)) && sel > 0)
        {
            g_app.pendingClientW = kRes[sel - 1].x;
            g_app.pendingClientH = kRes[sel - 1].y;
        }
    }
    {
        // Dataset combo built from discovered assets/*.vox at startup.
        std::vector<const char*> names;
        names.reserve(g_app.datasetPaths.size());
        for (const auto& p : g_app.datasetPaths)
            names.push_back(p.c_str());
        if (names.empty())
        {
            ImGui::TextUnformatted("Dataset: (no .vox files in assets/)");
        }
        else
        {
            int idx = g_app.datasetIdx;
            if (idx < 0 || idx >= (int)names.size())
                idx = 0;
            if (ImGui::Combo("Dataset", &idx, names.data(), (int)names.size()))
            {
                g_app.datasetIdx = idx;
                SaveLastVoxToSettings(g_app.datasetPaths[idx]);
                g_app.reloadRequested.store(true);
            }
        }
    }
    // Tech pulldowns â€” only PointCS_Block wired today; others reserved for
    // future octet-based revivals. Close/Far checkboxes gate each ring.
    struct TechEntry { const char* name; RenderTech val; };
    static const TechEntry kTechList[] = {
        {"PointCS_Block",   RenderTech::PointCS_Block},
        {"OctetBillboards", RenderTech::OctetBillboards},
        {"PolyAxis",        RenderTech::PolyAxis},
        {"Splat",           RenderTech::Splat},
    };
    const int kTechCount = (int)(sizeof(kTechList) / sizeof(kTechList[0]));
    auto techIdxFrom = [&](RenderTech v) -> int {
        for (int k = 0; k < kTechCount; ++k)
            if (kTechList[k].val == v) return k;
        return 0;
    };
    {
        const char* techNames[16];
        for (int k = 0; k < kTechCount; ++k) techNames[k] = kTechList[k].name;
        int tt = techIdxFrom(g_app.tech);
        ImGui::PushItemWidth(180.0f);
        if (ImGui::Combo("##TechClose", &tt, techNames, kTechCount, kTechCount))
            g_app.tech = kTechList[tt].val;
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Checkbox("Close", &g_app.closeEnabled);
        int tf = techIdxFrom(g_app.techFar);
        ImGui::PushItemWidth(180.0f);
        if (ImGui::Combo("##TechFar", &tf, techNames, kTechCount, kTechCount))
            g_app.techFar = kTechList[tf].val;
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Checkbox("Far", &g_app.farEnabled);
    }

    ImGui::Separator();
    const char* lods[] = {"L0 (1 per voxel)", "L1 (2x2x2)", "L2 (4x4x4)", "L3 (8x8x8)", "Auto"};
    int lo = (int)g_app.pointLod;
    if (ImGui::Combo("Point LOD", &lo, lods, IM_ARRAYSIZE(lods)))
    {
        g_app.pointLod = (PointLod)lo;
    }
    ImGui::SliderFloat("LOD distance", &g_app.pointLodScale, 0.25f, 8.0f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Splat radius", &g_app.splatRadius, 1, 16);
    const char* modes[] = {"Lit", "Flat Color", "Normals", "AO", "AO+LODViz"};
    int m = (int)g_app.mode;
    if (ImGui::Combo("Shading", &m, modes, IM_ARRAYSIZE(modes)))
    {
        g_app.mode = (ShadingMode)m;
    }
    ImGui::Checkbox("VSync", &g_app.vsync);
    ImGui::Checkbox("TAA", &g_app.taa);

    if (ImGui::BeginTabBar("##controlTabs"))
    {
        if (ImGui::BeginTabItem("Render"))
        {
            ImGui::SliderFloat("Sun pitch", &g_app.sunPitchDeg, 5.0f, 89.0f, "%.1f deg");
            ImGui::SliderFloat("Sun yaw", &g_app.sunYawDeg, -180.0f, 180.0f, "%.1f deg");
            ImGui::SliderFloat("Sun intensity", &g_app.sunIntensityEV, -4.0f, 4.0f, "%.2f EV");
            ImGui::SliderFloat("Exposure", &g_app.exposureEV, -3.0f, 3.0f, "%.2f EV");
            ImGui::SliderFloat("Roughness", &g_app.roughness, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Ambient boost", &g_app.ambient, 0.0f, 2.0f, "%.2f");
            // Baked per-face AO darkening (Lit mode). Always visible.
            ImGui::SliderFloat("Baked AO strength", &g_app.aoStrength, 0.0f, 1.0f, "%.2f");
            ImGui::Checkbox("LW: draw chunk bounds (LOD coloured)", &g_app.lwShowBounds);
            ImGui::Checkbox("LW: Cheap top-down AO", &g_app.cheapAO);
            if (g_app.cheapAO)
            {
                ImGui::SliderFloat("  AO fade units", &g_app.aoFadeUnits,  1.0f, 64.0f, "%.1f");
                ImGui::SliderFloat("  AO push (tx)",  &g_app.aoPushTexels, 0.0f, 8.0f,  "%.1f");
            }
            {
                float rs = g_app.streamRadiusScale.load();
                if (ImGui::SliderFloat("Stream radius", &rs, 0.25f, 8.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
                {
                    g_app.streamRadiusScale.store(rs);
                    g_app.loaderTrigger.store(true);
                    g_app.loaderCv.notify_one();
                }
            }
            ImGui::Checkbox("Skip backbuffer clear (post writes all px)", &g_app.skipBackbufferClear);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Camera"))
        {
            if (ImGui::Button("Copy camera (pos+rot)"))
            {
                float cp[3];
                hlslpp::store(cp, g_app.camera.position);
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "pos=(%.4f, %.4f, %.4f) yaw=%.6f pitch=%.6f fov=%.2f",
                              cp[0], cp[1], cp[2],
                              g_app.camera.yaw, g_app.camera.pitch, g_app.camera.fovDeg);
                ImGui::SetClipboardText(buf);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(yaw/pitch in radians)");
            ImGui::SliderFloat("Move speed", &g_app.camera.moveSpeed, 0.1f, 5000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("FOV", &g_app.camera.fovDeg, 30.0f, 110.0f, "%.0f");
            ImGui::Separator();
            if (ImGui::Button("View 1 (auto-fit)"))
            {
                if (g_app.view1Valid)
                {
                    g_app.camera.position = hlslpp::float3(
                        g_app.view1Pos[0], g_app.view1Pos[1], g_app.view1Pos[2]);
                    g_app.camera.yaw = g_app.view1Yaw;
                    g_app.camera.pitch = g_app.view1Pitch;
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled(g_app.view1Valid ? "captured at first load"
                                                  : "(load a scene first)");
            if (ImGui::Button("View 2 (curated)"))
            {
                g_app.camera.position = hlslpp::float3(274.4363f, 274.3316f, 353.6918f);
                g_app.camera.yaw = 0.802501f;
                g_app.camera.pitch = -0.505000f;
                g_app.camera.fovDeg = 70.0f;
            }
            ImGui::Separator();
            ImGui::TextUnformatted("RMB drag: look | WASD: move | Wheel: speed | Q/E or Ctrl/Space: down/up");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Shadow"))
        {
            ImGui::Checkbox("Sun shadows", &g_app.sunShadows);
            ImGui::SliderInt("Cascades", &g_app.shadowCascades, 1, 4, "%d (only 1 wired)");
            {
                const char* sizes[] = {"512", "1024", "2048", "4096"};
                int idx = std::clamp(g_app.shadowMapSizeIdx, 0, 3);
                if (ImGui::Combo("Shadow map", &idx, sizes, IM_ARRAYSIZE(sizes)))
                {
                    g_app.shadowMapSizeIdx = idx;
                }
            }
            ImGui::SliderFloat("Shadow bias", &g_app.shadowBias, 0.0f, 0.01f, "%.4f");
            ImGui::Checkbox("Shadow cull front (else back)", &g_app.shadowCullFront);
            ImGui::Checkbox("Force shadow rebuild (profiling)", &g_app.shadowForceRebuild);
            {
                const char* slods[] = {"Auto", "L0", "L1", "L2", "L3"};
                ImGui::Combo("Shadow LOD", &g_app.shadowLodIdx, slods, IM_ARRAYSIZE(slods));
            }
            ImGui::Checkbox("Shadow blur fill (empty texels = neighbour avg)", &g_app.shadowBlur);

            ImGui::Separator();
            ImGui::Text("Shadow map preview");
            D3D12_GPU_DESCRIPTOR_HANDLE srv = g_app.shadowBlur
                                                ? g_app.renderer.ShadowFilledSrv()
                                                : g_app.renderer.ShadowSrv();
            const uint32_t smSize = g_app.renderer.ShadowMapSize();
            if (srv.ptr && smSize > 0)
            {
                float avail = ImGui::GetContentRegionAvail().x;
                float side = std::max(64.0f, std::min(avail, 512.0f));
                ImGui::Image((ImTextureID)srv.ptr, ImVec2(side, side));
                ImGui::Text("source: %s  size: %u x %u",
                            g_app.shadowBlur ? "filled (blurred)" : "raw caster",
                            smSize, smSize);
            }
            else
            {
                ImGui::TextUnformatted("(no shadow map â€” enable Sun shadows)");
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Fog"))
        {
            const char* fogModes[] = {"Off", "Depth"};
            int fm = g_app.fogMode;
            if (ImGui::Combo("Mode", &fm, fogModes, IM_ARRAYSIZE(fogModes)))
            {
                g_app.fogMode = fm;
            }
            ImGui::SliderFloat("Density", &g_app.fogDensity, 0.0f, 0.005f, "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Height density", &g_app.heightFogDensity, 0.0f, 5.0f, "%.3f");
            ImGui::SliderFloat("Height falloff", &g_app.heightFogFalloff, 0.0f, 0.2f, "%.4f");
            ImGui::SliderFloat("Height start Y", &g_app.heightFogStart, -100.0f, 200.0f, "%.1f");
            ImGui::ColorEdit3("Color", g_app.fogColor);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("GodRays"))
        {
            ImGui::SliderFloat("Strength", &g_app.godrayStrength, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Angle (deg)", &g_app.godrayAngleDeg, 0.05f, 30.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Temporal blend", &g_app.godrayEmaAlpha, 0.02f, 1.0f, "%.2f (1 = none)");
            ImGui::Checkbox("Anisotropic blur (SampleGrad)", &g_app.godrayAniso);
            ImGui::Checkbox("Separable blur (2-pass sparse)", &g_app.godraySeparable);
            ImGui::SliderInt("Stride (px)", &g_app.godraySeparableStride, 1, 12);
            ImGui::ColorEdit3("Tint", g_app.godrayTint);
            ImGui::Separator();
            ImGui::BeginGroup();
            ImGui::TextUnformatted("Mark (pre-blur):");
            { auto s = g_app.renderer.GodrayMarkSrv();
              if (s.ptr) ImGui::Image((ImTextureID)s.ptr, ImVec2(192, 192)); }
            ImGui::EndGroup();
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::TextUnformatted("Blurred (sampled):");
            { auto s = g_app.renderer.GodraySrv();
              if (s.ptr) ImGui::Image((ImTextureID)s.ptr, ImVec2(192, 192)); }
            ImGui::EndGroup();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

} // namespace

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int)
{
    // Attach a console window so printf / fprintf(stderr) become visible.
    // Reuse parent console if launched from one (e.g. bash); otherwise allocate.
    if (!AttachConsole(ATTACH_PARENT_PROCESS))
        AllocConsole();
    FILE* fOut = nullptr;
    FILE* fErr = nullptr;
    freopen_s(&fOut, "CONOUT$", "w", stdout);
    freopen_s(&fErr, "CONOUT$", "w", stderr);
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    SetCwdToProjectRoot();
    WNDCLASSEXW wc = {sizeof(wc)};
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"VoxelTestWnd";
    RegisterClassExW(&wc);

    DWORD style = WS_OVERLAPPEDWINDOW;
    RECT rc = {0, 0, 1600, 900};
    AdjustWindowRect(&rc, style, FALSE);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"VoxelTest",
                                style, CW_USEDEFAULT, CW_USEDEFAULT,
                                rc.right - rc.left, rc.bottom - rc.top,
                                nullptr, nullptr, hInst, nullptr);
    g_app.hwnd = hwnd;

    // Place on monitor index from settings (0-based EnumDisplayMonitors order;
    // stable per boot but not guaranteed to match Display Settings numbering).
    // monitorIdx < 0 â†’ leave default placement.
    {
        int targetIdx = LoadSettings().monitorIdx;
        if (targetIdx >= 0)
        {
            struct EnumCtx { std::vector<RECT> rects; };
            EnumCtx ec;
            EnumDisplayMonitors(nullptr, nullptr,
                [](HMONITOR mon, HDC, LPRECT, LPARAM lp) -> BOOL {
                    MONITORINFO mi = {sizeof(mi)};
                    if (GetMonitorInfoW(mon, &mi))
                        reinterpret_cast<EnumCtx*>(lp)->rects.push_back(mi.rcWork);
                    return TRUE;
                },
                reinterpret_cast<LPARAM>(&ec));
            if ((int)ec.rects.size() > targetIdx)
            {
                const RECT& mr = ec.rects[targetIdx];
                int winW = rc.right - rc.left;
                int winH = rc.bottom - rc.top;
                int monW = mr.right - mr.left;
                int monH = mr.bottom - mr.top;
                int x = mr.left + (monW - winW) / 2;
                int y = mr.top + (monH - winH) / 2;
                SetWindowPos(hwnd, nullptr, x, y, winW, winH, SWP_NOZORDER | SWP_NOACTIVATE);
            }
        }
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    g_app.graphicsAdapters = Renderer::EnumerateAdapters();
    {
        Settings st = LoadSettings();
        g_app.adapterIdx = st.adapterIdx;
    }
    g_app.activeAdapterIdx = g_app.adapterIdx;
    if (!g_app.renderer.Init(hwnd, g_app.adapterIdx))
    {
        MessageBoxA(nullptr, "Renderer init failed", "VoxelTest", MB_ICONERROR);
        return 1;
    }

    MicroProfileOnThreadCreate("Main");
    MicroProfileSetForceEnable(true);
    MicroProfileSetEnableAllGroups(true);
    MicroProfileSetForceMetaCounters(true);
    MicroProfileGpuInitD3D12(g_app.renderer.Device(), g_app.renderer.CommandQueue());
    MicroProfileWebServerStart();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(hwnd);
    {
        ImGui_ImplDX12_InitInfo info{};
        info.Device              = g_app.renderer.Device();
        info.CommandQueue        = g_app.renderer.CommandQueue();
        info.NumFramesInFlight   = (int)Renderer::kFrameCount;
        info.RTVFormat           = g_app.renderer.BackBufferFormat();
        info.DSVFormat           = DXGI_FORMAT_UNKNOWN;
        info.SrvDescriptorHeap   = g_app.renderer.SrvHeapForImGui();
        info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo*,
                                        D3D12_CPU_DESCRIPTOR_HANDLE* outCpu,
                                        D3D12_GPU_DESCRIPTOR_HANDLE* outGpu)
        { g_app.renderer.ImGuiSrvAlloc(outCpu, outGpu); };
        info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo*,
                                       D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                                       D3D12_GPU_DESCRIPTOR_HANDLE gpu)
        { g_app.renderer.ImGuiSrvFree(cpu, gpu); };
        ImGui_ImplDX12_Init(&info);
    }

    // Enumerate available datasets + restore last-used selection.
    g_app.datasetPaths = DiscoverDatasets();
    {
        std::string last = LoadLastVoxFromSettings();
        if (!last.empty())
        {
            for (int i = 0; i < (int)g_app.datasetPaths.size(); ++i)
            {
                if (g_app.datasetPaths[i] == last)
                {
                    g_app.datasetIdx = i;
                    break;
                }
            }
        }
    }

    // Continuous loader thread (Phase 2a): runs forever, re-streams when
    // main signals a meaningful camera move. Per-LOD shell radii decide
    // which chunks load each cycle. No eviction yet â€” Phase 2b adds slot
    // pool + per-chunk deltas.
    auto pickDataset = []() -> std::string
    {
        if (g_app.datasetPaths.empty())
            return "";
        int idx = g_app.datasetIdx;
        if (idx < 0 || idx >= (int)g_app.datasetPaths.size())
            idx = 0;
        return g_app.datasetPaths[idx];
    };

    auto loaderBody = []()
    {
        MicroProfileOnThreadCreate("LwLoader");
        const std::string voxPath = g_app.currentVoxPath;
        bool firstCycle = true;
        while (!g_app.loaderQuit.load())
        {
            MICROPROFILE_SCOPEI("Loader", "Cycle", 0xff40c0ff);
            auto tCycleStart = std::chrono::steady_clock::now();
            lw::StreamCfg cfg{};
            cfg.camX = g_app.camPosAtomic[0].load();
            cfg.camY = g_app.camPosAtomic[1].load();
            cfg.camZ = g_app.camPosAtomic[2].load();
            // Per-LOD shell radii (world units). Tighter than before so each
            // cycle finishes faster + finer LODs aren't dragged down by huge
            // LOD3 shells. LOD4 always full (small world-wide coarse view).
            const float rs = g_app.streamRadiusScale.load();
            cfg.radius[4] = 0.0f;                                     // full
            cfg.radius[3] = rs * 2.0f * (float)lw::kChunkVoxX * 8.0f; // 4096 @1.0
            cfg.radius[2] = rs * 2.0f * (float)lw::kChunkVoxX * 4.0f; // 2048 @1.0
            cfg.radius[1] = rs * 2.0f * (float)lw::kChunkVoxX * 2.0f; // 1024 @1.0
            cfg.radius[0] = rs * 2.0f * (float)lw::kChunkVoxX * 1.0f; // 512 @1.0
            cfg.hasFrustum = g_app.frustumValid.load();
            if (cfg.hasFrustum)
            {
                for (int i = 0; i < 24; ++i)
                {
                    ((float*)cfg.frustumPlanes)[i] = g_app.frustumAtomic[i].load();
                }
            }
            // First cycle: AABB might not be primed if main hasn't placed
            // the camera yet. Use file header AABB center as fallback.
            if (firstCycle)
            {
                FILE* fHdr = fopen(voxPath.c_str(), "rb");
                if (fHdr)
                {
                    lw::FileHeader fh{};
                    if (fread(&fh, sizeof(fh), 1, fHdr) == 1 && fh.magic == lw::kFileMagic)
                    {
                        cfg.camX = 0.5f * (float)(fh.worldAabbMin[0] + fh.worldAabbMax[0]);
                        cfg.camY = (float)fh.worldAabbMax[1];
                        cfg.camZ = 0.5f * (float)(fh.worldAabbMin[2] + fh.worldAabbMax[2]);
                    }
                    fclose(fHdr);
                }
                firstCycle = false;
            }
            // Wait for main to consume any previous-cycle uploads still pending
            // (flag==0). Resetting flags too early loses the last LODs of the
            // previous cycle and they never get uploaded.
            for (int L = 0; L < lw::kLodCount; ++L)
            {
                while (g_app.lodReadyFlag[L].load() == 0 && !g_app.loaderQuit.load())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // Reset per-LOD ready flags so main waits for the new load.
            for (int L = 0; L < lw::kLodCount; ++L)
                g_app.lodReadyFlag[L].store(-1);
            std::string err;
            // Decode into a thread-local World, then per-LOD move into the
            // shared pendingLwWorld under that LOD's mutex. Keeps the lock
            // held only for the move + signal, not the full decode.
            static thread_local lw::World tlsWorld;
            tlsWorld = lw::World{};
            auto onLod = [](void* user, int L)
            {
                lw::World* tls = (lw::World*)user;
                {
                    std::lock_guard<std::mutex> lk(g_app.lodMu[L]);
                    g_app.pendingLwWorld.worldAabbMin[0] = tls->worldAabbMin[0];
                    g_app.pendingLwWorld.worldAabbMin[1] = tls->worldAabbMin[1];
                    g_app.pendingLwWorld.worldAabbMin[2] = tls->worldAabbMin[2];
                    g_app.pendingLwWorld.worldAabbMax[0] = tls->worldAabbMax[0];
                    g_app.pendingLwWorld.worldAabbMax[1] = tls->worldAabbMax[1];
                    g_app.pendingLwWorld.worldAabbMax[2] = tls->worldAabbMax[2];
                    g_app.pendingLwWorld.lods[L] = std::move(tls->lods[L]);
                }
                // Heavy GPU upload prep (staging build, memcpy, resource create,
                // command-list record) runs HERE on the loader thread so the main
                // render thread never stalls. Main only executes the recorded copy
                // (CommitLwLod). lods[L] is owned by the loader until commit.
                g_app.renderer.PrepareLwLod(g_app.pendingLwWorld, L);
                g_app.lodReadyFlag[L].store(0);
            };
            bool ok = lw::LoadWorldStreaming(voxPath.c_str(), tlsWorld, err,
                                             onLod, &tlsWorld, &cfg);
            g_app.loadErr = err;
            g_app.loadOk.store(ok);
            g_app.loadDone.store(true);
            double tCycleMs = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - tCycleStart)
                                  .count();
            std::printf("[Loader] cycle done in %.1fms\n", tCycleMs);
            std::fflush(stdout);
            // Wait for main to trigger re-stream (cam moved) or quit.
            std::unique_lock<std::mutex> lk(g_app.loaderMu);
            g_app.loaderCv.wait(lk, []
                                { return g_app.loaderTrigger.load() || g_app.loaderQuit.load(); });
            g_app.loaderTrigger.store(false);
        }
    };

    auto kickLoader = [&loaderBody, &pickDataset]()
    {
        if (g_app.loaderThread.joinable())
        {
            g_app.loaderQuit.store(true);
            g_app.loaderCv.notify_all();
            g_app.loaderThread.join();
            g_app.loaderQuit.store(false);
        }
        g_app.currentVoxPath = pickDataset();
        if (g_app.currentVoxPath.empty())
        {
            g_app.loadErr = "no .vox files found in assets/";
            g_app.loadOk.store(false);
            g_app.loadDone.store(true);
            return;
        }
        g_app.loadStatus = std::string("Loading ") + g_app.currentVoxPath + "...";
        g_app.loaderThread = std::thread(loaderBody);
    };
    kickLoader();

    auto last = std::chrono::high_resolution_clock::now();

    MSG msg = {};
    while (!g_app.wantQuit)
    {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
                g_app.wantQuit = true;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (g_app.wantQuit)
            break;

        if (g_app.pendingClientW > 0)
        {
            RECT wr = {0, 0, g_app.pendingClientW, g_app.pendingClientH};
            AdjustWindowRect(&wr, (DWORD)GetWindowLongPtrW(g_app.hwnd, GWL_STYLE), FALSE);
            SetWindowPos(g_app.hwnd, nullptr, 0, 0, wr.right - wr.left, wr.bottom - wr.top,
                         SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
            g_app.pendingClientW = g_app.pendingClientH = 0;
        }

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        dt = std::min(dt, 0.1f);

        // Handle data-source reload request.
        if (g_app.reloadRequested.load() && g_app.sceneReady)
        {
            g_app.reloadRequested.store(false);
            g_app.sceneReady = false;
            g_app.everLoaded = false;
            g_app.loadDone.store(false);
            g_app.loadOk.store(false);
            for (int L = 0; L < lw::kLodCount; ++L)
                g_app.lodReadyFlag[L].store(-1);
            kickLoader();
        }

        // Per-LOD streaming: upload at most ONE ready LOD per frame so the
        // user sees coarse->fine progression instead of a single batch flash
        // when the worker thread finishes all LODs faster than the render
        // catches up.
        for (int L = lw::kLodCount - 1; L >= 0; --L)
        {
            int v = g_app.lodReadyFlag[L].load();
            if (v != 0)
                continue;
            if (!g_app.everLoaded || !g_app.sceneReady)
            {
                // First LOD to arrive: stash world metadata + place camera.
                g_app.renderer.PrepLwWorld(g_app.pendingLwWorld);
                const auto& w = g_app.pendingLwWorld;
                float cx = 0.5f * (float)(w.worldAabbMin[0] + w.worldAabbMax[0]);
                float cz = 0.5f * (float)(w.worldAabbMin[2] + w.worldAabbMax[2]);
                float dx = (float)(w.worldAabbMax[0] - w.worldAabbMin[0]);
                float dz = (float)(w.worldAabbMax[2] - w.worldAabbMin[2]);
                float ext = (dx > dz ? dx : dz);
                float topY = (float)w.worldAabbMax[1] + ext * 0.3f;
                // Capture legacy auto-fit values for the View1 button.
                g_app.view1Pos[0] = cx;
                g_app.view1Pos[1] = topY;
                g_app.view1Pos[2] = cz - ext * 0.5f;
                g_app.view1Yaw = 0.0f;
                g_app.view1Pitch = -0.5f;
                g_app.view1Valid = true;
                // Initial placement = View2 (curated).
                g_app.camera.position = hlslpp::float3(274.4363f, 274.3316f, 353.6918f);
                g_app.camera.yaw = 0.802501f;
                g_app.camera.pitch = -0.505000f;
                g_app.camera.fovDeg = 70.0f;
                g_app.camera.moveSpeed = ext * 0.05f;
                g_app.camera.farZ = ext * 4.0f + 1000.0f;
                g_app.everLoaded = true;
                g_app.sceneReady = true;
                g_app.loadStatus = "Streaming...";
                // First load placed the camera at View2; nudge the streamer so
                // the *next* cycle uses this camera position (rather than the
                // scene-center fallback used by firstCycle).
                g_app.lastTriggerCam[0] = -1e9f;
                g_app.lastTriggerCam[1] = -1e9f;
                g_app.lastTriggerCam[2] = -1e9f;
            }
            // Cheap: execute the loader-prepared copy + install buffers. No lock
            // (pendingUpload_[L] published via the lodReadyFlag release/acquire).
            g_app.renderer.CommitLwLod(L);
            g_app.lodReadyFlag[L].store(1);
            break; // one LOD per frame
        }

        // Combined-LOD GPU rebuild deferred until no more pending LOD uploads
        // remain. Avoids stuttering during streaming (each LOD upload would
        // otherwise rebuild ~100+ MB of concatenated SRVs).
        {
            bool anyPending = false;
            for (int L = 0; L < lw::kLodCount; ++L)
            {
                if (g_app.lodReadyFlag[L].load() == 0)
                {
                    anyPending = true;
                    break;
                }
            }
            if (!anyPending && g_app.sceneReady)
                g_app.renderer.FinalizeLwUploads();
        }

        // Loader thread completion: free CPU world struct + finalize status.
        if (g_app.loadDone.load() && g_app.loadStatus == "Streaming...")
        {
            g_app.pendingLwWorld = lw::World{};
            g_app.loadStatus = "Loaded.";
        }
        else if (g_app.loadDone.load() && !g_app.everLoaded && !g_app.loadOk.load())
        {
            g_app.loadStatus = "Load failed: " + g_app.loadErr;
            g_app.sceneReady = true;
        }

        // Manage cursor capture for recording (rmb-less fly mode).
        {
            bool wantCapture = (g_app.recMode == AppState::RecMode::Recording);
            if (wantCapture && !g_app.recCursorHidden)
            {
                GetCursorPos(&g_app.lastMouse);
                SetCapture(hwnd);
                ShowCursor(FALSE);
                g_app.recCursorHidden = true;
            }
            else if (!wantCapture && g_app.recCursorHidden && !g_app.rmbDown)
            {
                ReleaseCapture();
                ShowCursor(TRUE);
                g_app.recCursorHidden = false;
            }
        }

        // Capture pending recording stop: WM_RBUTTONDOWN flipped recMode to Idle.
        static AppState::RecMode prevRec = AppState::RecMode::Idle;
        if (prevRec == AppState::RecMode::Recording &&
            g_app.recMode == AppState::RecMode::Idle &&
            !g_app.recSamples.empty())
        {
            std::string path = recfx::MakeFilename();
            recfx::Save(path, g_app.recSamples);
            g_app.recSamples.clear();
            g_app.recDirScanned = false; // refresh list
        }
        prevRec = g_app.recMode;

        // View hotkeys â€” work in playback too. Only suppressed for text input.
        if (!ImGui::GetIO().WantTextInput)
        {
            if (g_app.keys['1']) g_app.mode = ShadingMode::Lit;
            if (g_app.keys['2']) g_app.mode = ShadingMode::Ao;
            if (g_app.keys['3']) g_app.mode = ShadingMode::LodViz;
        }

        UpdateCamera(dt);

        // Record sample after camera updated.
        if (g_app.recMode == AppState::RecMode::Recording)
        {
            AppState::CamSample s;
            s.dt = dt;
            float p[3]; hlslpp::store(p, g_app.camera.position);
            s.pos[0] = p[0]; s.pos[1] = p[1]; s.pos[2] = p[2];
            s.yaw   = g_app.camera.yaw;
            s.pitch = g_app.camera.pitch;
            g_app.recSamples.push_back(s);
        }

        // Publish cam pos + frustum planes for streaming loader; trigger
        // re-stream when moved > 1 chunk width.
        {
            float cp[3];
            hlslpp::store(cp, g_app.camera.position);
            g_app.camPosAtomic[0].store(cp[0]);
            g_app.camPosAtomic[1].store(cp[1]);
            g_app.camPosAtomic[2].store(cp[2]);
            // Frustum planes from view*proj (same extraction as renderer's).
            float aspect = (float)g_app.renderer.Width() / (float)std::max(1u, g_app.renderer.Height());
            hlslpp::float4x4 vM = g_app.camera.view();
            hlslpp::float4x4 pM = g_app.camera.proj(aspect);
            hlslpp::float4x4 vp = hlslpp::mul(vM, pM);
            float M[16];
            hlslpp::store(M, vp);
            // Same Gribb-Hartmann (row-major / vector*matrix) extraction as
            // renderer.cpp ExtractFrustumPlanes â€” 6 planes packed as ax+by+cz+d.
            float pl[6][4] = {
                {M[0] + M[3], M[4] + M[7], M[8] + M[11], M[12] + M[15]},  // left
                {M[3] - M[0], M[7] - M[4], M[11] - M[8], M[15] - M[12]},  // right
                {M[1] + M[3], M[5] + M[7], M[9] + M[11], M[13] + M[15]},  // bottom
                {M[3] - M[1], M[7] - M[5], M[11] - M[9], M[15] - M[13]},  // top
                {M[2], M[6], M[10], M[14]},                               // near
                {M[3] - M[2], M[7] - M[6], M[11] - M[10], M[15] - M[14]}, // far
            };
            for (int i = 0; i < 24; ++i)
                g_app.frustumAtomic[i].store(((float*)pl)[i]);
            g_app.frustumValid.store(true);
            float dx = cp[0] - g_app.lastTriggerCam[0];
            float dy = cp[1] - g_app.lastTriggerCam[1];
            float dz = cp[2] - g_app.lastTriggerCam[2];
            const float kReStreamDist = (float)lw::kChunkVoxX; // 1 LOD0 chunk
            if (dx * dx + dy * dy + dz * dz > kReStreamDist * kReStreamDist)
            {
                g_app.lastTriggerCam[0] = cp[0];
                g_app.lastTriggerCam[1] = cp[1];
                g_app.lastTriggerCam[2] = cp[2];
                g_app.loaderTrigger.store(true);
                g_app.loaderCv.notify_one();
            }
        }

        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        FrameMenuBar();
        FrameRecordingWindow();
        FrameFpsWindow();
        FrameStatsWindow();
        FrameControlsWindow();
        FrameGpuProfileWindow();
        FrameGpuResourcesWindow();
        ImGui::Render();

        float clear[4];
        if (g_app.fogMode != 0)
        {
            clear[0] = g_app.fogColor[0];
            clear[1] = g_app.fogColor[1];
            clear[2] = g_app.fogColor[2];
            clear[3] = 1.0f;
        }
        else
        {
            clear[0] = g_app.bgColor[0];
            clear[1] = g_app.bgColor[1];
            clear[2] = g_app.bgColor[2];
            clear[3] = g_app.bgColor[3];
        }
        // Main DSV only read by wireframe bounds debug path.
        bool dsvUnused = !g_app.lwShowBounds;
        g_app.renderer.BeginFrame(clear, g_app.skipBackbufferClear, dsvUnused);
        // Draw as soon as ANY LOD is uploaded â€” streaming flips sceneReady on
        // first LOD ready. loadOk only flips after the worker has finished
        // every LOD; gating on it hides the coarse scene until full load.
        if (g_app.sceneReady)
        {
            // LodViz forces fog off so the per-LOD colors aren't dimmed/tinted.
            const bool fogOff = (g_app.fogMode == 0) || (g_app.mode == ShadingMode::LodViz);
            float effFogDensity = fogOff ? 0.0f : g_app.fogDensity;
            float effHeightFogDen = (g_app.mode == ShadingMode::LodViz) ? 0.0f : g_app.heightFogDensity;
            const float kDeg2Rad = 3.14159265358979f / 180.0f;
            float pr = g_app.sunPitchDeg * kDeg2Rad;
            float yr = g_app.sunYawDeg * kDeg2Rad;
            float sunDir[3] = {
                cosf(pr) * sinf(yr),
                sinf(pr),
                cosf(pr) * cosf(yr),
            };

            DrawSceneParams ps;
            ps.mode = g_app.mode;
            ps.gridSize = g_app.gridSize;
            ps.tech = g_app.tech;
            ps.techFar = g_app.techFar;
            ps.closeEnabled = g_app.closeEnabled;
            ps.farEnabled = g_app.farEnabled;
            ps.pointLight = g_app.pointLight;
            ps.pointLod = g_app.pointLod;
            ps.pointLodScale = g_app.pointLodScale;
            ps.splatFilter = g_app.splatFilter;
            ps.splatDilate2Pass = g_app.splatDilate2Pass;
            ps.fogColor[0] = g_app.fogColor[0];
            ps.fogColor[1] = g_app.fogColor[1];
            ps.fogColor[2] = g_app.fogColor[2];
            ps.fogDensity = effFogDensity;
            ps.heightFogDensity = effHeightFogDen;
            ps.heightFogFalloff = g_app.heightFogFalloff;
            ps.heightFogStart = g_app.heightFogStart;
            ps.godrayStrength = g_app.godrayStrength;
            ps.godrayAngleDeg = g_app.godrayAngleDeg;
            ps.godrayAniso = g_app.godrayAniso;
            ps.godraySeparable = g_app.godraySeparable;
            ps.godraySeparableStride = g_app.godraySeparableStride;
            {
                // Boost alpha (less smoothing) on translation. Rotation is
                // fine â€” sun stays at same world dir, just on different screen
                // pixel; small history error. Translation shifts parallax so
                // depth pixels under each godray texel change â†’ history stale.
                float cx, cy, cz;
                {
                    float tmp[3];
                    hlslpp::store(tmp, g_app.camera.position);
                    cx = tmp[0];
                    cy = tmp[1];
                    cz = tmp[2];
                }
                float dx = cx - g_app.lastGodrayCamPos[0];
                float dy = cy - g_app.lastGodrayCamPos[1];
                float dz = cz - g_app.lastGodrayCamPos[2];
                float delta = std::sqrt(dx * dx + dy * dy + dz * dz);
                float boost = std::min(1.0f, delta / 2.0f); // 2 units/frame = no history
                ps.godrayEmaAlpha = std::max(g_app.godrayEmaAlpha, boost);
                g_app.lastGodrayCamPos[0] = cx;
                g_app.lastGodrayCamPos[1] = cy;
                g_app.lastGodrayCamPos[2] = cz;
            }
            ps.godrayTint[0] = g_app.godrayTint[0];
            ps.godrayTint[1] = g_app.godrayTint[1];
            ps.godrayTint[2] = g_app.godrayTint[2];
            ps.splatRadius = g_app.splatRadius;
            ps.taa = g_app.taa;
            ps.sunDir[0] = sunDir[0];
            ps.sunDir[1] = sunDir[1];
            ps.sunDir[2] = sunDir[2];
            ps.sunIntensity = exp2f(g_app.sunIntensityEV);
            ps.sunShadows = g_app.sunShadows;
            ps.shadowCascades = g_app.shadowCascades;
            {
                const int sizes[] = {512, 1024, 2048, 4096};
                ps.shadowMapSize = sizes[std::clamp(g_app.shadowMapSizeIdx, 0, 3)];
            }
            ps.shadowBias = g_app.shadowBias;
            ps.shadowCullFront = g_app.shadowCullFront;
            ps.shadowForceRebuild = g_app.shadowForceRebuild;
            ps.shadowLod = g_app.shadowLodIdx - 1; // 0 -> Auto (-1)
            ps.shadowBlur = g_app.shadowBlur;
            ps.lwShowBounds = g_app.lwShowBounds;
            ps.cheapAO = g_app.cheapAO;
            ps.aoStrength = g_app.aoStrength;
            ps.ambient = g_app.ambient;
            ps.aoFadeUnits = g_app.aoFadeUnits;
            ps.aoPushTexels = g_app.aoPushTexels;
            ps.exposure = exp2f(g_app.exposureEV);
            ps.roughness = g_app.roughness;
            if (g_app.renderer.HasLwWorld())
            {
                g_app.renderer.DrawLwScene(g_app.camera, ps);
            }
        }
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_app.renderer.CommandList());
        g_app.renderer.EndFrame(g_app.vsync);

        MicroProfileFlip();

        auto end = std::chrono::high_resolution_clock::now();
        g_app.cpuFrameMs = std::chrono::duration<double, std::milli>(end - now).count();
    }

    // MicroProfileShutdown joins worker threads (web server, context-switch ETW
    // tracer, GPU timers). On Windows the ETW unregister can take seconds.
    // Skip it â€” the OS reclaims sockets/threads at process exit.

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    // Stop loader thread.
    if (g_app.loaderThread.joinable())
    {
        g_app.loaderQuit.store(true);
        g_app.loaderCv.notify_all();
        g_app.loaderThread.join();
    }
    g_app.renderer.Shutdown();
    DestroyWindow(hwnd);
    return 0;
}
