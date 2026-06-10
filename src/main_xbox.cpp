// main_xbox.cpp — Scarlett entry. Identical render path to PC: same LW world,
// same DrawSceneParams (default-constructed → matches PC g_app defaults),
// same shaders (compiled to .cso at build, loaded at startup). Camera is
// driven by GameInput controller (left stick = move, right stick = look,
// LB/RB = down/up, RT = sprint). No window, no imgui — pure render loop.

#define _CRT_SECURE_NO_WARNINGS
#include "renderer.h"
#include "camera.h"
#include "lodworld.h"
#include <windows.h>
#include <GameInput.h>
#include <wrl/client.h>
#include <string>
#include <algorithm>
#include <cmath>

using XComPtr = Microsoft::WRL::ComPtr<IGameInput>;
using XReadPtr = Microsoft::WRL::ComPtr<IGameInputReading>;

static float Deadzone(float v, float dz = 0.15f)
{
    if (v >  dz) return (v - dz) / (1.0f - dz);
    if (v < -dz) return (v + dz) / (1.0f - dz);
    return 0.0f;
}

int WINAPI wWinMain(_In_ HINSTANCE, _In_opt_ HINSTANCE, _In_ LPWSTR, _In_ int)
{
    Renderer renderer;
    if (!renderer.Init(nullptr, -1))
    {
        OutputDebugStringA("[xbox] Renderer::Init failed\n");
        return 1;
    }

    lw::World world;
    std::string err;
    const char* lwPath = "kingslanding_sky.lw";
    bool haveWorld = false;
    if (!lw::LoadWorld(lwPath, world, err))
    {
        OutputDebugStringA(("[xbox] LoadWorld failed: " + err + "\n").c_str());
    }
    else
    {
        renderer.PrepLwWorld(world);
        if (!renderer.UploadLwWorld(world))
            OutputDebugStringA("[xbox] UploadLwWorld failed\n");
        else
        {
            OutputDebugStringA("[xbox] LW world uploaded\n");
            haveWorld = true;
        }
    }

    // Camera = PC default. Camera struct defaults match what PC starts with on
    // first run (no settings persistence for cam pos). Gamepad drives motion.
    Camera cam;

    // Same DrawSceneParams as PC. Values copied from main.cpp AppState
    // initial values (not DrawSceneParams struct defaults — PC tunes a bunch
    // of fog/godray/sun fields at AppState init time).
    DrawSceneParams ps{};
    ps.splatRadius      = 1;
    ps.taa              = true;
    ps.fogColor[0]      = 0.55f;
    ps.fogColor[1]      = 0.60f;
    ps.fogColor[2]      = 0.70f;
    ps.fogDensity       = 0.0004f;
    ps.heightFogDensity = 4.5f;
    ps.heightFogFalloff = 0.05f;
    ps.heightFogStart   = -6.5f;
    ps.godrayStrength   = 0.55f;
    ps.godrayAngleDeg   = 10.0f;
    ps.godrayEmaAlpha   = 0.15f;
    ps.godrayTint[0]    = 1.00f;
    ps.godrayTint[1]    = 0.85f;
    ps.godrayTint[2]    = 0.45f;
    // Sun pitch=10°, yaw=63° → world-space dir.
    {
        const float kDeg2Rad = 3.14159265358979f / 180.0f;
        float pr = 10.0f * kDeg2Rad, yr = 63.0f * kDeg2Rad;
        ps.sunDir[0] = cosf(pr) * sinf(yr);
        ps.sunDir[1] = sinf(pr);
        ps.sunDir[2] = cosf(pr) * cosf(yr);
    }
    ps.sunIntensity     = 1.0f;     // 2^0 EV
    ps.exposure         = 1.0f;     // 2^0 EV
    ps.roughness        = 0.6f;
    ps.sunShadows       = true;
    ps.shadowCascades   = 1;
    ps.shadowMapSize    = 2048;
    ps.shadowBias       = 0.00015f;
    ps.aoFadeUnits      = 16.0f;
    ps.aoPushTexels     = 1.0f;
    ps.aoStrength       = 1.0f;

    XComPtr gameInput;
    if (FAILED(GameInputCreate(&gameInput)))
        OutputDebugStringA("[xbox] GameInputCreate failed — camera will be static\n");

    LARGE_INTEGER freq{}, prev{};
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&prev);

    for (;;)
    {
        LARGE_INTEGER now{};
        QueryPerformanceCounter(&now);
        float dt = (float)(now.QuadPart - prev.QuadPart) / (float)freq.QuadPart;
        prev = now;
        if (dt > 0.1f) dt = 0.1f;

        if (gameInput)
        {
            XReadPtr reading;
            if (SUCCEEDED(gameInput->GetCurrentReading(GameInputKindGamepad, nullptr, &reading)))
            {
                GameInputGamepadState s{};
                reading->GetGamepadState(&s);

                float lx = Deadzone(s.leftThumbstickX);
                float ly = Deadzone(s.leftThumbstickY);
                float rx = Deadzone(s.rightThumbstickX);
                float ry = Deadzone(s.rightThumbstickY);

                // Look: right stick. Sens ~ 2 rad/s at full deflection.
                cam.yaw   += rx * 2.0f * dt;
                cam.pitch += ry * 2.0f * dt;
                const float lim = 1.55f;
                cam.pitch = std::clamp(cam.pitch, -lim, lim);

                // Move speed: triggers scale. RT = sprint, LT = creep.
                float speedMul = 1.0f + s.rightTrigger * 8.0f - s.leftTrigger * 0.8f;
                float speed = cam.moveSpeed * speedMul;

                hlslpp::float3 fwd = cam.forward();
                hlslpp::float3 rt  = cam.right();
                hlslpp::float3 up  = hlslpp::float3(0.0f, 1.0f, 0.0f);

                cam.position = cam.position + fwd * (ly * speed * dt);
                cam.position = cam.position + rt  * (lx * speed * dt);

                // Bumpers: vertical.
                if (s.buttons & GameInputGamepadRightShoulder)
                    cam.position = cam.position + up * (speed * dt);
                if (s.buttons & GameInputGamepadLeftShoulder)
                    cam.position = cam.position - up * (speed * dt);

                // D-pad up/down: cycle move speed.
                static bool prevDU = false, prevDD = false;
                bool du = (s.buttons & GameInputGamepadDPadUp) != 0;
                bool dd = (s.buttons & GameInputGamepadDPadDown) != 0;
                if (du && !prevDU) cam.moveSpeed *= 2.0f;
                if (dd && !prevDD) cam.moveSpeed *= 0.5f;
                cam.moveSpeed = std::clamp(cam.moveSpeed, 1.0f, 5000.0f);
                prevDU = du; prevDD = dd;
            }
        }

        float clear[4] = { ps.fogColor[0], ps.fogColor[1], ps.fogColor[2], 1.0f };
        renderer.BeginFrame(clear, false, false);
        if (haveWorld) renderer.DrawLwScene(cam, ps);
        renderer.EndFrame(true);
    }
    return 0;
}
