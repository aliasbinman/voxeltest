// main_xbox.cpp — minimal Scarlett entry. Brings up renderer, runs a
// clear+present loop. No window, no imgui, no input yet. First-boot smoke test:
// confirm device + RegisterFrameEventsX + PresentX pipeline lights the TV.

// renderer.h must be first — pulls d3d12_xs.h which fights any earlier DXGI.
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "renderer.h"
#include <windows.h>

int WINAPI wWinMain(_In_ HINSTANCE, _In_opt_ HINSTANCE, _In_ LPWSTR, _In_ int)
{
    Renderer renderer;
    if (!renderer.Init(nullptr, -1))
    {
        OutputDebugStringA("[xbox] Renderer::Init failed\n");
        return 1;
    }

    // Pulsing clear color so we can confirm frame loop is live on the devkit.
    uint32_t frame = 0;
    for (;;)
    {
        ++frame;
        // Bright magenta to rule out gamma/dim color issues.
        float clear[4] = { 1.0f, 0.0f, 1.0f, 1.0f };
        renderer.BeginFrame(clear, false, false);
        renderer.EndFrame(true);
    }
    return 0;
}
