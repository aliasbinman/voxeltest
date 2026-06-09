#pragma once
#include <hlsl++.h>
#include <cmath>

struct Camera
{
    hlslpp::float3 position = hlslpp::float3(0.0f, 50.0f, -100.0f);
    float yaw = 0.0f;
    float pitch = 0.0f;
    float fovDeg = 70.0f;
    float nearZ = 0.1f;
    float farZ = 5000.0f;
    float moveSpeed = 30.0f;
    float lookSens = 0.0025f;

    hlslpp::float3 forward() const
    {
        float cp = cosf(pitch), sp = sinf(pitch);
        float cy = cosf(yaw), sy = sinf(yaw);
        return hlslpp::float3(sy * cp, sp, cy * cp);
    }
    hlslpp::float3 right() const
    {
        return hlslpp::normalize(hlslpp::cross(hlslpp::float3(0.0f, 1.0f, 0.0f), forward()));
    }
    hlslpp::float4x4 view() const
    {
        return hlslpp::float4x4::look_at(position, position + forward(), hlslpp::float3(0.0f, 1.0f, 0.0f));
    }
    // Reverse-Z, infinite far. LH, DX clip space (z in [0,1] post-divide).
    // Maps view.z = nearZ -> clip.z/w = 1, view.z -> infty -> 0.
    // Row-major logical layout (vector*matrix). Depth func must be GREATER,
    // depth clear value must be 0.0.
    hlslpp::float4x4 proj(float aspect) const
    {
        const float fovRad = fovDeg * 3.14159265358979f / 180.0f;
        const float h = 1.0f / tanf(fovRad * 0.5f);
        const float w = h / aspect;
        return hlslpp::float4x4(
            w, 0.0f, 0.0f, 0.0f,
            0.0f, h, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, nearZ, 0.0f);
    }
};
